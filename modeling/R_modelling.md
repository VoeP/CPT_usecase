# Modelling CPT lithostratigraphy with R


### Functions

- The team was interested on which features were better at predicting
  the target variable.
- We should also show where our model predicts the whole series well,
  even if it misses the starting points with a few centimeters
- We should also show how far off the predicted starting points are from
  the true starting points (in cm)

``` r
# Tune model hyperparameters
tune_model <- function(workflow, folds, grid) {
  set.seed(SET_SEED)
  plan(multisession, workers = parallel::detectCores() - 4)
  tune_results <- tune_grid(
    workflow,
    resamples = folds,
    grid = grid,
    metrics = yardstick::metric_set(accuracy, mn_log_loss),
    control = control_grid(save_pred = TRUE)
  )
  plan(sequential) # back to sequential
  tune_results
}

# Fit best model from tuning results
fit_best_model <- function(workflow,
                           tune_results,
                           train_data,
                           metric = "mn_log_loss") {
  finalize_workflow(workflow, select_best(tune_results,
                                          metric = metric)) |>
    fit(data = train_data)
}


per_class_metrics <- function(pred_dt,
                              truth_col = "lithostrat_id",
                              estimate_col = ".pred_class") {
  lvls <- levels(pred_dt[[truth_col]])
  rows <- lapply(lvls, function(cls) {
    data <- pred_dt |>
      dplyr::mutate(
        truth_bin = factor(ifelse(.data[[truth_col]] == cls, cls, "other"),
                           levels = c("other", cls)),
        pred_bin = factor(ifelse(.data[[estimate_col]] == cls, cls, "other"),
                          levels = c("other", cls))
      )
    tibble::tibble(
      lithostrat_id = cls,
      support = sum(data$truth_bin == cls),
      precision = yardstick::precision(
        data,
        truth = truth_bin,
        estimate = pred_bin,
        event_level = "second"
      )$.estimate,
      recall = yardstick::recall(
        data,
        truth = truth_bin,
        estimate = pred_bin,
        event_level = "second"
      )$.estimate,
      specificity = yardstick::spec(
        data,
        truth = truth_bin,
        estimate = pred_bin,
        event_level = "second"
      )$.estimate,
      accuracy = yardstick::accuracy(data, truth = truth_bin,
                                     estimate = pred_bin)$.estimate
    )
  })
  dplyr::bind_rows(rows)
}


evaluate_model <- function(fitted_model,
                           test_data,
                           train_data,
                           id_col = "sondering_id",
                           cols_to_include = c("sondering_id", 
                                               "lithostrat_id", 
                                               "depth_bin")) {
  # Predict
  preds_class <- predict(fitted_model,
                         new_data = test_data)
  preds_prob <- predict(fitted_model,
                        new_data = test_data,
                        type = "prob")
  
  # Combine predictions with ID and truth
  pred_dt <- bind_cols(test_data[, .SD, 
                                 .SDcols = cols_to_include],
                       preds_class, preds_prob)
  
  # Align factor levels with training data
  if (!is.factor(train_data$lithostrat_id)) {
    train_data[, lithostrat_id := factor(lithostrat_id)]
  }
  pred_dt[, lithostrat_id := factor(lithostrat_id, 
                                    levels = levels(train_data$lithostrat_id))]
  pred_dt[, .pred_class := factor(.pred_class, 
                                  levels = levels(train_data$lithostrat_id))]
  
  # Collect probability columns in the correct order of levels
  lvl <- levels(train_data$lithostrat_id)
  prob_cols <- paste0(".pred_", lvl)
  prob_cols <- prob_cols[prob_cols %in% names(pred_dt)]
  
  # Metrics
  acc_tbl <- yardstick::accuracy(pred_dt,
                                 truth = lithostrat_id, estimate = .pred_class)
  # balc_tbl <- yardstick::bal_accuracy(pred_dt, truth = lithostrat_id, estimate = .pred_class)
  
  # mn_log_loss (multiclass needs all .pred_* columns)
  if (length(prob_cols) > 0L) {
    mnll_tbl <- rlang::exec(
      yardstick::mn_log_loss,
      data  = pred_dt,
      truth = rlang::expr(lithostrat_id),
      !!!rlang::syms(prob_cols)
    )
  } else {
    mnll_tbl <- NULL
    warning("No .pred_* columns found; mn_log_loss skipped.")
  }
  
  # roc_auc (handle binary vs multiclass explicitly)
  if (length(lvl) == 2L) {
    pos <- lvl[2L] # treat second level as the 'event'
    pos_col <- paste0(".pred_", pos)
    if (pos_col %in% names(pred_dt)) {
      roc_tbl <- yardstick::roc_auc(
        pred_dt,
        truth = lithostrat_id,
        estimate = !!rlang::sym(pos_col),
        event_level = "second"
      )
    } else {
      roc_tbl <- NULL
      warning("Binary AUC skipped: probability column
              for positive class not found.")
    }
  } else if (length(prob_cols) > 0L) {
    roc_tbl <- rlang::exec(
      yardstick::roc_auc,
      data = pred_dt,
      truth = rlang::expr(lithostrat_id),
      !!!rlang::syms(prob_cols),
      estimator = "macro_weighted"
    )
  } else {
    roc_tbl <- NULL
    warning("Multiclass AUC skipped: no .pred_* columns found.")
  }
  
  metrics <- dplyr::bind_rows(acc_tbl, mnll_tbl, roc_tbl)
  data.table::setDT(metrics)
  
  # Confusion matrix
  cm <- yardstick::conf_mat(pred_dt, 
                            truth = lithostrat_id,
                            estimate = .pred_class)
  
  # Per-class metrics
  per_class_df <- per_class_metrics(pred_dt)
  
  list(
    predictions = pred_dt,
    metrics = metrics,
    confusion_matrix = cm,
    per_class_metrics = per_class_df
  )
}

# Convert confusion matrix to data frame for kable
conf_mat_to_df <- function(cm) {
  as.data.frame.matrix(cm$table)
}
```

``` r
library(tidyverse)
library(data.table)
library(arrow)
library(here)
library(xgboost)
library(knitr)
library(zoo)
library(tidymodels)
library(bonsai)
library(future)
library(furrr)
# library(lightgbm)  # Fo

data_folder <- here("data")
results_folder <- here("results")

cat("Data folder:", data_folder, "\n")
```

    Data folder: /home/mburu/projects/uhasselt/CPT_usecase/data 

``` r
cat("Results folder:", results_folder, "\n")
```

    Results folder: /home/mburu/projects/uhasselt/CPT_usecase/results 

``` r
list.files(results_folder)
```

     [1] "cpt_features_true_0.6_42_additive.csv"      
     [2] "cpt_features_true_0.6_42_multiplicative.csv"
     [3] "cpt_ids_true_0.6_42_additive.csv"           
     [4] "cpt_ids_true_0.6_42_multiplicative.csv"     
     [5] "split_res.json"                             
     [6] "split_res.pkl"                              
     [7] "test_binned_true_0.6_42_additive.csv"       
     [8] "test_binned_true_0.6_42_multiplicative.csv" 
     [9] "train_binned_true_0.6_42_additive.csv"      
    [10] "train_binned_true_0.6_42_multiplicative.csv"

``` r
train_dt <- fread(here(
    results_folder,
    "train_binned_true_0.6_42_multiplicative.csv"
))
test_dt <- fread(here(
    results_folder,
    "test_binned_true_0.6_42_multiplicative.csv"
))
# factor lithostrat_id
# remove Onbekend
segments_oi <- c(
    "Quartair",
    "Diest",
    "Bolderberg",
    "Sint_Huibrechts_Hern",
    "Ursel",
    "Asse",
    "Wemmel",
    "Lede",
    "Brussel",
    "Merelbeke",
    "Kwatrecht",
    "Mont_Panisel",
    "Aalbeke",
    "Mons_en_Pevele"
)
train_dt <- train_dt[lithostrat_id %in% segments_oi]
test_dt <- test_dt[lithostrat_id %in% segments_oi]
train_dt <- train_dt[lithostrat_id != "Onbekend"]
test_dt <- test_dt[lithostrat_id != "Onbekend"]
# levels_litho <- unique(c(
#     train_dt$lithostrat_id,
#     test_dt$lithostrat_id
# ))
train_dt[, lithostrat_id := factor(lithostrat_id, levels = segments_oi)]
test_dt[, lithostrat_id := factor(lithostrat_id, levels = segments_oi)]
```

``` r
id_col <- "sondering_id"
depth_col <- "depth_bin"
label_col <- "lithostrat_id"
# Formula
nms_feat <- setdiff(names(dt), c(
    id_col, "lithostrat_id",
    depth_col, "diepte"
))
model_formula <- as.formula(paste(
    "lithostrat_id ~",
    paste(c(nms_feat, id_col),
        collapse = " + "
    )
))

rm_cols <- c(depth_col)
# Shared recipe
base_recipe <- recipe(lithostrat_id ~ .,
    data = train_dt[, .SD, .SDcols = !rm_cols]
) |>
    update_role(sondering_id, new_role = "group_id") |>
    step_rm(sondering_id) |>
    step_impute_mean(all_predictors()) |>
    step_zv(all_predictors()) |>
    step_nzv(all_predictors())
# Model specifications
# ADD NEW MODELS HERE - that's the ONLY place you need to change!
model_specs <- list(
    xgb = boost_tree(
        trees = tune(),
        tree_depth = tune(),
        learn_rate = tune(),
        loss_reduction = tune(),
        min_n = tune(),
        mtry = tune(),
        sample_size = tune()
    ) |> set_mode("classification") |>
        set_engine("xgboost"),
    rf = rand_forest(
        trees = tune(),
        mtry = tune(),
        min_n = tune()
    ) |>
        set_mode("classification") |>
        set_engine("ranger",
            importance = "permutation",
            splitrule = "extratrees"
        ),
    lgbm = boost_tree(
        trees = tune(),
        tree_depth = tune(),
        learn_rate = tune(),
        loss_reduction = tune(),
        min_n = tune(),
        mtry = tune(),
        sample_size = tune()
    ) |> set_mode("classification") |>
        set_engine("lightgbm")
)

# Workflows
workflows <- lapply(model_specs, function(spec) {
    workflow() |>
        add_recipe(base_recipe) |>
        add_model(spec)
})

# CV folds
set.seed(SET_SEED)
folds <- group_vfold_cv(train_dt,
    group = sondering_id,
    v = 10
)
```

## Tune and train the models

``` r
# Generate parameter grids (automatically for all models)

grids <- lapply(names(workflows), function(model_name) {
    if (model_name == "rf") {
        extract_parameter_set_dials(workflows[[model_name]]) |>
            finalize(train_dt |> dplyr::select(-lithostrat_id, -depth_bin)) |>
            grid_latin_hypercube(size = size_tune)
    } else {
        extract_parameter_set_dials(workflows[[model_name]]) |>
            finalize(train_dt |> dplyr::select(-lithostrat_id, -depth_bin)) |>
            grid_latin_hypercube(size = size_tune)
    }
})
names(grids) <- names(workflows)

# Train and evaluate ALL models in parallel
# Train and evaluate ALL models sequentially
results <- list()
library(tictoc)
tic()
for (model_name in names(workflows)) {
    cat("\nTraining", toupper(model_name), "\n")

    # Tune hyperparameters
    cat("Tuning hyperparameters...\n")

    # paralize this step if needed
    # library(future)
    # plan(multisession, workers = parallel::detectCores() - 4)

    tune_res <- tune_model(
        workflows[[model_name]],
        folds, grids[[model_name]]
    )
    # plan(sequential)  # back to sequential

    #  Fit best model
    cat("Fitting best model...\n")
    best_model <- fit_best_model(
        workflows[[model_name]],
        tune_res, train_dt
    )

    # Evaluate on test set
    cat("Evaluating on test set...\n")
    eval_res <- evaluate_model(best_model,
        test_dt, train_dt,
        id_col = "sondering_id"
    )
    eval_train <- evaluate_model(best_model,
        train_dt, train_dt,
        id_col = "sondering_id"
    )

    # Add model name to metrics
    eval_res$metrics[, model := model_name]
    eval_train$metrics[, model := model_name]

    # Store results
    results[[model_name]] <- list(
        tune_results = tune_res,
        fitted_model = best_model,
        predictions = eval_res$predictions,
        metrics = eval_res$metrics,
        confusion_matrix = eval_res$confusion_matrix,
        per_class_metrics = eval_res$per_class_metrics,
        train_metrics = eval_train$metrics
    )

    cat("Completed\n")
}
```

Training XGB Tuning hyperparameters… Fitting best model… Evaluating on
test set… Completed

Training RF Tuning hyperparameters… Fitting best model… Evaluating on
test set… Completed

Training LGBM Tuning hyperparameters… Fitting best model… Evaluating on
test set… Completed

``` r
toc()
```

3031.179 sec elapsed

## Results Summary

- For each model, we list the tuning metrics (mean and standard error)
  across resamples.
- Use this to spot which hyperparameters tend to perform better.
- Look for a good mix of high accuracy and stable (low std error)
  results.

``` r
# Display tuning metrics for all models
for (model_name in names(results)) {
  cat("\n####", toupper(model_name), "Tuning Metrics\n")
  print(kable(collect_metrics(results[[model_name]]$tune_results)))
}
```

#### XGB Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 5 | 297 | 5 | 3 | 0.0094634 | 0.0000000 | 0.4129629 | accuracy | multiclass | 0.7000843 | 10 | 0.0206429 | pre0_mod01_post0 |
| 5 | 297 | 5 | 3 | 0.0094634 | 0.0000000 | 0.4129629 | mn_log_loss | multiclass | 1.0540058 | 10 | 0.0419748 | pre0_mod01_post0 |
| 11 | 420 | 29 | 15 | 0.0023303 | 2.6422084 | 0.4781826 | accuracy | multiclass | 0.6560692 | 10 | 0.0257333 | pre0_mod02_post0 |
| 11 | 420 | 29 | 15 | 0.0023303 | 2.6422084 | 0.4781826 | mn_log_loss | multiclass | 1.5761103 | 10 | 0.0328561 | pre0_mod02_post0 |
| 25 | 887 | 13 | 9 | 0.0012530 | 0.1911300 | 0.1985896 | accuracy | multiclass | 0.6603476 | 10 | 0.0240409 | pre0_mod03_post0 |
| 25 | 887 | 13 | 9 | 0.0012530 | 0.1911300 | 0.1985896 | mn_log_loss | multiclass | 1.5164222 | 10 | 0.0340523 | pre0_mod03_post0 |
| 34 | 1958 | 38 | 8 | 0.0419831 | 0.0000000 | 0.2979045 | accuracy | multiclass | 0.6637173 | 10 | 0.0199412 | pre0_mod04_post0 |
| 34 | 1958 | 38 | 8 | 0.0419831 | 0.0000000 | 0.2979045 | mn_log_loss | multiclass | 1.0242481 | 10 | 0.0624638 | pre0_mod04_post0 |
| 37 | 1550 | 8 | 13 | 0.0266249 | 0.0070468 | 0.5619981 | accuracy | multiclass | 0.7274204 | 10 | 0.0126541 | pre0_mod05_post0 |
| 37 | 1550 | 8 | 13 | 0.0266249 | 0.0070468 | 0.5619981 | mn_log_loss | multiclass | 0.8820164 | 10 | 0.0635001 | pre0_mod05_post0 |
| 49 | 1081 | 17 | 6 | 0.0043163 | 0.0627205 | 0.1260413 | accuracy | multiclass | 0.6567479 | 10 | 0.0212593 | pre0_mod06_post0 |
| 49 | 1081 | 17 | 6 | 0.0043163 | 0.0627205 | 0.1260413 | mn_log_loss | multiclass | 1.1233974 | 10 | 0.0530927 | pre0_mod06_post0 |
| 56 | 646 | 15 | 11 | 0.2538757 | 0.0000034 | 0.9043393 | accuracy | multiclass | 0.7116038 | 10 | 0.0149449 | pre0_mod07_post0 |
| 56 | 646 | 15 | 11 | 0.2538757 | 0.0000034 | 0.9043393 | mn_log_loss | multiclass | 1.0248207 | 10 | 0.0779051 | pre0_mod07_post0 |
| 68 | 1247 | 25 | 5 | 0.0162167 | 0.0000099 | 0.9103769 | accuracy | multiclass | 0.7054475 | 10 | 0.0111414 | pre0_mod08_post0 |
| 68 | 1247 | 25 | 5 | 0.0162167 | 0.0000099 | 0.9103769 | mn_log_loss | multiclass | 0.8748139 | 10 | 0.0527903 | pre0_mod08_post0 |
| 76 | 1692 | 24 | 2 | 0.0643893 | 0.0000001 | 0.8081360 | accuracy | multiclass | 0.7011324 | 10 | 0.0119862 | pre0_mod09_post0 |
| 76 | 1692 | 24 | 2 | 0.0643893 | 0.0000001 | 0.8081360 | mn_log_loss | multiclass | 0.9836961 | 10 | 0.0636578 | pre0_mod09_post0 |
| 85 | 38 | 33 | 10 | 0.1392561 | 0.0006256 | 0.7269481 | accuracy | multiclass | 0.7036978 | 10 | 0.0139285 | pre0_mod10_post0 |
| 85 | 38 | 33 | 10 | 0.1392561 | 0.0006256 | 0.7269481 | mn_log_loss | multiclass | 0.8971837 | 10 | 0.0482770 | pre0_mod10_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    1 |  1322 |    22 | accuracy    | multiclass | 0.6939822 |  10 | 0.0226224 | pre0_mod01_post0 |
|    1 |  1322 |    22 | mn_log_loss | multiclass | 1.0176903 |  10 | 0.0475577 | pre0_mod01_post0 |
|   13 |  1782 |    28 | accuracy    | multiclass | 0.7302628 |  10 | 0.0175754 | pre0_mod02_post0 |
|   13 |  1782 |    28 | mn_log_loss | multiclass | 0.8276038 |  10 | 0.0446150 | pre0_mod02_post0 |
|   26 |   988 |    20 | accuracy    | multiclass | 0.7357755 |  10 | 0.0167944 | pre0_mod03_post0 |
|   26 |   988 |    20 | mn_log_loss | multiclass | 0.7896456 |  10 | 0.0442050 | pre0_mod03_post0 |
|   33 |   637 |    32 | accuracy    | multiclass | 0.7294438 |  10 | 0.0185401 | pre0_mod04_post0 |
|   33 |   637 |    32 | mn_log_loss | multiclass | 0.8117043 |  10 | 0.0423575 | pre0_mod04_post0 |
|   40 |  1158 |     6 | accuracy    | multiclass | 0.7394609 |  10 | 0.0171767 | pre0_mod05_post0 |
|   40 |  1158 |     6 | mn_log_loss | multiclass | 0.7690744 |  10 | 0.0501847 | pre0_mod05_post0 |
|   49 |   437 |    10 | accuracy    | multiclass | 0.7395637 |  10 | 0.0170451 | pre0_mod06_post0 |
|   49 |   437 |    10 | mn_log_loss | multiclass | 0.7745451 |  10 | 0.0507943 | pre0_mod06_post0 |
|   58 |  1546 |    15 | accuracy    | multiclass | 0.7383024 |  10 | 0.0169083 | pre0_mod07_post0 |
|   58 |  1546 |    15 | mn_log_loss | multiclass | 0.7795947 |  10 | 0.0441412 | pre0_mod07_post0 |
|   67 |    59 |     9 | accuracy    | multiclass | 0.7331267 |  10 | 0.0184292 | pre0_mod08_post0 |
|   67 |    59 |     9 | mn_log_loss | multiclass | 0.9323490 |  10 | 0.0820385 | pre0_mod08_post0 |
|   73 |  1968 |    37 | accuracy    | multiclass | 0.7219850 |  10 | 0.0197301 | pre0_mod09_post0 |
|   73 |  1968 |    37 | mn_log_loss | multiclass | 0.8158635 |  10 | 0.0417793 | pre0_mod09_post0 |
|   84 |   210 |    32 | accuracy    | multiclass | 0.7213089 |  10 | 0.0212993 | pre0_mod10_post0 |
|   84 |   210 |    32 | mn_log_loss | multiclass | 0.8215807 |  10 | 0.0484204 | pre0_mod10_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 5 | 932 | 17 | 7 | 0.0037343 | 10.5618371 | 0.6603661 | accuracy | multiclass | 0.7132801 | 10 | 0.0197993 | pre0_mod01_post0 |
| 5 | 932 | 17 | 7 | 0.0037343 | 10.5618371 | 0.6603661 | mn_log_loss | multiclass | 0.8797074 | 10 | 0.0527833 | pre0_mod01_post0 |
| 10 | 1142 | 32 | 4 | 0.0000001 | 0.0000999 | 0.2888438 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod02_post0 |
| 10 | 1142 | 32 | 4 | 0.0000001 | 0.0000999 | 0.2888438 | mn_log_loss | multiclass | 2.0640503 | 10 | 0.0528515 | pre0_mod02_post0 |
| 21 | 1655 | 38 | 4 | 0.0000000 | 0.0000001 | 0.1601417 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod03_post0 |
| 21 | 1655 | 38 | 4 | 0.0000000 | 0.0000001 | 0.1601417 | mn_log_loss | multiclass | 2.0644458 | 10 | 0.0528585 | pre0_mod03_post0 |
| 27 | 1551 | 23 | 1 | 0.0330461 | 0.0000040 | 0.5870954 | accuracy | multiclass | 0.7106640 | 10 | 0.0159217 | pre0_mod04_post0 |
| 27 | 1551 | 23 | 1 | 0.0330461 | 0.0000040 | 0.5870954 | mn_log_loss | multiclass | 0.9240160 | 10 | 0.0946895 | pre0_mod04_post0 |
| 43 | 159 | 8 | 12 | 0.0000152 | 0.0000000 | 0.9542964 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod05_post0 |
| 43 | 159 | 8 | 12 | 0.0000152 | 0.0000000 | 0.9542964 | mn_log_loss | multiclass | 2.0509979 | 10 | 0.0525689 | pre0_mod05_post0 |
| 47 | 680 | 3 | 6 | 0.0001697 | 0.0000132 | 0.4708015 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod06_post0 |
| 47 | 680 | 3 | 6 | 0.0001697 | 0.0000132 | 0.4708015 | mn_log_loss | multiclass | 1.7102498 | 10 | 0.0504837 | pre0_mod06_post0 |
| 58 | 1339 | 27 | 8 | 0.0000000 | 0.0022944 | 0.2209518 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod07_post0 |
| 58 | 1339 | 27 | 8 | 0.0000000 | 0.0022944 | 0.2209518 | mn_log_loss | multiclass | 2.0644608 | 10 | 0.0528585 | pre0_mod07_post0 |
| 66 | 418 | 10 | 14 | 0.0000017 | 0.0566046 | 0.4162612 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod08_post0 |
| 66 | 418 | 10 | 14 | 0.0000017 | 0.0566046 | 0.4162612 | mn_log_loss | multiclass | 2.0606411 | 10 | 0.0527485 | pre0_mod08_post0 |
| 76 | 394 | 19 | 14 | 0.0000000 | 0.2082535 | 0.7568843 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod09_post0 |
| 76 | 394 | 19 | 14 | 0.0000000 | 0.2082535 | 0.7568843 | mn_log_loss | multiclass | 2.0644426 | 10 | 0.0528580 | pre0_mod09_post0 |
| 81 | 1869 | 34 | 10 | 0.0014577 | 0.0000000 | 0.8909146 | accuracy | multiclass | 0.7199481 | 10 | 0.0162995 | pre0_mod10_post0 |
| 81 | 1869 | 34 | 10 | 0.0014577 | 0.0000000 | 0.8909146 | mn_log_loss | multiclass | 0.8749564 | 10 | 0.0668095 | pre0_mod10_post0 |

### Compare tuning results across models

- We summarize tuning results by model to see which family performs best
  overall.
- Higher accuracy and balanced accuracy indicate better performance
  across classes.
- This helps decide where to focus further tuning or feature work.

``` r
# Combine metrics from all models
combined_metrics <- bind_rows(lapply(names(results), function(model_name) {
  collect_metrics(results[[model_name]]$tune_results) |>
    mutate(model = model_name)
}))

comparison_summary <- combined_metrics |>
  group_by(model, .metric) |>
  summarize(mean = mean(mean), std_err = mean(std_err), .groups = "drop") |>
  arrange(.metric, desc(mean))

kable(comparison_summary)
```

| model | .metric     |      mean |   std_err |
|:------|:------------|----------:|----------:|
| rf    | accuracy    | 0.7283212 | 0.0186121 |
| xgb   | accuracy    | 0.6886268 | 0.0176273 |
| lgbm  | accuracy    | 0.4596146 | 0.0242237 |
| lgbm  | mn_log_loss | 1.6757968 | 0.0581510 |
| xgb   | mn_log_loss | 1.0956715 | 0.0530570 |
| rf    | mn_log_loss | 0.8339651 | 0.0496093 |

### Perfomance on training set

- We report training set metrics to check for overfitting.

``` r
# train_metrics from all models
train_metrics <- bind_rows(lapply(results, function(r) r$train_metrics))
kable(train_metrics)
```

| .metric     | .estimator     | .estimate | model |
|:------------|:---------------|----------:|:------|
| accuracy    | multiclass     | 0.9812476 | xgb   |
| mn_log_loss | multiclass     | 0.1359799 | xgb   |
| roc_auc     | macro_weighted | 0.9998134 | xgb   |
| accuracy    | multiclass     | 0.9992346 | rf    |
| mn_log_loss | multiclass     | 0.1418070 | rf    |
| roc_auc     | macro_weighted | 0.9999988 | rf    |
| accuracy    | multiclass     | 0.9959816 | lgbm  |
| mn_log_loss | multiclass     | 0.0859765 | lgbm  |
| roc_auc     | macro_weighted | 0.9999833 | lgbm  |

### performance on test set

- Final metrics reported on the held-out test boreholes (never seen
  during training).
- Confusion matrices show where predictions go wrong per class.
- Use this to understand practical performance and class-specific
  issues.

``` r
# Combined test metrics from all models
test_metrics <- bind_rows(lapply(results, function(r) r$metrics))
kable(test_metrics)
```

| .metric     | .estimator     | .estimate | model |
|:------------|:---------------|----------:|:------|
| accuracy    | multiclass     | 0.7166894 | xgb   |
| mn_log_loss | multiclass     | 0.8912685 | xgb   |
| roc_auc     | macro_weighted | 0.9495041 | xgb   |
| accuracy    | multiclass     | 0.7271487 | rf    |
| mn_log_loss | multiclass     | 0.7769903 | rf    |
| roc_auc     | macro_weighted | 0.9622073 | rf    |
| accuracy    | multiclass     | 0.7039563 | lgbm  |
| mn_log_loss | multiclass     | 0.9478924 | lgbm  |
| roc_auc     | macro_weighted | 0.9439026 | lgbm  |

### Confusion Matrices

``` r
for (model_name in names(results)) {
  cat("\n####", toupper(model_name), "Confusion Matrix\n")
  cm_df <- conf_mat_to_df(results[[model_name]]$confusion_matrix)
  print(kable(cm_df))
  cat("\n")
}
```

#### XGB Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 690 | 1 | 15 | 4 | 4 | 2 | 8 | 30 | 73 | 0 | 4 | 12 | 11 | 21 |
| Diest | 2 | 36 | 14 | 1 | 7 | 4 | 17 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 0 | 0 | 4 | 0 | 17 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 1 | 0 | 0 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 0 | 0 | 5 | 36 | 9 | 7 | 0 | 0 | 0 | 0 | 0 |
| Lede | 7 | 0 | 0 | 0 | 0 | 0 | 3 | 114 | 27 | 0 | 1 | 2 | 0 | 16 |
| Brussel | 51 | 0 | 0 | 0 | 0 | 0 | 4 | 20 | 280 | 0 | 1 | 7 | 0 | 12 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 23 | 12 | 4 | 2 |
| Mont_Panisel | 5 | 0 | 0 | 0 | 7 | 6 | 1 | 2 | 1 | 4 | 12 | 164 | 7 | 34 |
| Aalbeke | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 46 | 2 |
| Mons_en_Pevele | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 10 | 35 | 24 | 155 |

#### RF Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 679 | 3 | 21 | 5 | 4 | 2 | 9 | 26 | 50 | 0 | 1 | 10 | 10 | 14 |
| Diest | 1 | 34 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 1 | 0 | 0 | 0 | 20 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 10 | 0 | 10 | 16 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 1 | 0 | 0 | 5 | 40 | 18 | 3 | 0 | 0 | 0 | 0 | 0 |
| Lede | 16 | 0 | 0 | 0 | 0 | 0 | 3 | 85 | 13 | 0 | 1 | 4 | 0 | 14 |
| Brussel | 54 | 0 | 0 | 0 | 0 | 0 | 12 | 47 | 323 | 0 | 1 | 12 | 0 | 25 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 1 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 27 | 14 | 1 | 4 |
| Mont_Panisel | 2 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 0 | 2 | 14 | 166 | 11 | 33 |
| Aalbeke | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 54 | 4 |
| Mons_en_Pevele | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 1 | 8 | 26 | 16 | 147 |

#### LGBM Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 691 | 2 | 21 | 4 | 5 | 4 | 10 | 35 | 70 | 1 | 5 | 12 | 13 | 21 |
| Diest | 1 | 35 | 8 | 1 | 7 | 4 | 15 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 0 | 0 | 3 | 0 | 14 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 2 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 0 | 0 | 4 | 36 | 11 | 4 | 0 | 0 | 0 | 0 | 1 |
| Lede | 5 | 0 | 0 | 0 | 0 | 0 | 4 | 91 | 20 | 0 | 2 | 0 | 0 | 13 |
| Brussel | 57 | 0 | 0 | 0 | 0 | 0 | 5 | 40 | 290 | 0 | 0 | 10 | 3 | 24 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 17 | 1 | 0 |
| Mont_Panisel | 5 | 0 | 0 | 0 | 9 | 6 | 1 | 1 | 1 | 5 | 19 | 163 | 11 | 37 |
| Aalbeke | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 3 | 55 | 2 |
| Mons_en_Pevele | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 | 0 | 10 | 32 | 9 | 144 |

### Per-class metrics

- We report precision, recall, specificity, accuracy per lithostrat
  class.

``` r
# Combined per-class metrics from all models
for (model_name in names(results)) {
  cat("\n####", toupper(model_name), "Per-Class Metrics\n")
  per_class_df <- results[[model_name]]$per_class
  print(kable(per_class_df))
}
```

#### XGB Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.7885714 | 0.8961039 |   0.8705388 | 0.8794907 |
| Diest                |      37 | 0.4337349 | 0.9729730 |   0.9782609 | 0.9781719 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9845384 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9963537 | 0.9940882 |
| Ursel                |      36 | 0.5862069 | 0.4722222 |   0.9944521 | 0.9859027 |
| Asse                 |      38 | 0.7272727 | 0.2105263 |   0.9986118 | 0.9849932 |
| Wemmel               |      72 | 0.6206897 | 0.5000000 |   0.9896568 | 0.9736244 |
| Lede                 |     181 | 0.6705882 | 0.6298343 |   0.9722498 | 0.9440655 |
| Brussel              |     391 | 0.7466667 | 0.7161125 |   0.9474558 | 0.9063211 |
| Merelbeke            |      12 | 0.8750000 | 0.5833333 |   0.9995428 | 0.9972715 |
| Kwatrecht            |      52 | 0.5476190 | 0.4423077 |   0.9911504 | 0.9781719 |
| Mont_Panisel         |     237 | 0.6748971 | 0.6919831 |   0.9597350 | 0.9308777 |
| Aalbeke              |      92 | 0.8214286 | 0.5000000 |   0.9952539 | 0.9745339 |
| Mons_en_Pevele       |     242 | 0.6431535 | 0.6404959 |   0.9560552 | 0.9213279 |

#### RF Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.8141487 | 0.8818182 |   0.8915325 | 0.8881310 |
| Diest                |      37 | 0.8500000 | 0.9189189 |   0.9972248 | 0.9959072 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9845384 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9954421 | 0.9931787 |
| Ursel                |      36 | 0.8000000 | 0.5555556 |   0.9976884 | 0.9904502 |
| Asse                 |      38 | 0.4210526 | 0.4210526 |   0.9898195 | 0.9799909 |
| Wemmel               |      72 | 0.5882353 | 0.5555556 |   0.9868359 | 0.9727149 |
| Lede                 |     181 | 0.6250000 | 0.4696133 |   0.9747275 | 0.9331514 |
| Brussel              |     391 | 0.6814346 | 0.8260870 |   0.9164823 | 0.9004093 |
| Merelbeke            |      12 | 0.8000000 | 0.6666667 |   0.9990855 | 0.9972715 |
| Kwatrecht            |      52 | 0.5744681 | 0.5192308 |   0.9906847 | 0.9795362 |
| Mont_Panisel         |     237 | 0.7033898 | 0.7004219 |   0.9643221 | 0.9358799 |
| Aalbeke              |      92 | 0.8437500 | 0.5869565 |   0.9952539 | 0.9781719 |
| Mons_en_Pevele       |     242 | 0.6774194 | 0.6074380 |   0.9642310 | 0.9249659 |

#### LGBM Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.7729306 | 0.8974026 |   0.8579426 | 0.8717599 |
| Diest                |      37 | 0.4794521 | 0.9459459 |   0.9824237 | 0.9818099 |
| Bolderberg           |      34 | 0.0000000 | 0.0000000 |   0.9995381 | 0.9840837 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9972653 | 0.9949977 |
| Ursel                |      36 | 0.6086957 | 0.3888889 |   0.9958391 | 0.9859027 |
| Asse                 |      38 | 0.7500000 | 0.2368421 |   0.9986118 | 0.9854479 |
| Wemmel               |      72 | 0.6315789 | 0.5000000 |   0.9901269 | 0.9740791 |
| Lede                 |     181 | 0.6740741 | 0.5027624 |   0.9781962 | 0.9390632 |
| Brussel              |     391 | 0.6759907 | 0.7416880 |   0.9231195 | 0.8908595 |
| Merelbeke            |      12 | 0.8000000 | 0.3333333 |   0.9995428 | 0.9959072 |
| Kwatrecht            |      52 | 0.4705882 | 0.3076923 |   0.9916162 | 0.9754434 |
| Mont_Panisel         |     237 | 0.6317829 | 0.6877637 |   0.9515800 | 0.9231469 |
| Aalbeke              |      92 | 0.8730159 | 0.5978261 |   0.9962031 | 0.9795362 |
| Mons_en_Pevele       |     242 | 0.6889952 | 0.5950413 |   0.9667859 | 0.9258754 |
