# Modelling CPT lithostratigraphy with R


### Functions

- The team was interested on which features were better at predicting
  the target variable.
- We should also show where our model predicts the whole series well,
  even if it misses the starting points with a few centimeters
- We should also show how far off the predicted starting points are from
  the true starting points (in cm)
- trying to see where our model gets the whole series can this be an
  updated research question

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
train_dt[, unique(lithostrat_id)]
```

     [1] "Quartair"             "Mont_Panisel"         "Aalbeke"             
     [4] "Mons_en_Pevele"       "Maldegem"             "Brussel"             
     [7] "Onbekend"             "Ursel"                "Asse"                
    [10] "Wemmel"               "Bolderberg"           "Merelbeke"           
    [13] "Kwatrecht"            "Lede"                 "Antropogeen"         
    [16] "Diest"                "Sint_Huibrechts_Hern"

``` r
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

3032.175 sec elapsed

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
| 5 | 297 | 5 | 3 | 0.0094634 | 0.0000000 | 0.4129629 | accuracy | multiclass | 0.7022063 | 10 | 0.0195835 | pre0_mod01_post0 |
| 5 | 297 | 5 | 3 | 0.0094634 | 0.0000000 | 0.4129629 | mn_log_loss | multiclass | 1.0493659 | 10 | 0.0418985 | pre0_mod01_post0 |
| 11 | 420 | 29 | 15 | 0.0023303 | 2.6422084 | 0.4781826 | accuracy | multiclass | 0.6563237 | 10 | 0.0251451 | pre0_mod02_post0 |
| 11 | 420 | 29 | 15 | 0.0023303 | 2.6422084 | 0.4781826 | mn_log_loss | multiclass | 1.5726497 | 10 | 0.0330254 | pre0_mod02_post0 |
| 25 | 887 | 13 | 9 | 0.0012530 | 0.1911300 | 0.1985896 | accuracy | multiclass | 0.6593439 | 10 | 0.0229730 | pre0_mod03_post0 |
| 25 | 887 | 13 | 9 | 0.0012530 | 0.1911300 | 0.1985896 | mn_log_loss | multiclass | 1.5126070 | 10 | 0.0336780 | pre0_mod03_post0 |
| 34 | 1958 | 38 | 8 | 0.0419831 | 0.0000000 | 0.2979045 | accuracy | multiclass | 0.6666642 | 10 | 0.0196869 | pre0_mod04_post0 |
| 34 | 1958 | 38 | 8 | 0.0419831 | 0.0000000 | 0.2979045 | mn_log_loss | multiclass | 1.0239970 | 10 | 0.0632980 | pre0_mod04_post0 |
| 37 | 1550 | 8 | 13 | 0.0266249 | 0.0070468 | 0.5619981 | accuracy | multiclass | 0.7249447 | 10 | 0.0116852 | pre0_mod05_post0 |
| 37 | 1550 | 8 | 13 | 0.0266249 | 0.0070468 | 0.5619981 | mn_log_loss | multiclass | 0.8759102 | 10 | 0.0613457 | pre0_mod05_post0 |
| 49 | 1081 | 17 | 6 | 0.0043163 | 0.0627205 | 0.1260413 | accuracy | multiclass | 0.6611649 | 10 | 0.0208381 | pre0_mod06_post0 |
| 49 | 1081 | 17 | 6 | 0.0043163 | 0.0627205 | 0.1260413 | mn_log_loss | multiclass | 1.1181139 | 10 | 0.0530012 | pre0_mod06_post0 |
| 56 | 646 | 15 | 11 | 0.2538757 | 0.0000034 | 0.9043393 | accuracy | multiclass | 0.7133430 | 10 | 0.0139271 | pre0_mod07_post0 |
| 56 | 646 | 15 | 11 | 0.2538757 | 0.0000034 | 0.9043393 | mn_log_loss | multiclass | 1.0077684 | 10 | 0.0773353 | pre0_mod07_post0 |
| 68 | 1247 | 25 | 5 | 0.0162167 | 0.0000099 | 0.9103769 | accuracy | multiclass | 0.7024083 | 10 | 0.0123083 | pre0_mod08_post0 |
| 68 | 1247 | 25 | 5 | 0.0162167 | 0.0000099 | 0.9103769 | mn_log_loss | multiclass | 0.8696971 | 10 | 0.0536161 | pre0_mod08_post0 |
| 76 | 1692 | 24 | 2 | 0.0643893 | 0.0000001 | 0.8081360 | accuracy | multiclass | 0.7034568 | 10 | 0.0127158 | pre0_mod09_post0 |
| 76 | 1692 | 24 | 2 | 0.0643893 | 0.0000001 | 0.8081360 | mn_log_loss | multiclass | 0.9676415 | 10 | 0.0658539 | pre0_mod09_post0 |
| 85 | 38 | 33 | 10 | 0.1392561 | 0.0006256 | 0.7269481 | accuracy | multiclass | 0.7063387 | 10 | 0.0128166 | pre0_mod10_post0 |
| 85 | 38 | 33 | 10 | 0.1392561 | 0.0006256 | 0.7269481 | mn_log_loss | multiclass | 0.8942211 | 10 | 0.0466513 | pre0_mod10_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    1 |  1322 |    22 | accuracy    | multiclass | 0.6980494 |  10 | 0.0229167 | pre0_mod01_post0 |
|    1 |  1322 |    22 | mn_log_loss | multiclass | 1.0071894 |  10 | 0.0477305 | pre0_mod01_post0 |
|   13 |  1782 |    28 | accuracy    | multiclass | 0.7308354 |  10 | 0.0170808 | pre0_mod02_post0 |
|   13 |  1782 |    28 | mn_log_loss | multiclass | 0.8246680 |  10 | 0.0432792 | pre0_mod02_post0 |
|   26 |   988 |    20 | accuracy    | multiclass | 0.7396691 |  10 | 0.0172138 | pre0_mod03_post0 |
|   26 |   988 |    20 | mn_log_loss | multiclass | 0.7847761 |  10 | 0.0435440 | pre0_mod03_post0 |
|   33 |   637 |    32 | accuracy    | multiclass | 0.7274942 |  10 | 0.0184683 | pre0_mod04_post0 |
|   33 |   637 |    32 | mn_log_loss | multiclass | 0.8077580 |  10 | 0.0420345 | pre0_mod04_post0 |
|   40 |  1158 |     6 | accuracy    | multiclass | 0.7415758 |  10 | 0.0169014 | pre0_mod05_post0 |
|   40 |  1158 |     6 | mn_log_loss | multiclass | 0.7660332 |  10 | 0.0505253 | pre0_mod05_post0 |
|   49 |   437 |    10 | accuracy    | multiclass | 0.7428908 |  10 | 0.0159778 | pre0_mod06_post0 |
|   49 |   437 |    10 | mn_log_loss | multiclass | 0.7733942 |  10 | 0.0450667 | pre0_mod06_post0 |
|   58 |  1546 |    15 | accuracy    | multiclass | 0.7384983 |  10 | 0.0163823 | pre0_mod07_post0 |
|   58 |  1546 |    15 | mn_log_loss | multiclass | 0.7765649 |  10 | 0.0439357 | pre0_mod07_post0 |
|   67 |    59 |     9 | accuracy    | multiclass | 0.7358849 |  10 | 0.0172504 | pre0_mod08_post0 |
|   67 |    59 |     9 | mn_log_loss | multiclass | 0.9237516 |  10 | 0.0796525 | pre0_mod08_post0 |
|   73 |  1968 |    37 | accuracy    | multiclass | 0.7223534 |  10 | 0.0188076 | pre0_mod09_post0 |
|   73 |  1968 |    37 | mn_log_loss | multiclass | 0.8132765 |  10 | 0.0421253 | pre0_mod09_post0 |
|   84 |   210 |    32 | accuracy    | multiclass | 0.7274419 |  10 | 0.0170507 | pre0_mod10_post0 |
|   84 |   210 |    32 | mn_log_loss | multiclass | 0.8120708 |  10 | 0.0423082 | pre0_mod10_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 5 | 932 | 17 | 7 | 0.0037343 | 10.5618371 | 0.6603661 | accuracy | multiclass | 0.7146519 | 10 | 0.0192157 | pre0_mod01_post0 |
| 5 | 932 | 17 | 7 | 0.0037343 | 10.5618371 | 0.6603661 | mn_log_loss | multiclass | 0.8734304 | 10 | 0.0524855 | pre0_mod01_post0 |
| 10 | 1142 | 32 | 4 | 0.0000001 | 0.0000999 | 0.2888438 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod02_post0 |
| 10 | 1142 | 32 | 4 | 0.0000001 | 0.0000999 | 0.2888438 | mn_log_loss | multiclass | 2.0640519 | 10 | 0.0528521 | pre0_mod02_post0 |
| 21 | 1655 | 38 | 4 | 0.0000000 | 0.0000001 | 0.1601417 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod03_post0 |
| 21 | 1655 | 38 | 4 | 0.0000000 | 0.0000001 | 0.1601417 | mn_log_loss | multiclass | 2.0644458 | 10 | 0.0528585 | pre0_mod03_post0 |
| 27 | 1551 | 23 | 1 | 0.0330461 | 0.0000040 | 0.5870954 | accuracy | multiclass | 0.7145484 | 10 | 0.0150161 | pre0_mod04_post0 |
| 27 | 1551 | 23 | 1 | 0.0330461 | 0.0000040 | 0.5870954 | mn_log_loss | multiclass | 0.9175319 | 10 | 0.0936620 | pre0_mod04_post0 |
| 43 | 159 | 8 | 12 | 0.0000152 | 0.0000000 | 0.9542964 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod05_post0 |
| 43 | 159 | 8 | 12 | 0.0000152 | 0.0000000 | 0.9542964 | mn_log_loss | multiclass | 2.0509398 | 10 | 0.0525499 | pre0_mod05_post0 |
| 47 | 680 | 3 | 6 | 0.0001697 | 0.0000132 | 0.4708015 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod06_post0 |
| 47 | 680 | 3 | 6 | 0.0001697 | 0.0000132 | 0.4708015 | mn_log_loss | multiclass | 1.7093705 | 10 | 0.0506797 | pre0_mod06_post0 |
| 58 | 1339 | 27 | 8 | 0.0000000 | 0.0022944 | 0.2209518 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod07_post0 |
| 58 | 1339 | 27 | 8 | 0.0000000 | 0.0022944 | 0.2209518 | mn_log_loss | multiclass | 2.0644608 | 10 | 0.0528585 | pre0_mod07_post0 |
| 66 | 418 | 10 | 14 | 0.0000017 | 0.0566046 | 0.4162612 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod08_post0 |
| 66 | 418 | 10 | 14 | 0.0000017 | 0.0566046 | 0.4162612 | mn_log_loss | multiclass | 2.0606406 | 10 | 0.0527495 | pre0_mod08_post0 |
| 76 | 394 | 19 | 14 | 0.0000000 | 0.2082535 | 0.7568843 | accuracy | multiclass | 0.3503219 | 10 | 0.0271738 | pre0_mod09_post0 |
| 76 | 394 | 19 | 14 | 0.0000000 | 0.2082535 | 0.7568843 | mn_log_loss | multiclass | 2.0644426 | 10 | 0.0528580 | pre0_mod09_post0 |
| 81 | 1869 | 34 | 10 | 0.0014577 | 0.0000000 | 0.8909146 | accuracy | multiclass | 0.7245393 | 10 | 0.0146631 | pre0_mod10_post0 |
| 81 | 1869 | 34 | 10 | 0.0014577 | 0.0000000 | 0.8909146 | mn_log_loss | multiclass | 0.8695981 | 10 | 0.0657897 | pre0_mod10_post0 |

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
| rf    | accuracy    | 0.7304693 | 0.0178050 |
| xgb   | accuracy    | 0.6896195 | 0.0171680 |
| lgbm  | accuracy    | 0.4605993 | 0.0239112 |
| lgbm  | mn_log_loss | 1.6738912 | 0.0579343 |
| xgb   | mn_log_loss | 1.0891972 | 0.0529703 |
| rf    | mn_log_loss | 0.8289483 | 0.0480202 |

### Perfomance on training set

- We report training set metrics to check for overfitting.

``` r
# train_metrics from all models
train_metrics <- bind_rows(lapply(results, function(r) r$train_metrics))
kable(train_metrics)
```

| .metric     | .estimator     | .estimate | model |
|:------------|:---------------|----------:|:------|
| accuracy    | multiclass     | 0.9818217 | xgb   |
| mn_log_loss | multiclass     | 0.1351351 | xgb   |
| roc_auc     | macro_weighted | 0.9998118 | xgb   |
| accuracy    | multiclass     | 0.9992346 | rf    |
| mn_log_loss | multiclass     | 0.1419501 | rf    |
| roc_auc     | macro_weighted | 0.9999990 | rf    |
| accuracy    | multiclass     | 0.9961730 | lgbm  |
| mn_log_loss | multiclass     | 0.0862543 | lgbm  |
| roc_auc     | macro_weighted | 0.9999812 | lgbm  |

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
| accuracy    | multiclass     | 0.7203274 | xgb   |
| mn_log_loss | multiclass     | 0.8823687 | xgb   |
| roc_auc     | macro_weighted | 0.9508555 | xgb   |
| accuracy    | multiclass     | 0.7294225 | rf    |
| mn_log_loss | multiclass     | 0.7750683 | rf    |
| roc_auc     | macro_weighted | 0.9626825 | rf    |
| accuracy    | multiclass     | 0.7053206 | lgbm  |
| mn_log_loss | multiclass     | 0.9371423 | lgbm  |
| roc_auc     | macro_weighted | 0.9459833 | lgbm  |

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
| Quartair | 699 | 1 | 14 | 4 | 4 | 2 | 8 | 29 | 73 | 0 | 4 | 12 | 11 | 21 |
| Diest | 2 | 36 | 14 | 1 | 7 | 4 | 17 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 0 | 0 | 4 | 0 | 19 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 1 | 0 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 0 | 0 | 5 | 36 | 9 | 6 | 0 | 0 | 0 | 0 | 0 |
| Lede | 4 | 0 | 0 | 0 | 0 | 0 | 3 | 116 | 30 | 0 | 1 | 1 | 1 | 16 |
| Brussel | 47 | 0 | 0 | 0 | 0 | 0 | 4 | 19 | 277 | 0 | 1 | 6 | 0 | 12 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 23 | 12 | 4 | 2 |
| Mont_Panisel | 5 | 0 | 1 | 0 | 5 | 6 | 1 | 2 | 3 | 5 | 12 | 161 | 5 | 32 |
| Aalbeke | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 48 | 3 |
| Mons_en_Pevele | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 10 | 40 | 23 | 156 |

#### RF Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 680 | 4 | 22 | 5 | 4 | 1 | 9 | 27 | 49 | 0 | 1 | 11 | 10 | 12 |
| Diest | 1 | 33 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 1 | 0 | 0 | 0 | 20 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 10 | 0 | 10 | 15 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| Wemmel | 1 | 0 | 1 | 0 | 0 | 6 | 40 | 19 | 1 | 0 | 0 | 0 | 0 | 0 |
| Lede | 14 | 0 | 0 | 0 | 0 | 0 | 3 | 88 | 14 | 0 | 1 | 4 | 0 | 14 |
| Brussel | 56 | 0 | 0 | 0 | 0 | 0 | 11 | 44 | 325 | 0 | 1 | 12 | 0 | 26 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 1 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 27 | 14 | 2 | 4 |
| Mont_Panisel | 2 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 0 | 2 | 14 | 169 | 11 | 33 |
| Aalbeke | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 51 | 4 |
| Mons_en_Pevele | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | 8 | 22 | 18 | 148 |

#### LGBM Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 690 | 2 | 22 | 4 | 5 | 4 | 10 | 34 | 69 | 2 | 5 | 12 | 13 | 20 |
| Diest | 1 | 35 | 8 | 1 | 7 | 4 | 15 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 0 | 0 | 3 | 0 | 14 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 1 | 0 | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 0 | 0 | 4 | 36 | 10 | 4 | 0 | 0 | 0 | 0 | 1 |
| Lede | 5 | 0 | 0 | 0 | 0 | 0 | 4 | 91 | 22 | 0 | 2 | 1 | 0 | 12 |
| Brussel | 58 | 0 | 0 | 0 | 0 | 0 | 5 | 42 | 290 | 0 | 0 | 9 | 3 | 25 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 17 | 1 | 1 |
| Mont_Panisel | 5 | 0 | 0 | 0 | 8 | 6 | 1 | 0 | 1 | 5 | 19 | 160 | 10 | 31 |
| Aalbeke | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 3 | 55 | 2 |
| Mons_en_Pevele | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 8 | 35 | 10 | 150 |

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
| Quartair             |     770 | 0.7925170 | 0.9077922 |   0.8719384 | 0.8844930 |
| Diest                |      37 | 0.4337349 | 0.9729730 |   0.9782609 | 0.9781719 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9845384 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9963537 | 0.9940882 |
| Ursel                |      36 | 0.5937500 | 0.5277778 |   0.9939898 | 0.9863574 |
| Asse                 |      38 | 0.7000000 | 0.1842105 |   0.9986118 | 0.9845384 |
| Wemmel               |      72 | 0.6315789 | 0.5000000 |   0.9901269 | 0.9740791 |
| Lede                 |     181 | 0.6744186 | 0.6408840 |   0.9722498 | 0.9449750 |
| Brussel              |     391 | 0.7568306 | 0.7084399 |   0.9507743 | 0.9076853 |
| Merelbeke            |      12 | 0.8571429 | 0.5000000 |   0.9995428 | 0.9968167 |
| Kwatrecht            |      52 | 0.5476190 | 0.4423077 |   0.9911504 | 0.9781719 |
| Mont_Panisel         |     237 | 0.6764706 | 0.6793249 |   0.9607543 | 0.9304229 |
| Aalbeke              |      92 | 0.8135593 | 0.5217391 |   0.9947793 | 0.9749886 |
| Mons_en_Pevele       |     242 | 0.6419753 | 0.6446281 |   0.9555442 | 0.9213279 |

#### RF Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.8143713 | 0.8831169 |   0.8915325 | 0.8885857 |
| Diest                |      37 | 0.8461538 | 0.8918919 |   0.9972248 | 0.9954525 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9845384 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9954421 | 0.9931787 |
| Ursel                |      36 | 0.7692308 | 0.5555556 |   0.9972261 | 0.9899955 |
| Asse                 |      38 | 0.3947368 | 0.3947368 |   0.9893568 | 0.9790814 |
| Wemmel               |      72 | 0.5882353 | 0.5555556 |   0.9868359 | 0.9727149 |
| Lede                 |     181 | 0.6376812 | 0.4861878 |   0.9752230 | 0.9349704 |
| Brussel              |     391 | 0.6842105 | 0.8312020 |   0.9170354 | 0.9017735 |
| Merelbeke            |      12 | 0.8000000 | 0.6666667 |   0.9990855 | 0.9972715 |
| Kwatrecht            |      52 | 0.5625000 | 0.5192308 |   0.9902189 | 0.9790814 |
| Mont_Panisel         |     237 | 0.7071130 | 0.7130802 |   0.9643221 | 0.9372442 |
| Aalbeke              |      92 | 0.8500000 | 0.5543478 |   0.9957285 | 0.9772624 |
| Mons_en_Pevele       |     242 | 0.6948357 | 0.6115702 |   0.9667859 | 0.9276944 |

#### LGBM Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.7735426 | 0.8961039 |   0.8586424 | 0.8717599 |
| Diest                |      37 | 0.4794521 | 0.9459459 |   0.9824237 | 0.9818099 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9845384 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9972653 | 0.9949977 |
| Ursel                |      36 | 0.5833333 | 0.3888889 |   0.9953768 | 0.9854479 |
| Asse                 |      38 | 0.8000000 | 0.2105263 |   0.9990745 | 0.9854479 |
| Wemmel               |      72 | 0.6428571 | 0.5000000 |   0.9905971 | 0.9745339 |
| Lede                 |     181 | 0.6642336 | 0.5027624 |   0.9772052 | 0.9381537 |
| Brussel              |     391 | 0.6712963 | 0.7416880 |   0.9214602 | 0.8894952 |
| Merelbeke            |      12 | 0.8000000 | 0.3333333 |   0.9995428 | 0.9959072 |
| Kwatrecht            |      52 | 0.4864865 | 0.3461538 |   0.9911504 | 0.9758981 |
| Mont_Panisel         |     237 | 0.6504065 | 0.6751055 |   0.9561672 | 0.9258754 |
| Aalbeke              |      92 | 0.8593750 | 0.5978261 |   0.9957285 | 0.9790814 |
| Mons_en_Pevele       |     242 | 0.6912442 | 0.6198347 |   0.9657639 | 0.9276944 |
