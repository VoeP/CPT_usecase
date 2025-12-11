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

     [1] "cpt_features_true_0.6_100_multiplicative.csv"
     [2] "cpt_features_true_0.6_42_additive.csv"       
     [3] "cpt_features_true_0.6_42_multiplicative.csv" 
     [4] "cpt_ids_true_0.6_100_multiplicative.csv"     
     [5] "cpt_ids_true_0.6_42_additive.csv"            
     [6] "cpt_ids_true_0.6_42_multiplicative.csv"      
     [7] "predictions_r_models"                        
     [8] "split_res.json"                              
     [9] "split_res.pkl"                               
    [10] "test_binned_true_0.6_100_multiplicative.csv" 
    [11] "test_binned_true_0.6_42_additive.csv"        
    [12] "test_binned_true_0.6_42_multiplicative.csv"  
    [13] "train_binned_true_0.6_100_multiplicative.csv"
    [14] "train_binned_true_0.6_42_additive.csv"       
    [15] "train_binned_true_0.6_42_multiplicative.csv" 

``` r
train_dt <- fread(here(
    results_folder,
    "train_binned_true_0.6_100_multiplicative.csv"
)) |>
  select(sondering_id, depth_bin, lithostrat_id, everything())
test_dt <- fread(here(
    results_folder,
    "test_binned_true_0.6_100_multiplicative.csv"
)) |>
  select(sondering_id, depth_bin, lithostrat_id, everything())
train_dt[, unique(lithostrat_id)]
```

     [1] "Quartair"             "Mont_Panisel"         "Aalbeke"             
     [4] "Mons_en_Pevele"       "Maldegem"             "Brussel"             
     [7] "Onbekend"             "Asse"                 "Wemmel"              
    [10] "Lede"                 "Bolderberg"           "Ursel"               
    [13] "Merelbeke"            "Kwatrecht"            "Antropogeen"         
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

6572.855 sec elapsed

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
| 5 | 79 | 27 | 8 | 0.0608613 | 2.8035324 | 0.3497933 | accuracy | multiclass | 0.6462328 | 10 | 0.0297998 | pre0_mod01_post0 |
| 5 | 79 | 27 | 8 | 0.0608613 | 2.8035324 | 0.3497933 | mn_log_loss | multiclass | 1.1231528 | 10 | 0.0781518 | pre0_mod01_post0 |
| 5 | 1695 | 4 | 13 | 0.0010761 | 1.1141530 | 0.7539592 | accuracy | multiclass | 0.6928266 | 10 | 0.0229429 | pre0_mod02_post0 |
| 5 | 1695 | 4 | 13 | 0.0010761 | 1.1141530 | 0.7539592 | mn_log_loss | multiclass | 1.2099147 | 10 | 0.0591211 | pre0_mod02_post0 |
| 12 | 411 | 31 | 2 | 0.0127187 | 0.0021171 | 0.6736923 | accuracy | multiclass | 0.6503160 | 10 | 0.0261592 | pre0_mod03_post0 |
| 12 | 411 | 31 | 2 | 0.0127187 | 0.0021171 | 0.6736923 | mn_log_loss | multiclass | 1.0916498 | 10 | 0.0718992 | pre0_mod03_post0 |
| 15 | 1850 | 17 | 12 | 0.1022960 | 0.0000002 | 0.5799099 | accuracy | multiclass | 0.6935757 | 10 | 0.0222948 | pre0_mod04_post0 |
| 15 | 1850 | 17 | 12 | 0.1022960 | 0.0000002 | 0.5799099 | mn_log_loss | multiclass | 1.0792589 | 10 | 0.0976227 | pre0_mod04_post0 |
| 19 | 1081 | 25 | 14 | 0.0033337 | 0.0000026 | 0.1551031 | accuracy | multiclass | 0.6102566 | 10 | 0.0297164 | pre0_mod05_post0 |
| 19 | 1081 | 25 | 14 | 0.0033337 | 0.0000026 | 0.1551031 | mn_log_loss | multiclass | 1.3215103 | 10 | 0.0828097 | pre0_mod05_post0 |
| 23 | 1349 | 24 | 11 | 0.0017564 | 0.0000059 | 0.4686107 | accuracy | multiclass | 0.6652434 | 10 | 0.0271230 | pre0_mod06_post0 |
| 23 | 1349 | 24 | 11 | 0.0017564 | 0.0000059 | 0.4686107 | mn_log_loss | multiclass | 1.2040538 | 10 | 0.0693635 | pre0_mod06_post0 |
| 27 | 744 | 38 | 15 | 0.0447986 | 0.0000000 | 0.2662227 | accuracy | multiclass | 0.6376995 | 10 | 0.0306738 | pre0_mod07_post0 |
| 27 | 744 | 38 | 15 | 0.0447986 | 0.0000000 | 0.2662227 | mn_log_loss | multiclass | 1.1353928 | 10 | 0.0901179 | pre0_mod07_post0 |
| 34 | 1515 | 39 | 5 | 0.0022958 | 0.0000589 | 0.2827376 | accuracy | multiclass | 0.6147879 | 10 | 0.0303581 | pre0_mod08_post0 |
| 34 | 1515 | 39 | 5 | 0.0022958 | 0.0000589 | 0.2827376 | mn_log_loss | multiclass | 1.2722995 | 10 | 0.0801670 | pre0_mod08_post0 |
| 36 | 130 | 33 | 4 | 0.0141256 | 0.4414687 | 0.1423113 | accuracy | multiclass | 0.5497527 | 10 | 0.0315819 | pre0_mod09_post0 |
| 36 | 130 | 33 | 4 | 0.0141256 | 0.4414687 | 0.1423113 | mn_log_loss | multiclass | 1.6315059 | 10 | 0.0656020 | pre0_mod09_post0 |
| 44 | 270 | 11 | 3 | 0.2418692 | 0.0000000 | 0.9894672 | accuracy | multiclass | 0.6834161 | 10 | 0.0252142 | pre0_mod10_post0 |
| 44 | 270 | 11 | 3 | 0.2418692 | 0.0000000 | 0.9894672 | mn_log_loss | multiclass | 1.1250264 | 10 | 0.1181781 | pre0_mod10_post0 |
| 44 | 1182 | 10 | 8 | 0.0236322 | 0.0000000 | 0.8520422 | accuracy | multiclass | 0.6990144 | 10 | 0.0216479 | pre0_mod11_post0 |
| 44 | 1182 | 10 | 8 | 0.0236322 | 0.0000000 | 0.8520422 | mn_log_loss | multiclass | 1.0382582 | 10 | 0.0929633 | pre0_mod11_post0 |
| 50 | 891 | 22 | 13 | 0.2228520 | 0.0000453 | 0.9252419 | accuracy | multiclass | 0.6933974 | 10 | 0.0224689 | pre0_mod12_post0 |
| 50 | 891 | 22 | 13 | 0.2228520 | 0.0000453 | 0.9252419 | mn_log_loss | multiclass | 1.1471897 | 10 | 0.1005738 | pre0_mod12_post0 |
| 54 | 1722 | 20 | 5 | 0.0764287 | 0.0115276 | 0.7209766 | accuracy | multiclass | 0.6993243 | 10 | 0.0212266 | pre0_mod13_post0 |
| 54 | 1722 | 20 | 5 | 0.0764287 | 0.0115276 | 0.7209766 | mn_log_loss | multiclass | 1.0902105 | 10 | 0.0985018 | pre0_mod13_post0 |
| 61 | 560 | 13 | 7 | 0.0313584 | 0.0046225 | 0.4474373 | accuracy | multiclass | 0.7079453 | 10 | 0.0225255 | pre0_mod14_post0 |
| 61 | 560 | 13 | 7 | 0.0313584 | 0.0046225 | 0.4474373 | mn_log_loss | multiclass | 0.9482662 | 10 | 0.0862200 | pre0_mod14_post0 |
| 65 | 692 | 14 | 1 | 0.0089412 | 0.0000003 | 0.5131386 | accuracy | multiclass | 0.6130660 | 10 | 0.0324245 | pre0_mod15_post0 |
| 65 | 692 | 14 | 1 | 0.0089412 | 0.0000003 | 0.5131386 | mn_log_loss | multiclass | 1.2141991 | 10 | 0.0758175 | pre0_mod15_post0 |
| 67 | 1214 | 34 | 10 | 0.0072295 | 0.0000000 | 0.2201001 | accuracy | multiclass | 0.6315441 | 10 | 0.0310988 | pre0_mod16_post0 |
| 67 | 1214 | 34 | 10 | 0.0072295 | 0.0000000 | 0.2201001 | mn_log_loss | multiclass | 1.1930187 | 10 | 0.0889955 | pre0_mod16_post0 |
| 73 | 1963 | 17 | 7 | 0.0025343 | 0.0459465 | 0.3913554 | accuracy | multiclass | 0.6877947 | 10 | 0.0235976 | pre0_mod17_post0 |
| 73 | 1963 | 17 | 7 | 0.0025343 | 0.0459465 | 0.3913554 | mn_log_loss | multiclass | 1.0075825 | 10 | 0.0781290 | pre0_mod17_post0 |
| 76 | 988 | 6 | 3 | 0.1575022 | 20.7752527 | 0.8066164 | accuracy | multiclass | 0.6571827 | 10 | 0.0242280 | pre0_mod18_post0 |
| 76 | 988 | 6 | 3 | 0.1575022 | 20.7752527 | 0.8066164 | mn_log_loss | multiclass | 1.0921228 | 10 | 0.0750442 | pre0_mod18_post0 |
| 83 | 1407 | 30 | 11 | 0.0408536 | 0.0000000 | 0.5967655 | accuracy | multiclass | 0.6909838 | 10 | 0.0218295 | pre0_mod19_post0 |
| 83 | 1407 | 30 | 11 | 0.0408536 | 0.0000000 | 0.5967655 | mn_log_loss | multiclass | 1.0138433 | 10 | 0.0841612 | pre0_mod19_post0 |
| 85 | 397 | 3 | 9 | 0.0046651 | 0.0002821 | 0.8950980 | accuracy | multiclass | 0.6834577 | 10 | 0.0227239 | pre0_mod20_post0 |
| 85 | 397 | 3 | 9 | 0.0046651 | 0.0002821 | 0.8950980 | mn_log_loss | multiclass | 1.2220257 | 10 | 0.0565208 | pre0_mod20_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    5 |  1675 |    17 | accuracy    | multiclass | 0.6899585 |  10 | 0.0208959 | pre0_mod01_post0 |
|    5 |  1675 |    17 | mn_log_loss | multiclass | 0.9504773 |  10 | 0.0569039 | pre0_mod01_post0 |
|    8 |  1041 |    13 | accuracy    | multiclass | 0.6956852 |  10 | 0.0199818 | pre0_mod02_post0 |
|    8 |  1041 |    13 | mn_log_loss | multiclass | 0.9152780 |  10 | 0.0580219 | pre0_mod02_post0 |
|   11 |  1587 |     7 | accuracy    | multiclass | 0.7009565 |  10 | 0.0197557 | pre0_mod03_post0 |
|   11 |  1587 |     7 | mn_log_loss | multiclass | 0.8897870 |  10 | 0.0586776 | pre0_mod03_post0 |
|   15 |  1385 |    10 | accuracy    | multiclass | 0.7034961 |  10 | 0.0193273 | pre0_mod04_post0 |
|   15 |  1385 |    10 | mn_log_loss | multiclass | 0.8878444 |  10 | 0.0594690 | pre0_mod04_post0 |
|   22 |   712 |    19 | accuracy    | multiclass | 0.7003034 |  10 | 0.0206017 | pre0_mod05_post0 |
|   22 |   712 |    19 | mn_log_loss | multiclass | 0.8996096 |  10 | 0.0613134 | pre0_mod05_post0 |
|   26 |    84 |     9 | accuracy    | multiclass | 0.7024715 |  10 | 0.0205680 | pre0_mod06_post0 |
|   26 |    84 |     9 | mn_log_loss | multiclass | 0.9690274 |  10 | 0.0924843 | pre0_mod06_post0 |
|   27 |  1818 |    35 | accuracy    | multiclass | 0.6926813 |  10 | 0.0212956 | pre0_mod07_post0 |
|   27 |  1818 |    35 | mn_log_loss | multiclass | 0.9249154 |  10 | 0.0610687 | pre0_mod07_post0 |
|   32 |   419 |    32 | accuracy    | multiclass | 0.6959563 |  10 | 0.0208574 | pre0_mod08_post0 |
|   32 |   419 |    32 | mn_log_loss | multiclass | 0.9187072 |  10 | 0.0620025 | pre0_mod08_post0 |
|   36 |   587 |     3 | accuracy    | multiclass | 0.7071665 |  10 | 0.0194489 | pre0_mod09_post0 |
|   36 |   587 |     3 | mn_log_loss | multiclass | 0.9184279 |  10 | 0.0752666 | pre0_mod09_post0 |
|   41 |   333 |    23 | accuracy    | multiclass | 0.6985887 |  10 | 0.0211381 | pre0_mod10_post0 |
|   41 |   333 |    23 | mn_log_loss | multiclass | 0.9075736 |  10 | 0.0640946 | pre0_mod10_post0 |
|   44 |  1289 |    34 | accuracy    | multiclass | 0.6971089 |  10 | 0.0224594 | pre0_mod11_post0 |
|   44 |  1289 |    34 | mn_log_loss | multiclass | 0.9203783 |  10 | 0.0625669 | pre0_mod11_post0 |
|   50 |   613 |    14 | accuracy    | multiclass | 0.7047793 |  10 | 0.0212578 | pre0_mod12_post0 |
|   50 |   613 |    14 | mn_log_loss | multiclass | 0.8887836 |  10 | 0.0652056 | pre0_mod12_post0 |
|   53 |  1200 |    23 | accuracy    | multiclass | 0.7012355 |  10 | 0.0211272 | pre0_mod13_post0 |
|   53 |  1200 |    23 | mn_log_loss | multiclass | 0.9019953 |  10 | 0.0628439 | pre0_mod13_post0 |
|   57 |   244 |    25 | accuracy    | multiclass | 0.7010105 |  10 | 0.0230368 | pre0_mod14_post0 |
|   57 |   244 |    25 | mn_log_loss | multiclass | 0.9219002 |  10 | 0.0704328 | pre0_mod14_post0 |
|   65 |   958 |    28 | accuracy    | multiclass | 0.6990086 |  10 | 0.0228398 | pre0_mod15_post0 |
|   65 |   958 |    28 | mn_log_loss | multiclass | 0.9121248 |  10 | 0.0652853 | pre0_mod15_post0 |
|   66 |   186 |    29 | accuracy    | multiclass | 0.6988793 |  10 | 0.0229306 | pre0_mod16_post0 |
|   66 |   186 |    29 | mn_log_loss | multiclass | 0.9341155 |  10 | 0.0747614 | pre0_mod16_post0 |
|   70 |  1737 |    16 | accuracy    | multiclass | 0.7040382 |  10 | 0.0217066 | pre0_mod17_post0 |
|   70 |  1737 |    16 | mn_log_loss | multiclass | 0.8899855 |  10 | 0.0629584 | pre0_mod17_post0 |
|   78 |  1474 |    36 | accuracy    | multiclass | 0.6986548 |  10 | 0.0232343 | pre0_mod18_post0 |
|   78 |  1474 |    36 | mn_log_loss | multiclass | 0.9211332 |  10 | 0.0632018 | pre0_mod18_post0 |
|   79 |   868 |    38 | accuracy    | multiclass | 0.6962287 |  10 | 0.0234956 | pre0_mod19_post0 |
|   79 |   868 |    38 | mn_log_loss | multiclass | 0.9344799 |  10 | 0.0684705 | pre0_mod19_post0 |
|   85 |  1996 |     5 | accuracy    | multiclass | 0.7052612 |  10 | 0.0215073 | pre0_mod20_post0 |
|   85 |  1996 |     5 | mn_log_loss | multiclass | 0.9012700 |  10 | 0.0768314 | pre0_mod20_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 4 | 1900 | 20 | 2 | 0.0000000 | 0.5873091 | 0.5295256 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod01_post0 |
| 4 | 1900 | 20 | 2 | 0.0000000 | 0.5873091 | 0.5295256 | mn_log_loss | multiclass | 2.1120214 | 10 | 0.0756906 | pre0_mod01_post0 |
| 9 | 953 | 33 | 12 | 0.0088443 | 0.0001441 | 0.4569949 | accuracy | multiclass | 0.7080520 | 10 | 0.0208471 | pre0_mod02_post0 |
| 9 | 953 | 33 | 12 | 0.0088443 | 0.0001441 | 0.4569949 | mn_log_loss | multiclass | 1.0461617 | 10 | 0.1092936 | pre0_mod02_post0 |
| 14 | 1415 | 31 | 4 | 0.0000000 | 0.0000370 | 0.3320076 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod03_post0 |
| 14 | 1415 | 31 | 4 | 0.0000000 | 0.0000370 | 0.3320076 | mn_log_loss | multiclass | 2.1121016 | 10 | 0.0756954 | pre0_mod03_post0 |
| 18 | 1517 | 10 | 14 | 0.0000000 | 0.0000000 | 0.4135592 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod04_post0 |
| 18 | 1517 | 10 | 14 | 0.0000000 | 0.0000000 | 0.4135592 | mn_log_loss | multiclass | 2.1120647 | 10 | 0.0756915 | pre0_mod04_post0 |
| 22 | 1306 | 13 | 5 | 0.0001226 | 0.0000001 | 0.1730588 | accuracy | multiclass | 0.3664547 | 10 | 0.0259969 | pre0_mod05_post0 |
| 22 | 1306 | 13 | 5 | 0.0001226 | 0.0000001 | 0.1730588 | mn_log_loss | multiclass | 1.7393074 | 10 | 0.0626266 | pre0_mod05_post0 |
| 26 | 691 | 5 | 8 | 0.0000003 | 0.0000020 | 0.6253956 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod06_post0 |
| 26 | 691 | 5 | 8 | 0.0000003 | 0.0000020 | 0.6253956 | mn_log_loss | multiclass | 2.1111265 | 10 | 0.0755934 | pre0_mod06_post0 |
| 30 | 1688 | 24 | 12 | 0.0000017 | 0.0422138 | 0.5790989 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod07_post0 |
| 30 | 1688 | 24 | 12 | 0.0000017 | 0.0422138 | 0.5790989 | mn_log_loss | multiclass | 2.0988324 | 10 | 0.0744691 | pre0_mod07_post0 |
| 34 | 1097 | 18 | 9 | 0.0000096 | 0.0000000 | 0.4907825 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod08_post0 |
| 34 | 1097 | 18 | 9 | 0.0000096 | 0.0000000 | 0.4907825 | mn_log_loss | multiclass | 2.0662881 | 10 | 0.0717005 | pre0_mod08_post0 |
| 37 | 518 | 36 | 10 | 0.0000411 | 0.0000005 | 0.1045997 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod09_post0 |
| 37 | 518 | 36 | 10 | 0.0000411 | 0.0000005 | 0.1045997 | mn_log_loss | multiclass | 2.0663535 | 10 | 0.0743041 | pre0_mod09_post0 |
| 40 | 32 | 27 | 3 | 0.0000000 | 0.0000000 | 0.9487551 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod10_post0 |
| 40 | 32 | 27 | 3 | 0.0000000 | 0.0000000 | 0.9487551 | mn_log_loss | multiclass | 2.1121041 | 10 | 0.0756956 | pre0_mod10_post0 |
| 44 | 1916 | 26 | 2 | 0.0008882 | 0.0020554 | 0.8081132 | accuracy | multiclass | 0.6423244 | 10 | 0.0267793 | pre0_mod11_post0 |
| 44 | 1916 | 26 | 2 | 0.0008882 | 0.0020554 | 0.8081132 | mn_log_loss | multiclass | 1.1637037 | 10 | 0.0710946 | pre0_mod11_post0 |
| 52 | 392 | 34 | 5 | 0.0000008 | 0.6011667 | 0.8717955 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod12_post0 |
| 52 | 392 | 34 | 5 | 0.0000008 | 0.6011667 | 0.8717955 | mn_log_loss | multiclass | 2.1107267 | 10 | 0.0755587 | pre0_mod12_post0 |
| 57 | 1243 | 22 | 8 | 0.0000001 | 0.0000041 | 0.1991896 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod13_post0 |
| 57 | 1243 | 22 | 8 | 0.0000001 | 0.0000041 | 0.1991896 | mn_log_loss | multiclass | 2.1117330 | 10 | 0.0756685 | pre0_mod13_post0 |
| 59 | 156 | 14 | 15 | 0.0437665 | 0.0029942 | 0.7014380 | accuracy | multiclass | 0.7001658 | 10 | 0.0211524 | pre0_mod14_post0 |
| 59 | 156 | 14 | 15 | 0.0437665 | 0.0029942 | 0.7014380 | mn_log_loss | multiclass | 1.1867103 | 10 | 0.1221597 | pre0_mod14_post0 |
| 65 | 282 | 7 | 10 | 0.0004847 | 10.4729690 | 0.2716258 | accuracy | multiclass | 0.3522334 | 10 | 0.0238255 | pre0_mod15_post0 |
| 65 | 282 | 7 | 10 | 0.0004847 | 10.4729690 | 0.2716258 | mn_log_loss | multiclass | 1.7698809 | 10 | 0.0616196 | pre0_mod15_post0 |
| 69 | 1705 | 15 | 11 | 0.0000000 | 0.0005663 | 0.8517456 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod16_post0 |
| 69 | 1705 | 15 | 11 | 0.0000000 | 0.0005663 | 0.8517456 | mn_log_loss | multiclass | 2.1120877 | 10 | 0.0756937 | pre0_mod16_post0 |
| 72 | 728 | 10 | 6 | 0.0000037 | 0.0000000 | 0.7646548 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod17_post0 |
| 72 | 728 | 10 | 6 | 0.0000037 | 0.0000000 | 0.7646548 | mn_log_loss | multiclass | 2.0983019 | 10 | 0.0742409 | pre0_mod17_post0 |
| 77 | 487 | 2 | 13 | 0.0024565 | 0.0345859 | 0.6726935 | accuracy | multiclass | 0.6755849 | 10 | 0.0222417 | pre0_mod18_post0 |
| 77 | 487 | 2 | 13 | 0.0024565 | 0.0345859 | 0.6726935 | mn_log_loss | multiclass | 1.1190730 | 10 | 0.0768455 | pre0_mod18_post0 |
| 81 | 1166 | 39 | 4 | 0.0000000 | 0.0000000 | 0.2837049 | accuracy | multiclass | 0.3287375 | 10 | 0.0218048 | pre0_mod19_post0 |
| 81 | 1166 | 39 | 4 | 0.0000000 | 0.0000000 | 0.2837049 | mn_log_loss | multiclass | 2.1120411 | 10 | 0.0756911 | pre0_mod19_post0 |

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
| rf    | accuracy    | 0.6996735 | 0.0213733 |
| xgb   | accuracy    | 0.6603909 | 0.0259818 |
| lgbm  | accuracy    | 0.4062317 | 0.0223319 |
| lgbm  | mn_log_loss | 1.8610853 | 0.0778596 |
| xgb   | mn_log_loss | 1.1580241 | 0.0824980 |
| rf    | mn_log_loss | 0.9153907 | 0.0660930 |

### Perfomance on training set

- We report training set metrics to check for overfitting.

``` r
# train_metrics from all models
train_metrics <- bind_rows(lapply(results, function(r) r$train_metrics))
kable(train_metrics)
```

| .metric     | .estimator     | .estimate | model |
|:------------|:---------------|----------:|:------|
| accuracy    | multiclass     | 0.9489871 | xgb   |
| mn_log_loss | multiclass     | 0.1927957 | xgb   |
| roc_auc     | macro_weighted | 0.9990472 | xgb   |
| accuracy    | multiclass     | 0.9567219 | rf    |
| mn_log_loss | multiclass     | 0.2523944 | rf    |
| roc_auc     | macro_weighted | 0.9992982 | rf    |
| accuracy    | multiclass     | 0.9596685 | lgbm  |
| mn_log_loss | multiclass     | 0.0704959 | lgbm  |
| roc_auc     | macro_weighted | 0.9995888 | lgbm  |

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
| accuracy    | multiclass     | 0.7397202 | xgb   |
| mn_log_loss | multiclass     | 0.7280606 | xgb   |
| roc_auc     | macro_weighted | 0.9658773 | xgb   |
| accuracy    | multiclass     | 0.7604917 | rf    |
| mn_log_loss | multiclass     | 0.7335488 | rf    |
| roc_auc     | macro_weighted | 0.9721663 | rf    |
| accuracy    | multiclass     | 0.7575244 | lgbm  |
| mn_log_loss | multiclass     | 0.7596244 | lgbm  |
| roc_auc     | macro_weighted | 0.9689665 | lgbm  |

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
| Quartair | 718 | 3 | 15 | 2 | 3 | 7 | 1 | 16 | 54 | 2 | 1 | 13 | 13 | 20 |
| Diest | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 7 | 0 | 0 | 10 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 5 | 0 | 5 | 0 | 25 | 10 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Asse | 0 | 0 | 2 | 0 | 6 | 28 | 6 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| Wemmel | 1 | 1 | 0 | 25 | 0 | 4 | 67 | 3 | 1 | 0 | 0 | 2 | 0 | 0 |
| Lede | 6 | 5 | 1 | 1 | 0 | 0 | 26 | 178 | 44 | 0 | 2 | 4 | 1 | 13 |
| Brussel | 57 | 0 | 0 | 0 | 0 | 0 | 1 | 24 | 303 | 0 | 0 | 1 | 0 | 12 |
| Merelbeke | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 14 | 8 | 2 | 7 |
| Mont_Panisel | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 8 | 20 | 180 | 9 | 20 |
| Aalbeke | 6 | 0 | 0 | 0 | 6 | 8 | 0 | 0 | 0 | 3 | 2 | 7 | 64 | 7 |
| Mons_en_Pevele | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 | 13 | 2 | 0 | 14 | 6 | 148 |

#### RF Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 725 | 1 | 18 | 2 | 5 | 9 | 5 | 30 | 38 | 2 | 1 | 17 | 9 | 20 |
| Diest | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 6 | 0 | 0 | 15 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 2 | 0 | 1 | 0 | 25 | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Asse | 0 | 0 | 4 | 0 | 4 | 28 | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 21 | 0 | 5 | 66 | 1 | 0 | 0 | 0 | 3 | 0 | 0 |
| Lede | 4 | 0 | 0 | 0 | 0 | 0 | 23 | 165 | 44 | 0 | 1 | 1 | 0 | 5 |
| Brussel | 53 | 0 | 0 | 0 | 0 | 0 | 1 | 24 | 327 | 0 | 1 | 3 | 0 | 18 |
| Merelbeke | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 6 | 3 | 12 |
| Mont_Panisel | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 4 | 7 | 21 | 186 | 9 | 11 |
| Aalbeke | 5 | 0 | 0 | 0 | 6 | 11 | 0 | 0 | 0 | 1 | 0 | 4 | 67 | 7 |
| Mons_en_Pevele | 6 | 0 | 0 | 0 | 0 | 2 | 5 | 11 | 7 | 2 | 2 | 9 | 7 | 156 |

#### LGBM Confusion Matrix

|  | Quartair | Diest | Bolderberg | Sint_Huibrechts_Hern | Ursel | Asse | Wemmel | Lede | Brussel | Merelbeke | Kwatrecht | Mont_Panisel | Aalbeke | Mons_en_Pevele |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 719 | 4 | 16 | 4 | 3 | 6 | 3 | 18 | 49 | 5 | 1 | 17 | 10 | 20 |
| Diest | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 7 | 0 | 0 | 10 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 3 | 0 | 2 | 0 | 25 | 9 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Asse | 0 | 0 | 4 | 0 | 6 | 29 | 5 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| Wemmel | 1 | 2 | 0 | 23 | 0 | 5 | 66 | 1 | 1 | 0 | 0 | 3 | 0 | 1 |
| Lede | 6 | 0 | 1 | 1 | 0 | 0 | 24 | 182 | 37 | 0 | 3 | 1 | 1 | 13 |
| Brussel | 58 | 0 | 0 | 0 | 0 | 0 | 2 | 21 | 319 | 0 | 0 | 2 | 0 | 13 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 7 | 2 | 11 |
| Mont_Panisel | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 4 | 6 | 9 | 15 | 183 | 6 | 15 |
| Aalbeke | 3 | 0 | 0 | 0 | 6 | 5 | 0 | 0 | 0 | 0 | 0 | 6 | 68 | 4 |
| Mons_en_Pevele | 5 | 0 | 0 | 0 | 0 | 2 | 4 | 8 | 8 | 0 | 0 | 11 | 8 | 152 |

### Per-class metrics

- We report precision, recall, specificity, accuracy per lithostrat
  class.
- There is a correlation between class support (number of samples) and
  these metrics.

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
| Quartair             |     806 | 0.8271889 | 0.8908189 |   0.9034127 | 0.8991098 |
| Diest                |      19 | 1.0000000 | 0.5263158 |   1.0000000 | 0.9961848 |
| Bolderberg           |      23 |        NA | 0.0000000 |   1.0000000 | 0.9902501 |
| Sint_Huibrechts_Hern |      38 | 0.4545455 | 0.2631579 |   0.9948298 | 0.9830437 |
| Ursel                |      40 | 0.5319149 | 0.6250000 |   0.9905132 | 0.9843154 |
| Asse                 |      63 | 0.6222222 | 0.4444444 |   0.9925958 | 0.9779568 |
| Wemmel               |     105 | 0.6442308 | 0.6380952 |   0.9835847 | 0.9682069 |
| Lede                 |     234 | 0.6334520 | 0.7606838 |   0.9515294 | 0.9325986 |
| Brussel              |     420 | 0.7613065 | 0.7214286 |   0.9510057 | 0.9101314 |
| Merelbeke            |      15 | 0.0000000 | 0.0000000 |   0.9991468 | 0.9927936 |
| Kwatrecht            |      39 | 0.4375000 | 0.3589744 |   0.9922414 | 0.9817719 |
| Mont_Panisel         |     233 | 0.7171315 | 0.7725322 |   0.9666040 | 0.9474354 |
| Aalbeke              |      95 | 0.6213592 | 0.6736842 |   0.9827739 | 0.9703264 |
| Mons_en_Pevele       |     229 | 0.7551020 | 0.6462882 |   0.9774648 | 0.9453158 |

#### RF Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     806 | 0.8219955 | 0.8995037 |   0.8989053 | 0.8991098 |
| Diest                |      19 | 1.0000000 | 0.9473684 |   1.0000000 | 0.9995761 |
| Bolderberg           |      23 |        NA | 0.0000000 |   1.0000000 | 0.9902501 |
| Sint_Huibrechts_Hern |      38 | 0.5357143 | 0.3947368 |   0.9943990 | 0.9847393 |
| Ursel                |      40 | 0.7575758 | 0.6250000 |   0.9965502 | 0.9902501 |
| Asse                 |      63 | 0.6829268 | 0.4444444 |   0.9943380 | 0.9796524 |
| Wemmel               |     105 | 0.6804124 | 0.6285714 |   0.9862467 | 0.9703264 |
| Lede                 |     234 | 0.6790123 | 0.7051282 |   0.9632941 | 0.9376855 |
| Brussel              |     420 | 0.7658080 | 0.7785714 |   0.9484270 | 0.9181857 |
| Merelbeke            |      15 | 1.0000000 | 0.2000000 |   1.0000000 | 0.9949131 |
| Kwatrecht            |      39 | 0.3823529 | 0.3333333 |   0.9909483 | 0.9800763 |
| Mont_Panisel         |     233 | 0.7591837 | 0.7982833 |   0.9722484 | 0.9550657 |
| Aalbeke              |      95 | 0.6633663 | 0.7052632 |   0.9849823 | 0.9737177 |
| Mons_en_Pevele       |     229 | 0.7536232 | 0.6812227 |   0.9760563 | 0.9474354 |

#### LGBM Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     806 | 0.8217143 | 0.8920596 |   0.8995493 | 0.8969903 |
| Diest                |      19 | 1.0000000 | 0.6842105 |   1.0000000 | 0.9974565 |
| Bolderberg           |      23 |        NA | 0.0000000 |   1.0000000 | 0.9902501 |
| Sint_Huibrechts_Hern |      38 | 0.4347826 | 0.2631579 |   0.9943990 | 0.9826198 |
| Ursel                |      40 | 0.6250000 | 0.6250000 |   0.9935317 | 0.9872827 |
| Asse                 |      63 | 0.6304348 | 0.4603175 |   0.9925958 | 0.9783807 |
| Wemmel               |     105 | 0.6407767 | 0.6285714 |   0.9835847 | 0.9677830 |
| Lede                 |     234 | 0.6765799 | 0.7777778 |   0.9590588 | 0.9410767 |
| Brussel              |     420 | 0.7686747 | 0.7595238 |   0.9504899 | 0.9164900 |
| Merelbeke            |      15 | 0.5000000 | 0.0666667 |   0.9995734 | 0.9936414 |
| Kwatrecht            |      39 | 0.5000000 | 0.5128205 |   0.9913793 | 0.9834676 |
| Mont_Panisel         |     233 | 0.7530864 | 0.7854077 |   0.9717780 | 0.9533701 |
| Aalbeke              |      95 | 0.7391304 | 0.7157895 |   0.9893993 | 0.9783807 |
| Mons_en_Pevele       |     229 | 0.7676768 | 0.6637555 |   0.9784038 | 0.9478593 |

``` r
predictios_folder <- here(
    results_folder,
    "predictions_r_models"
)
if (!dir.exists(predictios_folder)) {
    dir.create(predictios_folder)
}
for (model_name in names(results)) {
  pred_dt <- results[[model_name]]$predictions
  assign(paste0("predictions_", model_name), pred_dt)
  fwrite(pred_dt, file = here(
    predictios_folder,
    paste0(
      "predictions_",
      model_name,
      "_binned_true_0.6_42_multiplicative.csv"
    )
  ))
}
```

``` r
# Load the file mapping binned predictions to original CPT depth rows
cpt_ids_path <- here("results", "cpt_ids_true_0.6_100_multiplicative.csv")
if (!file.exists(cpt_ids_path)) {
  stop("cpt_ids file not found at: ", cpt_ids_path)
}
cpt_ids <- fread(cpt_ids_path)

# Define the metrics to calculate on the full dataset
full_data_metrics <- yardstick::metric_set(accuracy, bal_accuracy, f_meas, kap)
id_cols <- c("sondering_id", "lithostrat_id", "depth_bin")

# Process each model's predictions
all_model_metrics <- lapply(names(results), function(model_name) {
  # Get the predictions made on the binned test data
  pred_dt <- results[[model_name]]$predictions

  # Merge predictions back to the original, unbinned CPT data
  # This assigns the prediction for a bin to every raw measurement in that bin
  merged_dt <- merge(pred_dt, cpt_ids, by = id_cols, all.x = TRUE)

  # Ensure factor levels are consistent for both truth and prediction columns
  merged_dt[, lithostrat_id := factor(lithostrat_id, levels = segments_oi)]
  merged_dt[, .pred_class := factor(.pred_class, levels = segments_oi)]

  # Order the data for inspection or plotting
  if ("diepte" %in% names(merged_dt)) {
    setorder(merged_dt, sondering_id, diepte)
  }

  # Calculate the specified metrics on the full, unbinned data
  metrics_tbl <- full_data_metrics(merged_dt,
    truth = lithostrat_id, estimate = .pred_class
  )

  metrics_tbl$model <- model_name # Add model name for easy grouping

  return(metrics_tbl)
})

# Combine metrics from all models into a single, tidy data frame
combined_full_metrics <- bind_rows(all_model_metrics)

# Display the final comparison table
cat("\n### Overall Performance on Full (Unbinned) CPT Data\n")
```

### Overall Performance on Full (Unbinned) CPT Data

``` r
kable(combined_full_metrics,
  caption = "Metrics evaluated on the original, unbinned CPT measurements."
)
```

| .metric      | .estimator | .estimate | model |
|:-------------|:-----------|----------:|:------|
| accuracy     | multiclass | 0.7305279 | xgb   |
| bal_accuracy | macro      | 0.7537707 | xgb   |
| f_meas       | macro      | 0.5813205 | xgb   |
| kap          | multiclass | 0.6743542 | xgb   |
| accuracy     | multiclass | 0.7583613 | rf    |
| bal_accuracy | macro      | 0.7800811 | rf    |
| f_meas       | macro      | 0.6490032 | rf    |
| kap          | multiclass | 0.7069474 | rf    |
| accuracy     | multiclass | 0.7492604 | lgbm  |
| bal_accuracy | macro      | 0.7708402 | lgbm  |
| f_meas       | macro      | 0.6213850 | lgbm  |
| kap          | multiclass | 0.6963154 | lgbm  |

Metrics evaluated on the original, unbinned CPT measurements.
