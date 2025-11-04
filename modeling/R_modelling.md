# Modelling CPT lithostratigraphy with R


### Functions

- The team was interested on which features were better at predicting
  the target variable.

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
fit_best_model <- function(
    workflow,
    tune_results,
    train_data,
    metric = "mn_log_loss") {
    finalize_workflow(
        workflow,
        select_best(tune_results,
            metric = metric
        )
    ) |>
        fit(data = train_data)
}


per_class_metrics <- function(pred_dt,
                              truth_col = "lithostrat_id",
                              estimate_col = ".pred_class") {

    lvls <- levels(pred_dt[[truth_col]])
    rows <- lapply(lvls, function(cls) {
        data <- pred_dt |>
            dplyr::mutate(
                truth_bin = factor(ifelse(.data[[truth_col]] == cls, cls,
                    "other"
                ), levels = c("other", cls)),
                pred_bin = factor(ifelse(.data[[estimate_col]] == cls, cls,
                    "other"
                ), levels = c("other", cls))
            )
        tibble::tibble(
            lithostrat_id = cls,
            support = sum(data$truth_bin == cls),
            precision = yardstick::precision(data,
                truth = truth_bin,
                estimate = pred_bin, event_level = "second"
            )$.estimate,
            recall = yardstick::recall(data,
                truth = truth_bin,
                estimate = pred_bin, event_level = "second"
            )$.estimate,
            specificity = yardstick::spec(data,
                truth = truth_bin,
                estimate = pred_bin, event_level = "second"
            )$.estimate,
            accuracy = yardstick::accuracy(data,
                truth = truth_bin, estimate = pred_bin
            )$.estimate
        )
    })
    dplyr::bind_rows(rows)
}


evaluate_model <- function(
    fitted_model,
    test_data,
    train_data,
    id_col = "sondering_id",
    cols_to_include = c("sondering_id", "lithostrat_id", "depth_bin")) {
    # Predict
    preds_class <- predict(fitted_model, new_data = test_data)
    preds_prob <- predict(fitted_model, new_data = test_data, type = "prob")

    # Combine predictions with ID and truth
    pred_dt <- bind_cols(
        test_data[, .SD, .SDcols = cols_to_include],
        preds_class,
        preds_prob
    )

    # Align factor levels with training data
    if (!is.factor(train_data$lithostrat_id)) {
        train_data[, lithostrat_id := factor(lithostrat_id)]
    }
    pred_dt[, lithostrat_id := factor(lithostrat_id,
        levels = levels(train_data$lithostrat_id)
    )]
    pred_dt[, .pred_class := factor(.pred_class,
        levels = levels(train_data$lithostrat_id)
    )]

    # Collect probability columns in the correct order of levels
    lvl <- levels(train_data$lithostrat_id)
    prob_cols <- paste0(".pred_", lvl)
    prob_cols <- prob_cols[prob_cols %in% names(pred_dt)]

    # Metrics
    acc_tbl <- yardstick::accuracy(pred_dt,
        truth = lithostrat_id,
        estimate = .pred_class
    )
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
            warning("Binary AUC skipped: probability column for positive class not found.")
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
    cm <- yardstick::conf_mat(
        pred_dt,
        truth = lithostrat_id,
        estimate = .pred_class
    )

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
    [2] "cpt_ids_true_0.6_42_additive.csv"     
    [3] "split_res.pkl"                        
    [4] "test_binned_true_0.6_42_additive.csv" 
    [5] "train_binned_true_0.6_42_additive.csv"

``` r
train_dt <- fread(here(
    results_folder,
    "train_binned_true_0.6_42_additive.csv"
))
test_dt <- fread(here(
    results_folder,
    "test_binned_true_0.6_42_additive.csv"
))
# factor lithostrat_id
# remove Onbekend
train_dt <- train_dt[lithostrat_id != "Onbekend"]
test_dt <- test_dt[lithostrat_id != "Onbekend"]
levels_litho <- unique(c(
    train_dt$lithostrat_id,
    test_dt$lithostrat_id
))
train_dt[, lithostrat_id := factor(lithostrat_id, levels = levels_litho)]
test_dt[, lithostrat_id := factor(lithostrat_id, levels = levels_litho)]
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

2923.43 sec elapsed

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
| 5 | 1368 | 19 | 13 | 0.0681331 | 0.0000004 | 0.2103119 | accuracy | multiclass | 0.6437279 | 10 | 0.0181877 | pre0_mod01_post0 |
| 5 | 1368 | 19 | 13 | 0.0681331 | 0.0000004 | 0.2103119 | mn_log_loss | multiclass | 1.1550257 | 10 | 0.0634122 | pre0_mod01_post0 |
| 14 | 1970 | 5 | 3 | 0.0062143 | 0.0023817 | 0.5312806 | accuracy | multiclass | 0.6769982 | 10 | 0.0214396 | pre0_mod02_post0 |
| 14 | 1970 | 5 | 3 | 0.0062143 | 0.0023817 | 0.5312806 | mn_log_loss | multiclass | 0.9445206 | 10 | 0.0665046 | pre0_mod02_post0 |
| 18 | 1199 | 15 | 4 | 0.0553570 | 0.0000954 | 0.5516237 | accuracy | multiclass | 0.6693006 | 10 | 0.0207129 | pre0_mod03_post0 |
| 18 | 1199 | 15 | 4 | 0.0553570 | 0.0000954 | 0.5516237 | mn_log_loss | multiclass | 1.0889627 | 10 | 0.0711514 | pre0_mod03_post0 |
| 22 | 1451 | 6 | 10 | 0.0016249 | 3.1837264 | 0.6790510 | accuracy | multiclass | 0.6766722 | 10 | 0.0145300 | pre0_mod04_post0 |
| 22 | 1451 | 6 | 10 | 0.0016249 | 3.1837264 | 0.6790510 | mn_log_loss | multiclass | 1.1444564 | 10 | 0.0409719 | pre0_mod04_post0 |
| 33 | 311 | 25 | 11 | 0.0100386 | 0.0000368 | 0.3800557 | accuracy | multiclass | 0.6443362 | 10 | 0.0170195 | pre0_mod05_post0 |
| 33 | 311 | 25 | 11 | 0.0100386 | 0.0000368 | 0.3800557 | mn_log_loss | multiclass | 1.1896095 | 10 | 0.0487252 | pre0_mod05_post0 |
| 41 | 727 | 23 | 5 | 0.1352962 | 0.0000000 | 0.9040289 | accuracy | multiclass | 0.6607786 | 10 | 0.0205635 | pre0_mod06_post0 |
| 41 | 727 | 23 | 5 | 0.1352962 | 0.0000000 | 0.9040289 | mn_log_loss | multiclass | 1.1881445 | 10 | 0.0800189 | pre0_mod06_post0 |
| 45 | 572 | 36 | 8 | 0.0206511 | 0.1430200 | 0.3543775 | accuracy | multiclass | 0.6373749 | 10 | 0.0143464 | pre0_mod07_post0 |
| 45 | 572 | 36 | 8 | 0.0206511 | 0.1430200 | 0.3543775 | mn_log_loss | multiclass | 1.1336275 | 10 | 0.0578987 | pre0_mod07_post0 |
| 52 | 114 | 32 | 1 | 0.0040264 | 0.0000001 | 0.8150651 | accuracy | multiclass | 0.5174926 | 10 | 0.0160344 | pre0_mod08_post0 |
| 52 | 114 | 32 | 1 | 0.0040264 | 0.0000001 | 0.8150651 | mn_log_loss | multiclass | 2.2217407 | 10 | 0.0181500 | pre0_mod08_post0 |
| 62 | 898 | 11 | 8 | 0.0029947 | 0.0000000 | 0.1075710 | accuracy | multiclass | 0.6248663 | 10 | 0.0155413 | pre0_mod09_post0 |
| 62 | 898 | 11 | 8 | 0.0029947 | 0.0000000 | 0.1075710 | mn_log_loss | multiclass | 1.3381260 | 10 | 0.0476538 | pre0_mod09_post0 |
| 68 | 1615 | 37 | 15 | 0.2691414 | 1.8674710 | 0.9193752 | accuracy | multiclass | 0.6485016 | 10 | 0.0184172 | pre0_mod10_post0 |
| 68 | 1615 | 37 | 15 | 0.2691414 | 1.8674710 | 0.9193752 | mn_log_loss | multiclass | 1.1136559 | 10 | 0.0632110 | pre0_mod10_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    4 |   948 |    12 | accuracy    | multiclass | 0.6845643 |  10 | 0.0212093 | pre0_mod01_post0 |
|    4 |   948 |    12 | mn_log_loss | multiclass | 1.0615847 |  10 | 0.0850281 | pre0_mod01_post0 |
|    9 |   613 |    29 | accuracy    | multiclass | 0.6776226 |  10 | 0.0204133 | pre0_mod02_post0 |
|    9 |   613 |    29 | mn_log_loss | multiclass | 1.0622104 |  10 | 0.0857058 | pre0_mod02_post0 |
|   16 |  1743 |    22 | accuracy    | multiclass | 0.6845448 |  10 | 0.0215976 | pre0_mod03_post0 |
|   16 |  1743 |    22 | mn_log_loss | multiclass | 1.0198267 |  10 | 0.0836842 | pre0_mod03_post0 |
|   24 |  1813 |    16 | accuracy    | multiclass | 0.6912131 |  10 | 0.0217735 | pre0_mod04_post0 |
|   24 |  1813 |    16 | mn_log_loss | multiclass | 0.9968837 |  10 | 0.0837939 | pre0_mod04_post0 |
|   30 |   463 |    28 | accuracy    | multiclass | 0.6812549 |  10 | 0.0228521 | pre0_mod05_post0 |
|   30 |   463 |    28 | mn_log_loss | multiclass | 1.0096022 |  10 | 0.0787252 | pre0_mod05_post0 |
|   35 |  1488 |    33 | accuracy    | multiclass | 0.6769328 |  10 | 0.0229831 | pre0_mod06_post0 |
|   35 |  1488 |    33 | mn_log_loss | multiclass | 1.0247923 |  10 | 0.0857921 | pre0_mod06_post0 |
|   47 |  1112 |    19 | accuracy    | multiclass | 0.6881881 |  10 | 0.0239156 | pre0_mod07_post0 |
|   47 |  1112 |    19 | mn_log_loss | multiclass | 1.0053397 |  10 | 0.0865869 | pre0_mod07_post0 |
|   55 |   334 |    39 | accuracy    | multiclass | 0.6773924 |  10 | 0.0231055 | pre0_mod08_post0 |
|   55 |   334 |    39 | mn_log_loss | multiclass | 1.0444852 |  10 | 0.0862425 | pre0_mod08_post0 |
|   62 |    60 |     6 | accuracy    | multiclass | 0.6904925 |  10 | 0.0209086 | pre0_mod09_post0 |
|   62 |    60 |     6 | mn_log_loss | multiclass | 1.2141267 |  10 | 0.1392259 | pre0_mod09_post0 |
|   65 |  1305 |     4 | accuracy    | multiclass | 0.6959641 |  10 | 0.0229096 | pre0_mod10_post0 |
|   65 |  1305 |     4 | mn_log_loss | multiclass | 1.0055267 |  10 | 0.0981678 | pre0_mod10_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 6 | 1880 | 29 | 8 | 0.0001521 | 0.2301958 | 0.5463992 | accuracy | multiclass | 0.4753530 | 10 | 0.0214748 | pre0_mod01_post0 |
| 6 | 1880 | 29 | 8 | 0.0001521 | 0.2301958 | 0.5463992 | mn_log_loss | multiclass | 1.5970014 | 10 | 0.0587209 | pre0_mod01_post0 |
| 12 | 49 | 26 | 1 | 0.0000006 | 0.0000001 | 0.1598878 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod02_post0 |
| 12 | 49 | 26 | 1 | 0.0000006 | 0.0000001 | 0.1598878 | mn_log_loss | multiclass | 2.1233383 | 10 | 0.0618169 | pre0_mod02_post0 |
| 15 | 1159 | 5 | 5 | 0.0107921 | 0.1559652 | 0.8021545 | accuracy | multiclass | 0.6789349 | 10 | 0.0230921 | pre0_mod03_post0 |
| 15 | 1159 | 5 | 5 | 0.0107921 | 0.1559652 | 0.8021545 | mn_log_loss | multiclass | 1.1388906 | 10 | 0.0996075 | pre0_mod03_post0 |
| 27 | 1569 | 9 | 8 | 0.0000000 | 0.0000000 | 0.3428746 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod04_post0 |
| 27 | 1569 | 9 | 8 | 0.0000000 | 0.0000000 | 0.3428746 | mn_log_loss | multiclass | 2.1233575 | 10 | 0.0618166 | pre0_mod04_post0 |
| 30 | 932 | 38 | 11 | 0.0879622 | 0.0000000 | 0.6940243 | accuracy | multiclass | 0.6843594 | 10 | 0.0213483 | pre0_mod05_post0 |
| 30 | 932 | 38 | 11 | 0.0879622 | 0.0000000 | 0.6940243 | mn_log_loss | multiclass | 2.0480601 | 10 | 0.1575969 | pre0_mod05_post0 |
| 38 | 528 | 15 | 10 | 0.0000077 | 0.0000011 | 0.8294304 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod06_post0 |
| 38 | 528 | 15 | 10 | 0.0000077 | 0.0000011 | 0.8294304 | mn_log_loss | multiclass | 2.1013590 | 10 | 0.0598817 | pre0_mod06_post0 |
| 42 | 779 | 20 | 13 | 0.0000000 | 0.0000354 | 0.2118503 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod07_post0 |
| 42 | 779 | 20 | 13 | 0.0000000 | 0.0000354 | 0.2118503 | mn_log_loss | multiclass | 2.1233733 | 10 | 0.0618181 | pre0_mod07_post0 |
| 54 | 337 | 33 | 3 | 0.0013853 | 0.0007773 | 0.5514933 | accuracy | multiclass | 0.5690130 | 10 | 0.0132486 | pre0_mod08_post0 |
| 54 | 337 | 33 | 3 | 0.0013853 | 0.0007773 | 0.5514933 | mn_log_loss | multiclass | 1.4656112 | 10 | 0.0392646 | pre0_mod08_post0 |
| 58 | 1393 | 11 | 14 | 0.0000000 | 21.2181766 | 0.9716269 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod09_post0 |
| 58 | 1393 | 11 | 14 | 0.0000000 | 21.2181766 | 0.9716269 | mn_log_loss | multiclass | 2.1233095 | 10 | 0.0618112 | pre0_mod09_post0 |
| 64 | 1617 | 24 | 4 | 0.0000001 | 0.0102498 | 0.3876599 | accuracy | multiclass | 0.3454978 | 10 | 0.0242678 | pre0_mod10_post0 |
| 64 | 1617 | 24 | 4 | 0.0000001 | 0.0102498 | 0.3876599 | mn_log_loss | multiclass | 2.1228349 | 10 | 0.0617754 | pre0_mod10_post0 |

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
| rf    | accuracy    | 0.6848170 | 0.0221668 |
| xgb   | accuracy    | 0.6400049 | 0.0176792 |
| lgbm  | accuracy    | 0.4480647 | 0.0224771 |
| lgbm  | mn_log_loss | 1.8967136 | 0.0724110 |
| xgb   | mn_log_loss | 1.2517870 | 0.0557698 |
| rf    | mn_log_loss | 1.0444378 | 0.0912952 |

### Perfomance on training set

- We report training set metrics to check for overfitting.

``` r
# train_metrics from all models
train_metrics <- bind_rows(lapply(results, function(r) r$train_metrics))
kable(train_metrics)
```

| .metric     | .estimator     | .estimate | model |
|:------------|:---------------|----------:|:------|
| accuracy    | multiclass     | 0.9294473 | xgb   |
| mn_log_loss | multiclass     | 0.3137230 | xgb   |
| roc_auc     | macro_weighted | 0.9954558 | xgb   |
| accuracy    | multiclass     | 0.9720807 | rf    |
| mn_log_loss | multiclass     | 0.2937599 | rf    |
| roc_auc     | macro_weighted | 0.9994588 | rf    |
| accuracy    | multiclass     | 1.0000000 | lgbm  |
| mn_log_loss | multiclass     | 0.0186303 | lgbm  |
| roc_auc     | macro_weighted | 1.0000000 | lgbm  |

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
| accuracy    | multiclass     | 0.6850181 | xgb   |
| mn_log_loss | multiclass     | 0.9305791 | xgb   |
| roc_auc     | macro_weighted | 0.9418104 | xgb   |
| accuracy    | multiclass     | 0.6868231 | rf    |
| mn_log_loss | multiclass     | 0.9176729 | rf    |
| roc_auc     | macro_weighted | 0.9481665 | rf    |
| accuracy    | multiclass     | 0.6845668 | lgbm  |
| mn_log_loss | multiclass     | 1.0974732 | lgbm  |
| roc_auc     | macro_weighted | 0.9432709 | lgbm  |

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

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 675 | 12 | 10 | 28 | 0 | 84 | 4 | 3 | 5 | 23 | 0 | 7 | 41 | 16 | 1 | 5 |
| Mont_Panisel | 7 | 172 | 18 | 42 | 0 | 1 | 3 | 6 | 1 | 0 | 2 | 21 | 2 | 0 | 0 | 0 |
| Aalbeke | 2 | 6 | 50 | 2 | 0 | 0 | 6 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 13 | 26 | 13 | 139 | 0 | 5 | 0 | 0 | 3 | 0 | 0 | 6 | 2 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 55 | 8 | 0 | 20 | 0 | 278 | 0 | 0 | 16 | 0 | 0 | 0 | 41 | 1 | 0 | 0 |
| Ursel | 1 | 0 | 1 | 0 | 0 | 0 | 16 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 0 | 1 | 0 | 0 | 7 | 9 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 2 | 0 | 0 | 3 | 0 | 1 | 0 | 5 | 37 | 2 | 0 | 0 | 13 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 10 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 18 | 0 | 0 | 0 | 0 |
| Lede | 7 | 2 | 0 | 2 | 0 | 22 | 0 | 0 | 4 | 0 | 0 | 0 | 82 | 0 | 0 | 0 |
| Antropogeen | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 36 | 0 |
| Sint_Huibrechts_Hern | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### RF Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 690 | 13 | 8 | 29 | 0 | 85 | 5 | 3 | 10 | 20 | 0 | 6 | 42 | 15 | 7 | 5 |
| Mont_Panisel | 3 | 167 | 15 | 49 | 0 | 0 | 2 | 6 | 1 | 0 | 1 | 13 | 1 | 0 | 0 | 0 |
| Aalbeke | 0 | 4 | 52 | 2 | 0 | 0 | 7 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 12 | 17 | 12 | 128 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 7 | 1 | 0 | 0 | 0 |
| Maldegem | 2 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Brussel | 54 | 13 | 0 | 13 | 0 | 283 | 0 | 0 | 12 | 0 | 0 | 1 | 47 | 2 | 0 | 0 |
| Ursel | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 10 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 9 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 5 | 34 | 1 | 0 | 0 | 7 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 20 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 23 | 0 | 0 | 0 | 0 |
| Lede | 4 | 2 | 0 | 14 | 0 | 13 | 0 | 0 | 3 | 3 | 0 | 0 | 82 | 0 | 0 | 0 |
| Antropogeen | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 30 | 0 |
| Sint_Huibrechts_Hern | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### LGBM Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 667 | 7 | 10 | 31 | 0 | 79 | 4 | 4 | 6 | 22 | 0 | 5 | 32 | 16 | 3 | 5 |
| Mont_Panisel | 6 | 156 | 9 | 40 | 0 | 3 | 4 | 6 | 3 | 0 | 1 | 18 | 2 | 0 | 0 | 0 |
| Aalbeke | 1 | 5 | 50 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 16 | 39 | 18 | 138 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 7 | 3 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 65 | 9 | 0 | 15 | 0 | 277 | 0 | 0 | 5 | 0 | 0 | 1 | 28 | 1 | 3 | 0 |
| Ursel | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 1 | 1 | 0 | 0 | 7 | 9 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 2 | 0 | 0 | 3 | 0 | 3 | 0 | 4 | 39 | 1 | 0 | 0 | 11 | 0 | 0 | 0 |
| Bolderberg | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 18 | 2 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 21 | 0 | 0 | 0 | 0 |
| Lede | 7 | 2 | 0 | 5 | 0 | 27 | 0 | 0 | 6 | 0 | 0 | 0 | 105 | 0 | 0 | 0 |
| Antropogeen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 1 | 0 | 0 | 0 | 0 | 31 | 0 |
| Sint_Huibrechts_Hern | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

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
| Quartair             |     770 | 0.7385120 | 0.8766234 |   0.8347165 | 0.8492780 |
| Mont_Panisel         |     237 | 0.6254545 | 0.7257384 |   0.9479535 | 0.9241877 |
| Aalbeke              |      92 | 0.7246377 | 0.5434783 |   0.9910546 | 0.9724729 |
| Mons_en_Pevele       |     242 | 0.6714976 | 0.5743802 |   0.9655522 | 0.9228339 |
| Maldegem             |       0 |        NA |        NA |   1.0000000 | 1.0000000 |
| Brussel              |     391 | 0.6634845 | 0.7109974 |   0.9227397 | 0.8853791 |
| Ursel                |      36 | 0.5333333 | 0.4444444 |   0.9935780 | 0.9846570 |
| Asse                 |      38 | 0.3750000 | 0.2368421 |   0.9931129 | 0.9801444 |
| Wemmel               |      72 | 0.5873016 | 0.5138889 |   0.9878731 | 0.9724729 |
| Bolderberg           |      34 | 1.0000000 | 0.0294118 |   1.0000000 | 0.9851083 |
| Merelbeke            |      12 | 0.7142857 | 0.4166667 |   0.9990926 | 0.9959386 |
| Kwatrecht            |      52 | 0.5142857 | 0.3461538 |   0.9921442 | 0.9769856 |
| Lede                 |     181 | 0.6890756 | 0.4530387 |   0.9818182 | 0.9386282 |
| Antropogeen          |      17 | 0.0000000 | 0.0000000 |   0.9990905 | 0.9914260 |
| Diest                |      37 | 0.9230769 | 0.9729730 |   0.9986232 | 0.9981949 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9945726 | 0.9923285 |

#### RF Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.7356077 | 0.8961039 |   0.8284924 | 0.8519856 |
| Mont_Panisel         |     237 | 0.6472868 | 0.7046414 |   0.9540172 | 0.9273466 |
| Aalbeke              |      92 | 0.7536232 | 0.5652174 |   0.9919962 | 0.9742780 |
| Mons_en_Pevele       |     242 | 0.6994536 | 0.5289256 |   0.9721378 | 0.9237365 |
| Maldegem             |       0 | 0.0000000 |        NA |   0.9936823 | 0.9936823 |
| Brussel              |     391 | 0.6658824 | 0.7237852 |   0.9221918 | 0.8871841 |
| Ursel                |      36 | 0.5806452 | 0.5000000 |   0.9940367 | 0.9860108 |
| Asse                 |      38 | 0.4736842 | 0.2368421 |   0.9954086 | 0.9824007 |
| Wemmel               |      72 | 0.6666667 | 0.4722222 |   0.9920709 | 0.9751805 |
| Bolderberg           |      34 |        NA | 0.0000000 |   1.0000000 | 0.9846570 |
| Merelbeke            |      12 | 0.4615385 | 0.5000000 |   0.9968240 | 0.9941336 |
| Kwatrecht            |      52 | 0.4600000 | 0.4423077 |   0.9875231 | 0.9747292 |
| Lede                 |     181 | 0.6776860 | 0.4530387 |   0.9808354 | 0.9377256 |
| Antropogeen          |      17 | 0.0000000 | 0.0000000 |   0.9995452 | 0.9918773 |
| Diest                |      37 | 0.9090909 | 0.8108108 |   0.9986232 | 0.9954874 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9954772 | 0.9932310 |

#### LGBM Per-Class Metrics

| lithostrat_id        | support | precision |    recall | specificity |  accuracy |
|:---------------------|--------:|----------:|----------:|------------:|----------:|
| Quartair             |     770 | 0.7485971 | 0.8662338 |   0.8450899 | 0.8524368 |
| Mont_Panisel         |     237 | 0.6290323 | 0.6582278 |   0.9535119 | 0.9219314 |
| Aalbeke              |      92 | 0.7936508 | 0.5434783 |   0.9938795 | 0.9751805 |
| Mons_en_Pevele       |     242 | 0.6188341 | 0.5702479 |   0.9569402 | 0.9147112 |
| Maldegem             |       0 |        NA |        NA |   1.0000000 | 1.0000000 |
| Brussel              |     391 | 0.6856436 | 0.7084399 |   0.9304110 | 0.8912455 |
| Ursel                |      36 | 0.6129032 | 0.5277778 |   0.9944954 | 0.9869134 |
| Asse                 |      38 | 0.3461538 | 0.2368421 |   0.9921947 | 0.9792419 |
| Wemmel               |      72 | 0.6190476 | 0.5416667 |   0.9888060 | 0.9742780 |
| Bolderberg           |      34 | 0.0000000 | 0.0000000 |   0.9990834 | 0.9837545 |
| Merelbeke            |      12 | 0.5555556 | 0.4166667 |   0.9981851 | 0.9950361 |
| Kwatrecht            |      52 | 0.4117647 | 0.4038462 |   0.9861368 | 0.9724729 |
| Lede                 |     181 | 0.6907895 | 0.5801105 |   0.9769042 | 0.9444946 |
| Antropogeen          |      17 |        NA | 0.0000000 |   1.0000000 | 0.9923285 |
| Diest                |      37 | 0.7380952 | 0.8378378 |   0.9949518 | 0.9923285 |
| Sint_Huibrechts_Hern |       5 | 0.0000000 | 0.0000000 |   0.9950249 | 0.9927798 |
