# Modelling CPT lithostratigraphy with R


### Functions

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


# Evaluate model and return predictions with metrics
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

    list(
        predictions = pred_dt,
        metrics = metrics,
        confusion_matrix = cm
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
    [3] "test_binned_true_0.6_42_additive.csv" 
    [4] "train_binned_true_0.6_42_additive.csv"

``` r
train_dt <- fread(here(
    results_folder,
    "train_binned_true_0.6_42_additive.csv"
))
test_dt <- fread(here(
    results_folder,
    "test_binned_true_0.6_42_additive.csv"
))
# factoor lithostrat_id

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

    # Add model name to metrics
    eval_res$metrics[, model := model_name]

    # Store results
    results[[model_name]] <- list(
        tune_results = tune_res,
        fitted_model = best_model,
        predictions = eval_res$predictions,
        metrics = eval_res$metrics,
        confusion_matrix = eval_res$confusion_matrix
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

3383.326 sec elapsed

## What the models achieved

### What we saw while tuning

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
| 5 | 1368 | 19 | 13 | 0.0681331 | 0.0000004 | 0.2103119 | accuracy | multiclass | 0.5972451 | 10 | 0.0194810 | pre0_mod01_post0 |
| 5 | 1368 | 19 | 13 | 0.0681331 | 0.0000004 | 0.2103119 | mn_log_loss | multiclass | 1.2856809 | 10 | 0.0677618 | pre0_mod01_post0 |
| 14 | 1970 | 5 | 3 | 0.0062143 | 0.0023817 | 0.5312806 | accuracy | multiclass | 0.6384464 | 10 | 0.0243144 | pre0_mod02_post0 |
| 14 | 1970 | 5 | 3 | 0.0062143 | 0.0023817 | 0.5312806 | mn_log_loss | multiclass | 1.0675657 | 10 | 0.0686170 | pre0_mod02_post0 |
| 18 | 1199 | 15 | 4 | 0.0553570 | 0.0000954 | 0.5516237 | accuracy | multiclass | 0.6343798 | 10 | 0.0258705 | pre0_mod03_post0 |
| 18 | 1199 | 15 | 4 | 0.0553570 | 0.0000954 | 0.5516237 | mn_log_loss | multiclass | 1.2247067 | 10 | 0.0889771 | pre0_mod03_post0 |
| 22 | 1451 | 6 | 10 | 0.0016249 | 3.1837264 | 0.6790510 | accuracy | multiclass | 0.6363714 | 10 | 0.0187779 | pre0_mod04_post0 |
| 22 | 1451 | 6 | 10 | 0.0016249 | 3.1837264 | 0.6790510 | mn_log_loss | multiclass | 1.2766028 | 10 | 0.0469134 | pre0_mod04_post0 |
| 33 | 311 | 25 | 11 | 0.0100386 | 0.0000368 | 0.3800557 | accuracy | multiclass | 0.5991721 | 10 | 0.0168230 | pre0_mod05_post0 |
| 33 | 311 | 25 | 11 | 0.0100386 | 0.0000368 | 0.3800557 | mn_log_loss | multiclass | 1.3299540 | 10 | 0.0472267 | pre0_mod05_post0 |
| 41 | 727 | 23 | 5 | 0.1352962 | 0.0000000 | 0.9040289 | accuracy | multiclass | 0.6155721 | 10 | 0.0253994 | pre0_mod06_post0 |
| 41 | 727 | 23 | 5 | 0.1352962 | 0.0000000 | 0.9040289 | mn_log_loss | multiclass | 1.3357687 | 10 | 0.0990550 | pre0_mod06_post0 |
| 45 | 572 | 36 | 8 | 0.0206511 | 0.1430200 | 0.3543775 | accuracy | multiclass | 0.5908946 | 10 | 0.0146360 | pre0_mod07_post0 |
| 45 | 572 | 36 | 8 | 0.0206511 | 0.1430200 | 0.3543775 | mn_log_loss | multiclass | 1.2632136 | 10 | 0.0549815 | pre0_mod07_post0 |
| 52 | 114 | 32 | 1 | 0.0040264 | 0.0000001 | 0.8150651 | accuracy | multiclass | 0.4888793 | 10 | 0.0169331 | pre0_mod08_post0 |
| 52 | 114 | 32 | 1 | 0.0040264 | 0.0000001 | 0.8150651 | mn_log_loss | multiclass | 2.3157607 | 10 | 0.0180447 | pre0_mod08_post0 |
| 62 | 898 | 11 | 8 | 0.0029947 | 0.0000000 | 0.1075710 | accuracy | multiclass | 0.5818531 | 10 | 0.0145866 | pre0_mod09_post0 |
| 62 | 898 | 11 | 8 | 0.0029947 | 0.0000000 | 0.1075710 | mn_log_loss | multiclass | 1.4711641 | 10 | 0.0454068 | pre0_mod09_post0 |
| 68 | 1615 | 37 | 15 | 0.2691414 | 1.8674710 | 0.9193752 | accuracy | multiclass | 0.5995922 | 10 | 0.0243656 | pre0_mod10_post0 |
| 68 | 1615 | 37 | 15 | 0.2691414 | 1.8674710 | 0.9193752 | mn_log_loss | multiclass | 1.2467614 | 10 | 0.0773097 | pre0_mod10_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    4 |   948 |    12 | accuracy    | multiclass | 0.6480492 |  10 | 0.0247719 | pre0_mod01_post0 |
|    4 |   948 |    12 | mn_log_loss | multiclass | 1.1834086 |  10 | 0.0942569 | pre0_mod01_post0 |
|    9 |   613 |    29 | accuracy    | multiclass | 0.6382992 |  10 | 0.0240588 | pre0_mod02_post0 |
|    9 |   613 |    29 | mn_log_loss | multiclass | 1.1798414 |  10 | 0.0917971 | pre0_mod02_post0 |
|   16 |  1743 |    22 | accuracy    | multiclass | 0.6474032 |  10 | 0.0256778 | pre0_mod03_post0 |
|   16 |  1743 |    22 | mn_log_loss | multiclass | 1.1123807 |  10 | 0.0773434 | pre0_mod03_post0 |
|   24 |  1813 |    16 | accuracy    | multiclass | 0.6487406 |  10 | 0.0256883 | pre0_mod04_post0 |
|   24 |  1813 |    16 | mn_log_loss | multiclass | 1.1093790 |  10 | 0.0839001 | pre0_mod04_post0 |
|   30 |   463 |    28 | accuracy    | multiclass | 0.6426212 |  10 | 0.0258544 | pre0_mod05_post0 |
|   30 |   463 |    28 | mn_log_loss | multiclass | 1.1539394 |  10 | 0.0929463 | pre0_mod05_post0 |
|   35 |  1488 |    33 | accuracy    | multiclass | 0.6418691 |  10 | 0.0253530 | pre0_mod06_post0 |
|   35 |  1488 |    33 | mn_log_loss | multiclass | 1.1289203 |  10 | 0.0822760 | pre0_mod06_post0 |
|   47 |  1112 |    19 | accuracy    | multiclass | 0.6476125 |  10 | 0.0261786 | pre0_mod07_post0 |
|   47 |  1112 |    19 | mn_log_loss | multiclass | 1.1322590 |  10 | 0.0969275 | pre0_mod07_post0 |
|   55 |   334 |    39 | accuracy    | multiclass | 0.6387363 |  10 | 0.0252609 | pre0_mod08_post0 |
|   55 |   334 |    39 | mn_log_loss | multiclass | 1.1567221 |  10 | 0.0932266 | pre0_mod08_post0 |
|   62 |    60 |     6 | accuracy    | multiclass | 0.6501226 |  10 | 0.0233225 | pre0_mod09_post0 |
|   62 |    60 |     6 | mn_log_loss | multiclass | 1.3474183 |  10 | 0.1241405 | pre0_mod09_post0 |
|   65 |  1305 |     4 | accuracy    | multiclass | 0.6489191 |  10 | 0.0273490 | pre0_mod10_post0 |
|   65 |  1305 |     4 | mn_log_loss | multiclass | 1.1361983 |  10 | 0.1111508 | pre0_mod10_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 6 | 1880 | 29 | 8 | 0.0001521 | 0.2301958 | 0.5463992 | accuracy | multiclass | 0.4506579 | 10 | 0.0212709 | pre0_mod01_post0 |
| 6 | 1880 | 29 | 8 | 0.0001521 | 0.2301958 | 0.5463992 | mn_log_loss | multiclass | 1.7075413 | 10 | 0.0575804 | pre0_mod01_post0 |
| 12 | 49 | 26 | 1 | 0.0000006 | 0.0000001 | 0.1598878 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod02_post0 |
| 12 | 49 | 26 | 1 | 0.0000006 | 0.0000001 | 0.1598878 | mn_log_loss | multiclass | 2.2225763 | 10 | 0.0592999 | pre0_mod02_post0 |
| 15 | 1159 | 5 | 5 | 0.0107921 | 0.1559652 | 0.8021545 | accuracy | multiclass | 0.6356245 | 10 | 0.0265037 | pre0_mod03_post0 |
| 15 | 1159 | 5 | 5 | 0.0107921 | 0.1559652 | 0.8021545 | mn_log_loss | multiclass | 1.2761647 | 10 | 0.1071445 | pre0_mod03_post0 |
| 27 | 1569 | 9 | 8 | 0.0000000 | 0.0000000 | 0.3428746 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod04_post0 |
| 27 | 1569 | 9 | 8 | 0.0000000 | 0.0000000 | 0.3428746 | mn_log_loss | multiclass | 2.2225952 | 10 | 0.0592996 | pre0_mod04_post0 |
| 30 | 932 | 38 | 11 | 0.0879622 | 0.0000000 | 0.6940243 | accuracy | multiclass | 0.6347289 | 10 | 0.0262487 | pre0_mod05_post0 |
| 30 | 932 | 38 | 11 | 0.0879622 | 0.0000000 | 0.6940243 | mn_log_loss | multiclass | 2.3867799 | 10 | 0.1979824 | pre0_mod05_post0 |
| 38 | 528 | 15 | 10 | 0.0000077 | 0.0000011 | 0.8294304 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod06_post0 |
| 38 | 528 | 15 | 10 | 0.0000077 | 0.0000011 | 0.8294304 | mn_log_loss | multiclass | 2.2013695 | 10 | 0.0574633 | pre0_mod06_post0 |
| 42 | 779 | 20 | 13 | 0.0000000 | 0.0000354 | 0.2118503 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod07_post0 |
| 42 | 779 | 20 | 13 | 0.0000000 | 0.0000354 | 0.2118503 | mn_log_loss | multiclass | 2.2226104 | 10 | 0.0593010 | pre0_mod07_post0 |
| 54 | 337 | 33 | 3 | 0.0013853 | 0.0007773 | 0.5514933 | accuracy | multiclass | 0.5397141 | 10 | 0.0143884 | pre0_mod08_post0 |
| 54 | 337 | 33 | 3 | 0.0013853 | 0.0007773 | 0.5514933 | mn_log_loss | multiclass | 1.5857402 | 10 | 0.0396748 | pre0_mod08_post0 |
| 58 | 1393 | 11 | 14 | 0.0000000 | 21.2181766 | 0.9716269 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod09_post0 |
| 58 | 1393 | 11 | 14 | 0.0000000 | 21.2181766 | 0.9716269 | mn_log_loss | multiclass | 2.2225480 | 10 | 0.0592944 | pre0_mod09_post0 |
| 64 | 1617 | 24 | 4 | 0.0000001 | 0.0102498 | 0.3876599 | accuracy | multiclass | 0.3248150 | 10 | 0.0230571 | pre0_mod10_post0 |
| 64 | 1617 | 24 | 4 | 0.0000001 | 0.0102498 | 0.3876599 | mn_log_loss | multiclass | 2.2220805 | 10 | 0.0592608 | pre0_mod10_post0 |

### Compare models at a glance

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
| rf    | accuracy    | 0.6452373 | 0.0253515 |
| xgb   | accuracy    | 0.5982406 | 0.0201188 |
| lgbm  | accuracy    | 0.4209616 | 0.0226754 |
| lgbm  | mn_log_loss | 2.0270006 | 0.0756301 |
| xgb   | mn_log_loss | 1.3817179 | 0.0614294 |
| rf    | mn_log_loss | 1.1640467 | 0.0947965 |

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
| accuracy    | multiclass     | 0.6364785 | xgb   |
| mn_log_loss | multiclass     | 1.0656103 | xgb   |
| roc_auc     | macro_weighted | 0.9297718 | xgb   |
| accuracy    | multiclass     | 0.6427970 | rf    |
| mn_log_loss | multiclass     | 1.0589179 | rf    |
| roc_auc     | macro_weighted | 0.9322139 | rf    |
| accuracy    | multiclass     | 0.6377422 | lgbm  |
| mn_log_loss | multiclass     | 1.2579453 | lgbm  |
| roc_auc     | macro_weighted | 0.9315513 | lgbm  |

``` r
# Display confusion matrices for all models
for (model_name in names(results)) {
  cat("\n####", toupper(model_name), "Confusion Matrix\n")
  cm_df <- conf_mat_to_df(results[[model_name]]$confusion_matrix)
  print(kable(cm_df))
  cat("\n")
}
```

#### XGB Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 670 | 10 | 11 | 29 | 0 | 84 | 119 | 4 | 3 | 5 | 21 | 0 | 8 | 40 | 16 | 2 | 5 |
| Mont_Panisel | 6 | 174 | 17 | 45 | 0 | 1 | 1 | 4 | 6 | 1 | 0 | 2 | 21 | 2 | 0 | 0 | 0 |
| Aalbeke | 2 | 6 | 49 | 3 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 13 | 26 | 13 | 135 | 0 | 5 | 0 | 0 | 0 | 3 | 0 | 0 | 5 | 3 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 55 | 9 | 0 | 19 | 0 | 277 | 1 | 0 | 0 | 16 | 0 | 0 | 1 | 41 | 1 | 4 | 0 |
| Onbekend | 5 | 1 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 16 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 0 | 1 | 0 | 0 | 12 | 5 | 10 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 2 | 0 | 0 | 3 | 0 | 1 | 6 | 0 | 4 | 37 | 2 | 0 | 0 | 12 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 9 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 17 | 0 | 0 | 0 | 0 |
| Lede | 7 | 2 | 0 | 3 | 0 | 23 | 4 | 0 | 0 | 4 | 1 | 0 | 0 | 83 | 0 | 0 | 0 |
| Antropogeen | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 31 | 0 |
| Sint_Huibrechts_Hern | 3 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### RF Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 677 | 10 | 8 | 25 | 0 | 82 | 106 | 5 | 3 | 7 | 15 | 0 | 5 | 40 | 15 | 4 | 4 |
| Mont_Panisel | 3 | 165 | 16 | 50 | 0 | 0 | 1 | 2 | 6 | 1 | 0 | 1 | 15 | 1 | 0 | 0 | 0 |
| Aalbeke | 0 | 4 | 52 | 3 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 12 | 18 | 11 | 129 | 0 | 1 | 0 | 0 | 0 | 6 | 0 | 0 | 4 | 1 | 0 | 0 | 0 |
| Maldegem | 2 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Brussel | 58 | 14 | 0 | 11 | 0 | 284 | 8 | 0 | 0 | 15 | 0 | 0 | 1 | 47 | 2 | 0 | 0 |
| Onbekend | 7 | 4 | 0 | 3 | 0 | 0 | 12 | 0 | 0 | 0 | 5 | 0 | 1 | 1 | 0 | 0 | 1 |
| Ursel | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 17 | 10 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 4 | 9 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 1 | 0 | 0 | 3 | 0 | 0 | 6 | 0 | 5 | 34 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 21 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 24 | 0 | 0 | 0 | 0 |
| Lede | 5 | 1 | 0 | 12 | 0 | 13 | 4 | 0 | 0 | 4 | 3 | 0 | 0 | 84 | 0 | 0 | 0 |
| Antropogeen | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 33 | 0 |
| Sint_Huibrechts_Hern | 1 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### LGBM Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Bolderberg | Merelbeke | Kwatrecht | Lede | Antropogeen | Diest | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 658 | 9 | 13 | 33 | 0 | 78 | 118 | 4 | 4 | 6 | 20 | 0 | 7 | 32 | 16 | 3 | 5 |
| Mont_Panisel | 4 | 157 | 8 | 39 | 0 | 2 | 1 | 5 | 6 | 3 | 0 | 0 | 17 | 2 | 0 | 0 | 0 |
| Aalbeke | 1 | 4 | 52 | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 15 | 35 | 17 | 141 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 8 | 3 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 62 | 9 | 0 | 13 | 0 | 277 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 28 | 1 | 3 | 0 |
| Onbekend | 13 | 1 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ursel | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 17 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Asse | 0 | 0 | 0 | 1 | 0 | 0 | 11 | 7 | 9 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 2 | 0 | 0 | 3 | 0 | 3 | 8 | 0 | 4 | 38 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| Bolderberg | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| Kwatrecht | 0 | 18 | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 19 | 0 | 0 | 0 | 0 |
| Lede | 7 | 2 | 0 | 5 | 0 | 29 | 6 | 0 | 0 | 5 | 0 | 0 | 0 | 104 | 0 | 1 | 0 |
| Antropogeen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Diest | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 2 | 0 | 0 | 1 | 0 | 30 | 0 |
| Sint_Huibrechts_Hern | 3 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
