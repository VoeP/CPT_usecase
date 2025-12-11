# PDS Model - Refactored


## Small building blocks we’ll reuse

``` r
# Aggregate features by groups with multiple statistics
agg_features <- function(dt, feat_cols, by_cols, stats, suffix = "") {
    dt[,
        {
            out <- vector("list", length(feat_cols) * length(stats))
            nm <- character(length(out))
            k <- 1

            for (stat_name in names(stats)) {
                fn <- stats[[stat_name]]
                for (v in feat_cols) {
                    out[[k]] <- fn(.SD[[v]], na.rm = TRUE)
                    nm[k] <- paste0(v, suffix, "_", stat_name)
                    k <- k + 1
                }
            }
            setNames(out, nm)
        },
        by = by_cols,
        .SDcols = feat_cols
    ]
}

# Tune model hyperparameters
tune_model <- function(workflow, folds, grid) {
    set.seed(42)
    tune_grid(
        workflow,
        resamples = folds,
        grid = grid,
        metrics = yardstick::metric_set(accuracy, mn_log_loss),
        control = control_grid(save_pred = TRUE)
    )
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
    id_col = "sondering_id") {
    # Predict
    preds_class <- predict(fitted_model, new_data = test_data)
    preds_prob <- predict(fitted_model, new_data = test_data, type = "prob")

    # Combine predictions with ID and truth
    pred_dt <- bind_cols(
        test_data[, c(id_col, "lithostrat_id"), with = FALSE],
        preds_class,
        preds_prob
    )

    # Align factor levels
    pred_dt[, lithostrat_id := factor(lithostrat_id,
        levels = unique(train_data$lithostrat_id)
    )]
    pred_dt[, .pred_class := factor(.pred_class,
        levels = unique(train_data$lithostrat_id)
    )]

    # Calculate metrics
    metrics <- yardstick::metric_set(accuracy, bal_accuracy)(
        data = pred_dt,
        truth = lithostrat_id,
        estimate = .pred_class
    )
    setDT(metrics)

    # Confusion matrix
    cm <- yardstick::conf_mat(pred_dt,
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

## Load and clean the data

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
#library(lightgbm)  # Fo

# Paths
main_folder <- "year2/Project_Data_Science/project"
data_folder <- here(main_folder, "data")
results_folder <- here(main_folder, "results")

# Load data
cpt_df <- read_parquet(
  paste(data_folder, "vw_cpt_brussels_params_completeset_20250318_remapped.parquet", sep = "/")
)
setDT(cpt_df)

# Filter and clean
cpt_df <- cpt_df[!is.na(lithostrat_id)]
litho_counts <- cpt_df |> unique(by = c("sondering_id", "lithostrat_id")) |>
  _[, .N, by = lithostrat_id]
rare_litho <- litho_counts[N < 5, lithostrat_id]
cpt_df <- cpt_df[!lithostrat_id %in% rare_litho]
```

## Prepare time-series signals and split IDs

``` r
# Trend extraction for time series
extract_trend <- function(x, freq = 5L) {
  x <- as.numeric(x)
  if (!is.finite(freq) ||
      freq < 2L ||
      length(na.omit(x)) < 2L || length(x) < 2L * freq)
    return(x)
  
  tryCatch({
    tr <- decompose(ts(x, frequency = freq))$trend
    if (all(is.na(tr)))
      x
    else
      ifelse(is.na(tr), x, tr)
  }, error = function(e)
    x, warning = function(w)
      x)
}

freq_from_depth <- function(z, default = 25L) {
  if (length(z) < 2L)
    return(default)
  dz <- stats::median(diff(sort(z)), na.rm = TRUE)
  if (!is.finite(dz) || dz <= 0)
    return(default)
  as.integer(max(1, round(1 / dz)))
}

# Stratified split by dominant class
group_strat_split <- function(dt,
                              id_col = "sondering_id",
                              y_col = "lithostrat_id",
                              prop = 0.7,
                              tol = 0.05,
                              max_tries = 200,
                              seed = 42) {
  setDT(dt)
  ids_lab <- dt[, .(mode_class = names(which.max(table(get(
    y_col
  ))))), by = id_col]
  full_counts <- dt[, .N, by = y_col]
  
  for (s in seq_len(max_tries)) {
    set.seed(seed + s - 1)
    spl <- initial_split(as.data.frame(ids_lab),
                         prop = prop,
                         strata = mode_class)
    tr_ids <- training(spl)[[id_col]]
    tr_dt <- dt[get(id_col) %in% tr_ids]
    tr_counts <- tr_dt[, .N, by = y_col]
    chk <- tr_counts[full_counts, on = y_col][, pct := i.N / N][, all(abs(pct - prop) <= tol)]
    
    if (isTRUE(chk)) {
      return(list(
        train_ids = tr_ids,
        test_ids = testing(spl)[[id_col]],
        seed = seed + s - 1
      ))
    }
  }
  list(
    train_ids = tr_ids,
    test_ids = testing(spl)[[id_col]],
    seed = seed + max_tries - 1
  )
}
```

## Split boreholes into train and test

``` r
cpt_unique <- cpt_df |> unique(by = c("sondering_id", "lithostrat_id"))
split_res <- group_strat_split(cpt_unique, prop = 0.7, tol = 0.05, seed = 42)
train_ids <- split_res$train_ids
test_ids <- split_res$test_ids
```

## Build features from CPT data

``` r
# Configuration
id_col <- "sondering_id"
depth_col <- "diepte"
label_col <- "lithostrat_id"
feat_cols <- c("qc", "fs", "rf", "qtn", "fr")
geog_cols <- c("diepte", "diepte_mtaw")
stats <- list(
  mean = mean,
  min = min,
  max = max,
  sd = sd
)

BIN_W <- 0.6  # bin width in meters

# Order and bin data
setorderv(cpt_df, c(id_col, depth_col))
cpt_df[, depth_bin := cut(diepte,
                          breaks = seq(0, max(diepte), by = BIN_W),
                          include.lowest = TRUE)]

# Get unique lithostrat per bin
litho_dept <- cpt_df[, .(sondering_id, lithostrat_id, depth_bin)] |>
  unique(by = c("sondering_id", "depth_bin"))

# Aggregate features: per bin and whole borehole
summaries_bin <- agg_features(cpt_df, feat_cols, c(id_col, "depth_bin"), stats)
summaries_whole <- agg_features(cpt_df, geog_cols, id_col, stats, suffix = "_whole")

# Merge all
dt <- merge(summaries_bin,
            summaries_whole,
            by = id_col,
            all.x = TRUE)
dt <- merge(
  dt,
  litho_dept,
  by.x = c(id_col, "depth_bin"),
  by.y = c("sondering_id", "depth_bin"),
  all.x = TRUE
)
setnames(dt, "depth_bin", depth_col)
dt <- dt %>% select(sondering_id, diepte, lithostrat_id, everything())
dt[, diepte := as.factor(diepte)]

# Split data
train_dt <- dt[sondering_id %in% train_ids]
test_dt  <- dt[sondering_id %in% test_ids]
```

## Define the recipe and models

``` r
# Formula
nms_feat <- setdiff(names(dt), c(id_col, "lithostrat_id", 
                                 depth_col))
model_formula <- as.formula(paste("lithostrat_id ~", 
                                  paste(c(nms_feat, id_col),
                                        collapse = " + ")))

# Shared recipe
base_recipe <- recipe(model_formula, data = train_dt) |>
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
    set_engine("ranger", importance = "permutation", splitrule = "extratrees"),
  
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
  workflow() |> add_recipe(base_recipe) |> add_model(spec)
})

# CV folds
set.seed(42)
folds <- group_vfold_cv(train_dt, 
                        group = sondering_id,
                        v = 10)
```

## Tune and train the models

``` r
# Generate parameter grids (automatically for all models)
size_tune = 10
grids <- lapply(names(workflows), function(model_name) {
    if (model_name == "rf") {
        extract_parameter_set_dials(workflows[[model_name]]) |>
            finalize(train_dt |> dplyr::select(-lithostrat_id)) |>
            grid_latin_hypercube(size = size_tune)
    } else {
        extract_parameter_set_dials(workflows[[model_name]]) |>
            finalize(train_dt |> dplyr::select(-lithostrat_id, -diepte)) |>
            grid_latin_hypercube(size = size_tune)
    }
})
names(grids) <- names(workflows)

# Train and evaluate ALL models in parallel
# Train and evaluate ALL models sequentially
results <- list()

for (model_name in names(workflows)) {
    cat("\nTraining", toupper(model_name), "\n")

    # Tune hyperparameters
    cat("Tuning hyperparameters...\n")
    tune_res <- tune_model(workflows[[model_name]], folds, grids[[model_name]])

    #  Fit best model
    cat("Fitting best model...\n")
    best_model <- fit_best_model(workflows[[model_name]], tune_res, train_dt)

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


    Training XGB 
    Tuning hyperparameters...
    Fitting best model...
    Evaluating on test set...
    Completed

    Training RF 
    Tuning hyperparameters...
    Fitting best model...
    Evaluating on test set...
    Completed

    Training LGBM 
    Tuning hyperparameters...
    Fitting best model...
    Evaluating on test set...
    Completed

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
| 2 | 142 | 36 | 13 | 0.0069143 | 25.4243133 | 0.1620596 | accuracy | multiclass | 0.3279641 | 10 | 0.0220246 | pre0_mod01_post0 |
| 2 | 142 | 36 | 13 | 0.0069143 | 25.4243133 | 0.1620596 | mn_log_loss | multiclass | 2.3431783 | 10 | 0.0241244 | pre0_mod01_post0 |
| 5 | 1080 | 10 | 8 | 0.2203716 | 0.0000701 | 0.4844747 | accuracy | multiclass | 0.6410861 | 10 | 0.0207136 | pre0_mod02_post0 |
| 5 | 1080 | 10 | 8 | 0.2203716 | 0.0000701 | 0.4844747 | mn_log_loss | multiclass | 1.4030249 | 10 | 0.1171950 | pre0_mod02_post0 |
| 9 | 520 | 24 | 15 | 0.0231266 | 0.0000000 | 0.3585684 | accuracy | multiclass | 0.6221029 | 10 | 0.0173561 | pre0_mod03_post0 |
| 9 | 520 | 24 | 15 | 0.0231266 | 0.0000000 | 0.3585684 | mn_log_loss | multiclass | 1.2394361 | 10 | 0.0457896 | pre0_mod03_post0 |
| 10 | 1305 | 7 | 11 | 0.0948794 | 0.0000000 | 0.8295328 | accuracy | multiclass | 0.6549240 | 10 | 0.0230142 | pre0_mod04_post0 |
| 10 | 1305 | 7 | 11 | 0.0948794 | 0.0000000 | 0.8295328 | mn_log_loss | multiclass | 1.3757102 | 10 | 0.1192036 | pre0_mod04_post0 |
| 15 | 1518 | 17 | 6 | 0.0115987 | 0.0043856 | 0.7009809 | accuracy | multiclass | 0.6495061 | 10 | 0.0226476 | pre0_mod05_post0 |
| 15 | 1518 | 17 | 6 | 0.0115987 | 0.0043856 | 0.7009809 | mn_log_loss | multiclass | 1.1382292 | 10 | 0.0752830 | pre0_mod05_post0 |
| 17 | 889 | 19 | 1 | 0.0013178 | 0.0518794 | 0.7810448 | accuracy | multiclass | 0.4672716 | 10 | 0.0193917 | pre0_mod06_post0 |
| 17 | 889 | 19 | 1 | 0.0013178 | 0.0518794 | 0.7810448 | mn_log_loss | multiclass | 2.0641570 | 10 | 0.0194358 | pre0_mod06_post0 |
| 18 | 715 | 5 | 5 | 0.0467467 | 0.0000001 | 0.9768683 | accuracy | multiclass | 0.6561476 | 10 | 0.0213397 | pre0_mod07_post0 |
| 18 | 715 | 5 | 5 | 0.0467467 | 0.0000001 | 0.9768683 | mn_log_loss | multiclass | 1.3062938 | 10 | 0.1013974 | pre0_mod07_post0 |
| 23 | 289 | 30 | 3 | 0.1524093 | 1.8039472 | 0.4230694 | accuracy | multiclass | 0.6032939 | 10 | 0.0193294 | pre0_mod08_post0 |
| 23 | 289 | 30 | 3 | 0.1524093 | 1.8039472 | 0.4230694 | mn_log_loss | multiclass | 1.2986154 | 10 | 0.0569874 | pre0_mod08_post0 |
| 24 | 1752 | 27 | 9 | 0.0039015 | 0.0000093 | 0.6031338 | accuracy | multiclass | 0.6244861 | 10 | 0.0196357 | pre0_mod09_post0 |
| 24 | 1752 | 27 | 9 | 0.0039015 | 0.0000093 | 0.6031338 | mn_log_loss | multiclass | 1.2027964 | 10 | 0.0492835 | pre0_mod09_post0 |
| 27 | 1952 | 35 | 8 | 0.0020291 | 0.0000007 | 0.2144798 | accuracy | multiclass | 0.5391608 | 10 | 0.0176860 | pre0_mod10_post0 |
| 27 | 1952 | 35 | 8 | 0.0020291 | 0.0000007 | 0.2144798 | mn_log_loss | multiclass | 1.6410281 | 10 | 0.0397129 | pre0_mod10_post0 |

#### RF Tuning Metrics

| mtry | trees | min_n | .metric     | .estimator |      mean |   n |   std_err | .config          |
|-----:|------:|------:|:------------|:-----------|----------:|----:|----------:|:-----------------|
|    2 |  1813 |    21 | accuracy    | multiclass | 0.6334759 |  10 | 0.0171219 | pre0_mod01_post0 |
|    2 |  1813 |    21 | mn_log_loss | multiclass | 1.2229762 |  10 | 0.0399553 | pre0_mod01_post0 |
|    6 |   162 |     3 | accuracy    | multiclass | 0.6489262 |  10 | 0.0234454 | pre0_mod02_post0 |
|    6 |   162 |     3 | mn_log_loss | multiclass | 1.2787740 |  10 | 0.0762256 | pre0_mod02_post0 |
|    8 |  1460 |    18 | accuracy    | multiclass | 0.6487253 |  10 | 0.0202067 | pre0_mod03_post0 |
|    8 |  1460 |    18 | mn_log_loss | multiclass | 1.1440401 |  10 | 0.0495575 | pre0_mod03_post0 |
|   11 |   270 |    30 | accuracy    | multiclass | 0.6392342 |  10 | 0.0173226 | pre0_mod04_post0 |
|   11 |   270 |    30 | mn_log_loss | multiclass | 1.1865631 |  10 | 0.0562034 | pre0_mod04_post0 |
|   15 |   895 |     9 | accuracy    | multiclass | 0.6539260 |  10 | 0.0190909 | pre0_mod05_post0 |
|   15 |   895 |     9 | mn_log_loss | multiclass | 1.1448346 |  10 | 0.0678718 | pre0_mod05_post0 |
|   16 |  1115 |    33 | accuracy    | multiclass | 0.6447371 |  10 | 0.0186764 | pre0_mod06_post0 |
|   16 |  1115 |    33 | mn_log_loss | multiclass | 1.1494267 |  10 | 0.0451787 | pre0_mod06_post0 |
|   21 |  1386 |    40 | accuracy    | multiclass | 0.6380275 |  10 | 0.0170050 | pre0_mod07_post0 |
|   21 |  1386 |    40 | mn_log_loss | multiclass | 1.1511463 |  10 | 0.0414543 | pre0_mod07_post0 |
|   22 |   448 |    28 | accuracy    | multiclass | 0.6468345 |  10 | 0.0190494 | pre0_mod08_post0 |
|   22 |   448 |    28 | mn_log_loss | multiclass | 1.1340466 |  10 | 0.0456374 | pre0_mod08_post0 |
|   27 |  1787 |    15 | accuracy    | multiclass | 0.6524044 |  10 | 0.0187935 | pre0_mod09_post0 |
|   27 |  1787 |    15 | mn_log_loss | multiclass | 1.1221284 |  10 | 0.0552048 | pre0_mod09_post0 |
|   28 |   696 |    12 | accuracy    | multiclass | 0.6508174 |  10 | 0.0198469 | pre0_mod10_post0 |
|   28 |   696 |    12 | mn_log_loss | multiclass | 1.1466402 |  10 | 0.0680259 | pre0_mod10_post0 |

#### LGBM Tuning Metrics

| mtry | trees | min_n | tree_depth | learn_rate | loss_reduction | sample_size | .metric | .estimator | mean | n | std_err | .config |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|
| 1 | 681 | 30 | 3 | 0.0009739 | 0.0000000 | 0.6751268 | accuracy | multiclass | 0.3951543 | 10 | 0.0152726 | pre0_mod01_post0 |
| 1 | 681 | 30 | 3 | 0.0009739 | 0.0000000 | 0.6751268 | mn_log_loss | multiclass | 1.7872740 | 10 | 0.0206535 | pre0_mod01_post0 |
| 5 | 425 | 34 | 5 | 0.0001508 | 0.0001621 | 0.4821682 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod02_post0 |
| 5 | 425 | 34 | 5 | 0.0001508 | 0.0001621 | 0.4821682 | mn_log_loss | multiclass | 2.1350338 | 10 | 0.0339054 | pre0_mod02_post0 |
| 7 | 1385 | 4 | 8 | 0.0097062 | 0.0000000 | 0.5521302 | accuracy | multiclass | 0.6652208 | 10 | 0.0246727 | pre0_mod03_post0 |
| 7 | 1385 | 4 | 8 | 0.0097062 | 0.0000000 | 0.5521302 | mn_log_loss | multiclass | 1.8150981 | 10 | 0.1501743 | pre0_mod03_post0 |
| 10 | 1090 | 7 | 13 | 0.0000002 | 0.0000120 | 0.2798975 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod04_post0 |
| 10 | 1090 | 7 | 13 | 0.0000002 | 0.0000120 | 0.2798975 | mn_log_loss | multiclass | 2.3074915 | 10 | 0.0460270 | pre0_mod04_post0 |
| 12 | 1993 | 39 | 5 | 0.0364256 | 1.5716885 | 0.3012792 | accuracy | multiclass | 0.6512555 | 10 | 0.0212686 | pre0_mod05_post0 |
| 12 | 1993 | 39 | 5 | 0.0364256 | 1.5716885 | 0.3012792 | mn_log_loss | multiclass | 1.2072123 | 10 | 0.0890532 | pre0_mod05_post0 |
| 16 | 1763 | 22 | 8 | 0.0000000 | 0.0000009 | 0.9632932 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod06_post0 |
| 16 | 1763 | 22 | 8 | 0.0000000 | 0.0000009 | 0.9632932 | mn_log_loss | multiclass | 2.3085580 | 10 | 0.0461184 | pre0_mod06_post0 |
| 20 | 256 | 13 | 10 | 0.0000000 | 0.0083775 | 0.4477214 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod07_post0 |
| 20 | 256 | 13 | 10 | 0.0000000 | 0.0083775 | 0.4477214 | mn_log_loss | multiclass | 2.3085290 | 10 | 0.0461162 | pre0_mod07_post0 |
| 21 | 864 | 18 | 11 | 0.0000000 | 0.0000000 | 0.1169924 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod08_post0 |
| 21 | 864 | 18 | 11 | 0.0000000 | 0.0000000 | 0.1169924 | mn_log_loss | multiclass | 2.3085568 | 10 | 0.0461184 | pre0_mod08_post0 |
| 25 | 1529 | 28 | 1 | 0.0000240 | 22.5802711 | 0.8632164 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod09_post0 |
| 25 | 1529 | 28 | 1 | 0.0000240 | 22.5802711 | 0.8632164 | mn_log_loss | multiclass | 2.2661551 | 10 | 0.0431179 | pre0_mod09_post0 |
| 28 | 94 | 16 | 14 | 0.0000005 | 0.0216401 | 0.7646310 | accuracy | multiclass | 0.3272065 | 10 | 0.0218134 | pre0_mod10_post0 |
| 28 | 94 | 16 | 14 | 0.0000005 | 0.0216401 | 0.7646310 | mn_log_loss | multiclass | 2.3083248 | 10 | 0.0461016 | pre0_mod10_post0 |

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
| rf    | accuracy    | 0.6457108 | 0.0190559 |
| xgb   | accuracy    | 0.5785943 | 0.0203139 |
| lgbm  | accuracy    | 0.4002076 | 0.0213908 |
| lgbm  | mn_log_loss | 2.0752233 | 0.0567386 |
| xgb   | mn_log_loss | 1.5012469 | 0.0648413 |
| rf    | mn_log_loss | 1.1680576 | 0.0545315 |

### How they perform on new data

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

| .metric      | .estimator | .estimate | model |
|:-------------|:-----------|----------:|:------|
| accuracy     | multiclass | 0.6300926 | xgb   |
| bal_accuracy | macro      | 0.7245603 | xgb   |
| accuracy     | multiclass | 0.6523148 | rf    |
| bal_accuracy | macro      | 0.7597834 | rf    |
| accuracy     | multiclass | 0.6268519 | lgbm  |
| bal_accuracy | macro      | 0.7230215 | lgbm  |

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

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Lede | Kwatrecht | Antropogeen | Merelbeke | Diest | Bolderberg | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 591 | 3 | 4 | 18 | 1 | 34 | 99 | 0 | 0 | 4 | 9 | 0 | 5 | 0 | 3 | 15 | 11 |
| Mont_Panisel | 7 | 167 | 13 | 12 | 0 | 1 | 20 | 1 | 4 | 0 | 16 | 9 | 0 | 2 | 0 | 0 | 0 |
| Aalbeke | 9 | 20 | 74 | 8 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 19 | 35 | 11 | 139 | 0 | 6 | 1 | 0 | 0 | 2 | 7 | 1 | 0 | 0 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 8 | 0 | 0 | 2 | 2 | 118 | 3 | 0 | 0 | 2 | 3 | 0 | 1 | 0 | 3 | 0 | 4 |
| Onbekend | 44 | 11 | 0 | 1 | 0 | 1 | 28 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 2 | 2 | 0 |
| Ursel | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 17 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Asse | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 8 | 22 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 0 | 0 | 0 | 4 | 0 | 3 | 12 | 0 | 3 | 44 | 5 | 0 | 0 | 0 | 20 | 0 | 29 |
| Lede | 17 | 3 | 0 | 7 | 0 | 36 | 19 | 0 | 0 | 29 | 117 | 1 | 0 | 0 | 23 | 0 | 5 |
| Kwatrecht | 2 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| Antropogeen | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| Diest | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

#### RF Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Lede | Kwatrecht | Antropogeen | Merelbeke | Diest | Bolderberg | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 564 | 3 | 3 | 20 | 1 | 25 | 68 | 1 | 1 | 10 | 6 | 0 | 4 | 0 | 0 | 17 | 4 |
| Mont_Panisel | 9 | 185 | 11 | 8 | 0 | 0 | 25 | 1 | 5 | 2 | 24 | 24 | 0 | 1 | 0 | 1 | 0 |
| Aalbeke | 8 | 17 | 70 | 8 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| Mons_en_Pevele | 14 | 11 | 7 | 138 | 0 | 6 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Maldegem | 1 | 0 | 0 | 0 | 6 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 12 | 0 | 0 | 4 | 0 | 109 | 4 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| Onbekend | 86 | 23 | 8 | 4 | 0 | 5 | 61 | 0 | 3 | 0 | 5 | 5 | 0 | 0 | 0 | 3 | 11 |
| Ursel | 0 | 0 | 5 | 0 | 0 | 0 | 1 | 16 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Asse | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 5 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Wemmel | 0 | 0 | 0 | 4 | 0 | 3 | 9 | 0 | 4 | 48 | 5 | 0 | 0 | 0 | 3 | 0 | 26 |
| Lede | 11 | 3 | 0 | 2 | 0 | 42 | 17 | 0 | 0 | 26 | 115 | 0 | 0 | 0 | 0 | 0 | 8 |
| Kwatrecht | 1 | 4 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 0 |
| Antropogeen | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| Diest | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

#### LGBM Confusion Matrix

|  | Quartair | Mont_Panisel | Aalbeke | Mons_en_Pevele | Maldegem | Brussel | Onbekend | Ursel | Asse | Wemmel | Lede | Kwatrecht | Antropogeen | Merelbeke | Diest | Bolderberg | Sint_Huibrechts_Hern |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quartair | 595 | 6 | 6 | 17 | 1 | 35 | 102 | 0 | 0 | 4 | 13 | 0 | 5 | 0 | 2 | 17 | 20 |
| Mont_Panisel | 11 | 168 | 12 | 12 | 0 | 1 | 21 | 0 | 3 | 1 | 14 | 11 | 0 | 1 | 0 | 0 | 0 |
| Aalbeke | 6 | 18 | 73 | 9 | 0 | 0 | 0 | 2 | 7 | 0 | 0 | 7 | 0 | 0 | 0 | 1 | 0 |
| Mons_en_Pevele | 17 | 33 | 10 | 141 | 0 | 7 | 0 | 0 | 1 | 3 | 7 | 5 | 0 | 0 | 0 | 0 | 0 |
| Maldegem | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Brussel | 11 | 0 | 0 | 3 | 3 | 115 | 3 | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 2 | 0 | 0 |
| Onbekend | 41 | 8 | 0 | 1 | 0 | 0 | 23 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 5 | 0 | 0 |
| Ursel | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 18 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Asse | 1 | 0 | 2 | 1 | 0 | 0 | 4 | 7 | 21 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Wemmel | 0 | 0 | 0 | 2 | 0 | 1 | 13 | 0 | 2 | 42 | 5 | 0 | 0 | 0 | 23 | 0 | 25 |
| Lede | 15 | 7 | 0 | 5 | 0 | 40 | 20 | 0 | 0 | 30 | 115 | 1 | 0 | 0 | 19 | 0 | 3 |
| Kwatrecht | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| Antropogeen | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Merelbeke | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 0 | 0 | 0 |
| Diest | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bolderberg | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Sint_Huibrechts_Hern | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |

### Sample predictions

- A quick peek at the first few predictions per model: the ID, the true
  label, and the predicted class.
- Useful for sanity checks and understanding typical successes or
  misses.

``` r
# Display predictions for all models (first 10 rows each)
for (model_name in names(results)) {
  cat("\n####", toupper(model_name), "Predictions\n")
  print(kable(head(results[[model_name]]$predictions[, .(sondering_id, lithostrat_id, .pred_class)], 10)))
}
```


    #### XGB Predictions


    | sondering_id|lithostrat_id |.pred_class |
    |------------:|:-------------|:-----------|
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |

    #### RF Predictions


    | sondering_id|lithostrat_id |.pred_class |
    |------------:|:-------------|:-----------|
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |

    #### LGBM Predictions


    | sondering_id|lithostrat_id |.pred_class |
    |------------:|:-------------|:-----------|
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |
    |          314|Quartair      |Quartair    |

``` r
rf_pred <- results[[model_name]]$predictions
kable(head(rf_pred, 100))
```

| sondering_id | lithostrat_id | .pred_class | .pred_Aalbeke | .pred_Antropogeen | .pred_Asse | .pred_Bolderberg | .pred_Brussel | .pred_Diest | .pred_Kwatrecht | .pred_Lede | .pred_Maldegem | .pred_Merelbeke | .pred_Mons_en_Pevele | .pred_Mont_Panisel | .pred_Onbekend | .pred_Quartair | .pred_Sint_Huibrechts_Hern | .pred_Ursel | .pred_Wemmel |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 314 | Quartair | Quartair | 0.0021769 | 0.0000923 | 0.0001421 | 0.0001240 | 0.0000177 | 0.0001267 | 0.0001577 | 0.0004337 | 0.0001282 | 0.0000836 | 0.0035779 | 0.0036680 | 0.0157595 | 0.9719326 | 0.0001031 | 0.0014484 | 0.0000276 |
| 314 | Quartair | Quartair | 0.0000684 | 0.0000167 | 0.0000029 | 0.0000152 | 0.0000101 | 0.0000267 | 0.0000117 | 0.0004442 | 0.0000276 | 0.0000474 | 0.0011764 | 0.0002256 | 0.0028420 | 0.9950453 | 0.0000150 | 0.0000196 | 0.0000052 |
| 314 | Quartair | Quartair | 0.0021443 | 0.0001407 | 0.0002649 | 0.0000899 | 0.0000567 | 0.0001489 | 0.0000857 | 0.0008918 | 0.0001294 | 0.0001863 | 0.0044609 | 0.0085028 | 0.0110044 | 0.9709521 | 0.0000513 | 0.0008667 | 0.0000232 |
| 314 | Quartair | Quartair | 0.0009302 | 0.0000177 | 0.0000044 | 0.0000200 | 0.0000075 | 0.0000250 | 0.0000283 | 0.0002974 | 0.0000682 | 0.0000222 | 0.0032592 | 0.0013548 | 0.0017968 | 0.9921181 | 0.0000108 | 0.0000310 | 0.0000084 |
| 314 | Quartair | Quartair | 0.0024099 | 0.0000124 | 0.0000064 | 0.0000302 | 0.0000049 | 0.0000200 | 0.0000503 | 0.0000182 | 0.0001045 | 0.0000676 | 0.0057173 | 0.0002575 | 0.0014814 | 0.9897554 | 0.0000151 | 0.0000435 | 0.0000053 |
| 314 | Quartair | Quartair | 0.0763047 | 0.0000123 | 0.0000251 | 0.0000182 | 0.0000037 | 0.0000136 | 0.0001058 | 0.0000091 | 0.0000469 | 0.0002430 | 0.0201303 | 0.0000947 | 0.0018380 | 0.9009204 | 0.0000087 | 0.0002218 | 0.0000037 |
| 314 | Quartair | Quartair | 0.0475541 | 0.0000066 | 0.0000184 | 0.0000053 | 0.0000017 | 0.0000069 | 0.0000723 | 0.0000087 | 0.0000199 | 0.0000989 | 0.0133587 | 0.0000756 | 0.0006571 | 0.9380524 | 0.0000051 | 0.0000568 | 0.0000015 |
| 314 | Quartair | Quartair | 0.0221342 | 0.0000087 | 0.0000253 | 0.0000176 | 0.0000037 | 0.0000113 | 0.0000867 | 0.0000064 | 0.0000324 | 0.0001957 | 0.0165921 | 0.0000710 | 0.0009861 | 0.9595677 | 0.0000090 | 0.0002485 | 0.0000036 |
| 314 | Quartair | Quartair | 0.0008743 | 0.0000342 | 0.0000114 | 0.0000237 | 0.0000612 | 0.0000364 | 0.0000418 | 0.0012521 | 0.0000730 | 0.0000256 | 0.0107168 | 0.0017785 | 0.0015455 | 0.9834452 | 0.0000441 | 0.0000173 | 0.0000188 |
| 314 | Quartair | Quartair | 0.0020209 | 0.0000687 | 0.0000703 | 0.0000863 | 0.0000425 | 0.0001169 | 0.0001998 | 0.0007272 | 0.0001188 | 0.0001090 | 0.0221695 | 0.0043165 | 0.0087256 | 0.9610025 | 0.0000952 | 0.0001116 | 0.0000189 |
| 314 | Quartair | Quartair | 0.0332046 | 0.0000192 | 0.0000151 | 0.0000116 | 0.0000120 | 0.0000161 | 0.0001905 | 0.0000597 | 0.0000330 | 0.0000764 | 0.0168566 | 0.0000834 | 0.0012579 | 0.9479722 | 0.0000121 | 0.0001709 | 0.0000086 |
| 314 | Quartair | Quartair | 0.0141087 | 0.0000125 | 0.0000497 | 0.0000095 | 0.0000037 | 0.0000124 | 0.0002018 | 0.0000333 | 0.0000242 | 0.0000162 | 0.0498779 | 0.0002682 | 0.0018856 | 0.9334326 | 0.0000103 | 0.0000519 | 0.0000016 |
| 314 | Quartair | Quartair | 0.0105175 | 0.0000692 | 0.0001338 | 0.0000399 | 0.0000218 | 0.0000671 | 0.0001215 | 0.0001670 | 0.0001425 | 0.0001170 | 0.0374485 | 0.0022315 | 0.0065824 | 0.9422424 | 0.0000458 | 0.0000439 | 0.0000084 |
| 314 | Quartair | Quartair | 0.0008874 | 0.0003055 | 0.0000495 | 0.0003792 | 0.0004381 | 0.0001971 | 0.0001497 | 0.0362050 | 0.0011753 | 0.0001897 | 0.3242235 | 0.0020751 | 0.0050638 | 0.6281657 | 0.0003014 | 0.0000582 | 0.0001361 |
| 314 | Quartair | Mons_en_Pevele | 0.0033119 | 0.0003043 | 0.0000447 | 0.0002404 | 0.0019070 | 0.0005778 | 0.0001795 | 0.0124892 | 0.0022071 | 0.0002114 | 0.6610694 | 0.0041482 | 0.0029408 | 0.3101608 | 0.0000840 | 0.0000487 | 0.0000748 |
| 314 | Quartair | Mons_en_Pevele | 0.0051103 | 0.0003092 | 0.0000716 | 0.0002837 | 0.0014474 | 0.0002656 | 0.0001821 | 0.0108591 | 0.0019043 | 0.0002713 | 0.6639948 | 0.0053399 | 0.0039293 | 0.3056698 | 0.0001126 | 0.0000670 | 0.0001819 |
| 314 | Quartair | Mons_en_Pevele | 0.0007113 | 0.0002258 | 0.0000286 | 0.0001360 | 0.0019888 | 0.0005295 | 0.0001435 | 0.0022908 | 0.0010791 | 0.0001294 | 0.7658757 | 0.0023767 | 0.0016132 | 0.2227378 | 0.0000557 | 0.0000298 | 0.0000484 |
| 314 | Quartair | Mons_en_Pevele | 0.0076438 | 0.0002787 | 0.0000444 | 0.0004803 | 0.0003538 | 0.0003629 | 0.0003000 | 0.0127594 | 0.0012095 | 0.0002057 | 0.4913520 | 0.0058212 | 0.0051525 | 0.4722417 | 0.0015981 | 0.0001487 | 0.0000473 |
| 314 | Quartair | Mons_en_Pevele | 0.0016963 | 0.0000564 | 0.0000342 | 0.0000539 | 0.0002153 | 0.0000723 | 0.0001301 | 0.0030155 | 0.0001721 | 0.0000907 | 0.9709373 | 0.0012700 | 0.0006035 | 0.0213604 | 0.0000903 | 0.0000204 | 0.0001813 |
| 314 | Quartair | Mons_en_Pevele | 0.0005085 | 0.0001351 | 0.0000353 | 0.0000293 | 0.0008419 | 0.0001647 | 0.0000309 | 0.0007855 | 0.0001589 | 0.0000365 | 0.9565006 | 0.0022351 | 0.0031667 | 0.0352531 | 0.0000135 | 0.0000084 | 0.0000960 |
| 314 | Quartair | Mons_en_Pevele | 0.0004516 | 0.0001504 | 0.0000579 | 0.0000559 | 0.0011732 | 0.0003498 | 0.0000560 | 0.0010171 | 0.0002553 | 0.0000660 | 0.9435073 | 0.0026811 | 0.0032208 | 0.0468250 | 0.0000260 | 0.0000152 | 0.0000915 |
| 314 | Quartair | Mons_en_Pevele | 0.0008386 | 0.0001318 | 0.0000233 | 0.0000466 | 0.0016976 | 0.0002616 | 0.0000479 | 0.0007538 | 0.0002900 | 0.0000579 | 0.9377410 | 0.0031688 | 0.0021657 | 0.0526659 | 0.0000367 | 0.0000133 | 0.0000593 |
| 314 | Quartair | Mons_en_Pevele | 0.0057396 | 0.0000583 | 0.0000171 | 0.0000220 | 0.0001720 | 0.0001506 | 0.0000690 | 0.0005999 | 0.0002374 | 0.0000623 | 0.9229641 | 0.0010828 | 0.0004897 | 0.0681653 | 0.0000845 | 0.0000218 | 0.0000637 |
| 314 | Mont_Panisel | Aalbeke | 0.6119653 | 0.0000121 | 0.0000990 | 0.0000206 | 0.0000310 | 0.0001154 | 0.0002220 | 0.0000102 | 0.0000449 | 0.0318140 | 0.2985758 | 0.0549633 | 0.0001398 | 0.0009396 | 0.0001414 | 0.0008139 | 0.0000917 |
| 314 | Mont_Panisel | Aalbeke | 0.4999271 | 0.0000083 | 0.0000542 | 0.0000121 | 0.0000178 | 0.0000580 | 0.0004553 | 0.0000063 | 0.0000326 | 0.0329298 | 0.4540274 | 0.0110307 | 0.0001274 | 0.0007833 | 0.0000900 | 0.0004152 | 0.0000244 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0052404 | 0.0000232 | 0.0000909 | 0.0000096 | 0.0000755 | 0.0001068 | 0.0003188 | 0.0000297 | 0.0000382 | 0.0057303 | 0.8910116 | 0.0955783 | 0.0002228 | 0.0008312 | 0.0000738 | 0.0001869 | 0.0004320 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0009913 | 0.0000067 | 0.0000862 | 0.0000033 | 0.0000229 | 0.0000224 | 0.0000084 | 0.0000497 | 0.0000135 | 0.0000831 | 0.9390079 | 0.0590546 | 0.0000185 | 0.0001281 | 0.0000249 | 0.0000054 | 0.0004732 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0316978 | 0.0000086 | 0.0000514 | 0.0000141 | 0.0000234 | 0.0000565 | 0.0000284 | 0.0000061 | 0.0000271 | 0.0013433 | 0.9416891 | 0.0245354 | 0.0000659 | 0.0002609 | 0.0000759 | 0.0000416 | 0.0000745 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0008544 | 0.0000036 | 0.0001632 | 0.0000074 | 0.0000151 | 0.0000272 | 0.0000116 | 0.0000089 | 0.0000136 | 0.0001174 | 0.8466217 | 0.1516317 | 0.0000206 | 0.0001464 | 0.0000455 | 0.0000117 | 0.0003001 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0003279 | 0.0000033 | 0.0000716 | 0.0000040 | 0.0000200 | 0.0000252 | 0.0000210 | 0.0000120 | 0.0000112 | 0.0002696 | 0.9439302 | 0.0550227 | 0.0000321 | 0.0001018 | 0.0000374 | 0.0000087 | 0.0001015 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0008696 | 0.0000046 | 0.0000304 | 0.0000090 | 0.0000218 | 0.0000322 | 0.0000446 | 0.0000089 | 0.0000137 | 0.0003971 | 0.9177703 | 0.0803647 | 0.0000354 | 0.0002466 | 0.0000482 | 0.0000157 | 0.0000871 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0036852 | 0.0000136 | 0.0000720 | 0.0000150 | 0.0001153 | 0.0001047 | 0.0003395 | 0.0000339 | 0.0000357 | 0.0105957 | 0.8191212 | 0.1645375 | 0.0001210 | 0.0006651 | 0.0000557 | 0.0001698 | 0.0003192 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0018917 | 0.0000066 | 0.0000576 | 0.0000096 | 0.0000387 | 0.0000204 | 0.0002035 | 0.0000201 | 0.0000144 | 0.0002383 | 0.9491379 | 0.0481222 | 0.0000633 | 0.0001129 | 0.0000443 | 0.0000056 | 0.0000129 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0077010 | 0.0000134 | 0.0003567 | 0.0000110 | 0.0000771 | 0.0000415 | 0.0002011 | 0.0002031 | 0.0000196 | 0.0003588 | 0.9213614 | 0.0692769 | 0.0001111 | 0.0001345 | 0.0000431 | 0.0000185 | 0.0000714 |
| 314 | Mont_Panisel | Aalbeke | 0.8833669 | 0.0000012 | 0.0000507 | 0.0000018 | 0.0000010 | 0.0000034 | 0.0000034 | 0.0000009 | 0.0000028 | 0.0002052 | 0.1158412 | 0.0003969 | 0.0000154 | 0.0000358 | 0.0000135 | 0.0000591 | 0.0000007 |
| 314 | Mont_Panisel | Mons_en_Pevele | 0.0547479 | 0.0000052 | 0.0001677 | 0.0000038 | 0.0000073 | 0.0000206 | 0.0000307 | 0.0000065 | 0.0000098 | 0.0009496 | 0.9253866 | 0.0184833 | 0.0000430 | 0.0000793 | 0.0000220 | 0.0000225 | 0.0000143 |
| 314 | Aalbeke | Aalbeke | 0.6114132 | 0.0000049 | 0.0000548 | 0.0000037 | 0.0000033 | 0.0000112 | 0.0001679 | 0.0000151 | 0.0000071 | 0.0001980 | 0.3846231 | 0.0031684 | 0.0000526 | 0.0002187 | 0.0000287 | 0.0000276 | 0.0000019 |
| 314 | Aalbeke | Aalbeke | 0.6371268 | 0.0000024 | 0.0000235 | 0.0000020 | 0.0000024 | 0.0000060 | 0.0001438 | 0.0000065 | 0.0000049 | 0.0000728 | 0.3613009 | 0.0010155 | 0.0000346 | 0.0002248 | 0.0000193 | 0.0000130 | 0.0000009 |
| 314 | Aalbeke | Aalbeke | 0.7135817 | 0.0000012 | 0.0000271 | 0.0000011 | 0.0000009 | 0.0000030 | 0.0000048 | 0.0000030 | 0.0000025 | 0.0001613 | 0.2855311 | 0.0005838 | 0.0000171 | 0.0000638 | 0.0000079 | 0.0000091 | 0.0000006 |
| 314 | Aalbeke | Aalbeke | 0.7923539 | 0.0000009 | 0.0000156 | 0.0000010 | 0.0000005 | 0.0000023 | 0.0000016 | 0.0000012 | 0.0000018 | 0.0004033 | 0.2069076 | 0.0002260 | 0.0000128 | 0.0000516 | 0.0000079 | 0.0000115 | 0.0000005 |
| 314 | Aalbeke | Mons_en_Pevele | 0.4816077 | 0.0000012 | 0.0000317 | 0.0000028 | 0.0000011 | 0.0000035 | 0.0000046 | 0.0000024 | 0.0000035 | 0.0004243 | 0.5176201 | 0.0001434 | 0.0000175 | 0.0000857 | 0.0000135 | 0.0000366 | 0.0000007 |
| 315 | Quartair | Quartair | 0.0007789 | 0.0001477 | 0.0000439 | 0.0001447 | 0.0000198 | 0.0001721 | 0.0010993 | 0.0008231 | 0.0002303 | 0.0001468 | 0.0051987 | 0.0110821 | 0.0051793 | 0.9746386 | 0.0002456 | 0.0000272 | 0.0000218 |
| 315 | Quartair | Quartair | 0.0017915 | 0.0000740 | 0.0000146 | 0.0001418 | 0.0000119 | 0.0000891 | 0.0001635 | 0.0004334 | 0.0000895 | 0.0000835 | 0.0014120 | 0.0036130 | 0.0029962 | 0.9889131 | 0.0001432 | 0.0000220 | 0.0000077 |
| 315 | Quartair | Quartair | 0.0016020 | 0.0000168 | 0.0000075 | 0.0000061 | 0.0000039 | 0.0000152 | 0.0000197 | 0.0000253 | 0.0000307 | 0.0000259 | 0.0008394 | 0.0005568 | 0.0008095 | 0.9960209 | 0.0000058 | 0.0000139 | 0.0000007 |
| 315 | Quartair | Quartair | 0.0002506 | 0.0000083 | 0.0000014 | 0.0000073 | 0.0000028 | 0.0000128 | 0.0000859 | 0.0000169 | 0.0000234 | 0.0000161 | 0.0005096 | 0.0006624 | 0.0010940 | 0.9972716 | 0.0000041 | 0.0000320 | 0.0000007 |
| 315 | Quartair | Quartair | 0.0004880 | 0.0000286 | 0.0000055 | 0.0000315 | 0.0000232 | 0.0000511 | 0.0000197 | 0.0003413 | 0.0000336 | 0.0000422 | 0.0010773 | 0.0022668 | 0.0012602 | 0.9942801 | 0.0000329 | 0.0000142 | 0.0000037 |
| 315 | Quartair | Quartair | 0.0020974 | 0.0000422 | 0.0000069 | 0.0000319 | 0.0000217 | 0.0000494 | 0.0000735 | 0.0004136 | 0.0000610 | 0.0000374 | 0.0030538 | 0.0053875 | 0.0015727 | 0.9871011 | 0.0000359 | 0.0000116 | 0.0000025 |
| 315 | Quartair | Quartair | 0.0132683 | 0.0000220 | 0.0001161 | 0.0000126 | 0.0000076 | 0.0000238 | 0.0006703 | 0.0000460 | 0.0000373 | 0.0000263 | 0.0101721 | 0.0009708 | 0.0014296 | 0.9731284 | 0.0000273 | 0.0000403 | 0.0000011 |
| 315 | Quartair | Quartair | 0.0039433 | 0.0000242 | 0.0000302 | 0.0000086 | 0.0000057 | 0.0000237 | 0.0000332 | 0.0000262 | 0.0000599 | 0.0000241 | 0.0022655 | 0.0005780 | 0.0013032 | 0.9916468 | 0.0000092 | 0.0000176 | 0.0000007 |
| 315 | Quartair | Quartair | 0.0006227 | 0.0000283 | 0.0000121 | 0.0000102 | 0.0000184 | 0.0000175 | 0.0000081 | 0.0000340 | 0.0000423 | 0.0000265 | 0.0019118 | 0.0007904 | 0.0010599 | 0.9953914 | 0.0000179 | 0.0000073 | 0.0000013 |
| 315 | Quartair | Quartair | 0.0005324 | 0.0001363 | 0.0000449 | 0.0000533 | 0.0001559 | 0.0001011 | 0.0001432 | 0.0027850 | 0.0000843 | 0.0000600 | 0.0127065 | 0.0082927 | 0.0019381 | 0.9728345 | 0.0000658 | 0.0000528 | 0.0000131 |
| 315 | Quartair | Quartair | 0.0005340 | 0.0000177 | 0.0000219 | 0.0000068 | 0.0000078 | 0.0000125 | 0.0000261 | 0.0000275 | 0.0000393 | 0.0000171 | 0.0021269 | 0.0010173 | 0.0008628 | 0.9952722 | 0.0000041 | 0.0000054 | 0.0000006 |
| 315 | Quartair | Quartair | 0.0575981 | 0.0000081 | 0.0000241 | 0.0000082 | 0.0000027 | 0.0000125 | 0.0001122 | 0.0000174 | 0.0000270 | 0.0000168 | 0.0056392 | 0.0002271 | 0.0004502 | 0.9357872 | 0.0000072 | 0.0000610 | 0.0000009 |
| 315 | Quartair | Quartair | 0.0152549 | 0.0000453 | 0.0000739 | 0.0000185 | 0.0000117 | 0.0000362 | 0.0002852 | 0.0000825 | 0.0000699 | 0.0000370 | 0.0137846 | 0.0007646 | 0.0017072 | 0.9677808 | 0.0000190 | 0.0000265 | 0.0000024 |
| 315 | Quartair | Quartair | 0.0008461 | 0.0000704 | 0.0000105 | 0.0000168 | 0.0001117 | 0.0000546 | 0.0000263 | 0.0032778 | 0.0001608 | 0.0000473 | 0.0104254 | 0.0097869 | 0.0007468 | 0.9743658 | 0.0000345 | 0.0000107 | 0.0000076 |
| 315 | Quartair | Quartair | 0.0028084 | 0.0003202 | 0.0000452 | 0.0007453 | 0.0035366 | 0.0004534 | 0.0002749 | 0.0419457 | 0.0021525 | 0.0004648 | 0.2314618 | 0.0312856 | 0.0063444 | 0.6775461 | 0.0004350 | 0.0000679 | 0.0001122 |
| 315 | Quartair | Mons_en_Pevele | 0.0024323 | 0.0002857 | 0.0000241 | 0.0001670 | 0.0023436 | 0.0007036 | 0.0002050 | 0.0072303 | 0.0008965 | 0.0002191 | 0.5287787 | 0.0056719 | 0.0014640 | 0.4491573 | 0.0003417 | 0.0000357 | 0.0000434 |
| 315 | Quartair | Quartair | 0.0001545 | 0.0001841 | 0.0000103 | 0.0004289 | 0.0011504 | 0.0001690 | 0.0000846 | 0.0380116 | 0.0005418 | 0.0000765 | 0.2394062 | 0.0042444 | 0.0022770 | 0.7131411 | 0.0000517 | 0.0000136 | 0.0000541 |
| 315 | Quartair | Quartair | 0.0003524 | 0.0001417 | 0.0000132 | 0.0001863 | 0.0003735 | 0.0001353 | 0.0002053 | 0.0064641 | 0.0017583 | 0.0001285 | 0.0892736 | 0.0037258 | 0.0011172 | 0.8959759 | 0.0000587 | 0.0000177 | 0.0000722 |
| 315 | Quartair | Quartair | 0.0007949 | 0.0003013 | 0.0000365 | 0.0003914 | 0.0012891 | 0.0003094 | 0.0005196 | 0.0142819 | 0.0011059 | 0.0002796 | 0.2219382 | 0.0168694 | 0.0028573 | 0.7384913 | 0.0002882 | 0.0000364 | 0.0002096 |
| 315 | Quartair | Mons_en_Pevele | 0.0091130 | 0.0005375 | 0.0000906 | 0.0002646 | 0.0017623 | 0.0003615 | 0.0004690 | 0.0098435 | 0.0026698 | 0.0002901 | 0.6357463 | 0.0047057 | 0.0010144 | 0.3323590 | 0.0004824 | 0.0000460 | 0.0002443 |
| 315 | Quartair | Mons_en_Pevele | 0.0104686 | 0.0003540 | 0.0000436 | 0.0001137 | 0.0018878 | 0.0002214 | 0.0004818 | 0.0031002 | 0.0019579 | 0.0001342 | 0.5663326 | 0.0022540 | 0.0008581 | 0.4114163 | 0.0001555 | 0.0000243 | 0.0001959 |
| 315 | Quartair | Mons_en_Pevele | 0.0064360 | 0.0003280 | 0.0000875 | 0.0001628 | 0.0008018 | 0.0005363 | 0.0007330 | 0.0177942 | 0.0012119 | 0.0004420 | 0.5430352 | 0.2265276 | 0.0020288 | 0.1972645 | 0.0024206 | 0.0001062 | 0.0000837 |
| 315 | Quartair | Quartair | 0.0241659 | 0.0001452 | 0.0000320 | 0.0001558 | 0.0006000 | 0.0001368 | 0.0001656 | 0.0037429 | 0.0016563 | 0.0001597 | 0.3239998 | 0.0023742 | 0.0017669 | 0.6403751 | 0.0004086 | 0.0000211 | 0.0000939 |
| 315 | Quartair | Mons_en_Pevele | 0.0524160 | 0.0001808 | 0.0000468 | 0.0000668 | 0.0001737 | 0.0004635 | 0.0001931 | 0.0042013 | 0.0007901 | 0.0004074 | 0.5704201 | 0.3257635 | 0.0006818 | 0.0426493 | 0.0013631 | 0.0000491 | 0.0001336 |
| 315 | Mont_Panisel | Mont_Panisel | 0.0047391 | 0.0000320 | 0.0000506 | 0.0000121 | 0.0001246 | 0.0001223 | 0.0001667 | 0.0000984 | 0.0000619 | 0.0003219 | 0.2827335 | 0.7099467 | 0.0001450 | 0.0012081 | 0.0000894 | 0.0000525 | 0.0000951 |
| 315 | Mont_Panisel | Mons_en_Pevele | 0.0075760 | 0.0000101 | 0.0001743 | 0.0000168 | 0.0000428 | 0.0000779 | 0.0000225 | 0.0000235 | 0.0000335 | 0.0001524 | 0.6065808 | 0.3846824 | 0.0000366 | 0.0003880 | 0.0000519 | 0.0000096 | 0.0001208 |
| 315 | Mont_Panisel | Mont_Panisel | 0.0147862 | 0.0000378 | 0.0000493 | 0.0000286 | 0.0000565 | 0.0003072 | 0.0001181 | 0.0003255 | 0.0001139 | 0.0002345 | 0.2095118 | 0.7698631 | 0.0001839 | 0.0041301 | 0.0001127 | 0.0000692 | 0.0000717 |
| 315 | Mont_Panisel | Mons_en_Pevele | 0.0095998 | 0.0000069 | 0.0000935 | 0.0000053 | 0.0000446 | 0.0000659 | 0.0000147 | 0.0002788 | 0.0000237 | 0.0000239 | 0.9572742 | 0.0319158 | 0.0000289 | 0.0004722 | 0.0000630 | 0.0000018 | 0.0000871 |
| 315 | Mont_Panisel | Mont_Panisel | 0.0016351 | 0.0000302 | 0.0001544 | 0.0000052 | 0.0001013 | 0.0000876 | 0.0000431 | 0.0002103 | 0.0000372 | 0.0002247 | 0.3779389 | 0.6182738 | 0.0000332 | 0.0004724 | 0.0001226 | 0.0000126 | 0.0006174 |
| 315 | Mont_Panisel | Mons_en_Pevele | 0.0090006 | 0.0000054 | 0.0002139 | 0.0000088 | 0.0000449 | 0.0000440 | 0.0000120 | 0.0000123 | 0.0000312 | 0.0002408 | 0.7857845 | 0.2040843 | 0.0000096 | 0.0001536 | 0.0000353 | 0.0000090 | 0.0003098 |
| 315 | Mont_Panisel | Mons_en_Pevele | 0.0010493 | 0.0000089 | 0.0000513 | 0.0000030 | 0.0000233 | 0.0000337 | 0.0000067 | 0.0000387 | 0.0000214 | 0.0001066 | 0.7605058 | 0.2377155 | 0.0000092 | 0.0001377 | 0.0000109 | 0.0000051 | 0.0002730 |
| 315 | Mont_Panisel | Mont_Panisel | 0.0057808 | 0.0000098 | 0.0000442 | 0.0000118 | 0.0000653 | 0.0000901 | 0.0000193 | 0.0000422 | 0.0000232 | 0.0002233 | 0.2535829 | 0.7392489 | 0.0000467 | 0.0004114 | 0.0000392 | 0.0000391 | 0.0003218 |
| 315 | Mont_Panisel | Mons_en_Pevele | 0.0266223 | 0.0000309 | 0.0005370 | 0.0000136 | 0.0000559 | 0.0000791 | 0.0004993 | 0.0000902 | 0.0000634 | 0.0020039 | 0.5599939 | 0.4091515 | 0.0000786 | 0.0005465 | 0.0001177 | 0.0000950 | 0.0000211 |
| 315 | Aalbeke | Mons_en_Pevele | 0.2736155 | 0.0000876 | 0.0002071 | 0.0000675 | 0.0001054 | 0.0001809 | 0.0132506 | 0.0002967 | 0.0001225 | 0.0009430 | 0.6169596 | 0.0792468 | 0.0004657 | 0.0138881 | 0.0005060 | 0.0000452 | 0.0000119 |
| 315 | Aalbeke | Mons_en_Pevele | 0.1312164 | 0.0000501 | 0.0004623 | 0.0000282 | 0.0000868 | 0.0001289 | 0.0410878 | 0.0002031 | 0.0000884 | 0.0019159 | 0.6012088 | 0.2216740 | 0.0001599 | 0.0013756 | 0.0002658 | 0.0000345 | 0.0000134 |
| 315 | Aalbeke | Mons_en_Pevele | 0.0804223 | 0.0000549 | 0.0001072 | 0.0000324 | 0.0000663 | 0.0001132 | 0.0029325 | 0.0001371 | 0.0000551 | 0.0005291 | 0.7161882 | 0.1947744 | 0.0003377 | 0.0038543 | 0.0003629 | 0.0000258 | 0.0000064 |
| 315 | Aalbeke | Mons_en_Pevele | 0.3463559 | 0.0000314 | 0.0003179 | 0.0000189 | 0.0000458 | 0.0000846 | 0.0013240 | 0.0001268 | 0.0000677 | 0.0010382 | 0.4940860 | 0.1554657 | 0.0001022 | 0.0006906 | 0.0002103 | 0.0000270 | 0.0000071 |
| 315 | Aalbeke | Mons_en_Pevele | 0.4054372 | 0.0000108 | 0.0002117 | 0.0000106 | 0.0000139 | 0.0000312 | 0.0004830 | 0.0000337 | 0.0000202 | 0.0000612 | 0.5389125 | 0.0541093 | 0.0000664 | 0.0004251 | 0.0001567 | 0.0000151 | 0.0000015 |
| 315 | Aalbeke | Mons_en_Pevele | 0.3320181 | 0.0000112 | 0.0000922 | 0.0000074 | 0.0000112 | 0.0000260 | 0.0010944 | 0.0000330 | 0.0000126 | 0.0000423 | 0.6333257 | 0.0326821 | 0.0000853 | 0.0004572 | 0.0000917 | 0.0000081 | 0.0000015 |
| 315 | Aalbeke | Aalbeke | 0.6354856 | 0.0000083 | 0.0000681 | 0.0000050 | 0.0000075 | 0.0000204 | 0.0005761 | 0.0000150 | 0.0000129 | 0.0000575 | 0.3392069 | 0.0240957 | 0.0000550 | 0.0003124 | 0.0000649 | 0.0000065 | 0.0000021 |
| 315 | Aalbeke | Aalbeke | 0.6310676 | 0.0000184 | 0.0001231 | 0.0000167 | 0.0000124 | 0.0000388 | 0.0004557 | 0.0000541 | 0.0000480 | 0.0002409 | 0.3402088 | 0.0263584 | 0.0000951 | 0.0011220 | 0.0001119 | 0.0000238 | 0.0000043 |
| 316 | Quartair | Quartair | 0.0111996 | 0.0000135 | 0.0000053 | 0.0000485 | 0.0000024 | 0.0000229 | 0.0000637 | 0.0000094 | 0.0000654 | 0.0005497 | 0.0000316 | 0.0002946 | 0.0021294 | 0.9852787 | 0.0000346 | 0.0002431 | 0.0000076 |
| 316 | Quartair | Quartair | 0.0002983 | 0.0000055 | 0.0000016 | 0.0000108 | 0.0000005 | 0.0000097 | 0.0000065 | 0.0000071 | 0.0000221 | 0.0000228 | 0.0000079 | 0.0001991 | 0.0015370 | 0.9978452 | 0.0000076 | 0.0000142 | 0.0000042 |
| 316 | Quartair | Quartair | 0.0001055 | 0.0000073 | 0.0000006 | 0.0000117 | 0.0000013 | 0.0000131 | 0.0000073 | 0.0000260 | 0.0000206 | 0.0000082 | 0.0000121 | 0.0001518 | 0.0025626 | 0.9970224 | 0.0000046 | 0.0000436 | 0.0000013 |
| 316 | Quartair | Quartair | 0.0011579 | 0.0000683 | 0.0000351 | 0.0001100 | 0.0000056 | 0.0000675 | 0.0005566 | 0.0000941 | 0.0000848 | 0.0001012 | 0.0001324 | 0.0032814 | 0.0070877 | 0.9870572 | 0.0001272 | 0.0000215 | 0.0000115 |
| 316 | Quartair | Quartair | 0.0014130 | 0.0000685 | 0.0000102 | 0.0000804 | 0.0000084 | 0.0000971 | 0.0001032 | 0.0018180 | 0.0000981 | 0.0000801 | 0.0000754 | 0.0038869 | 0.0028636 | 0.9892363 | 0.0000819 | 0.0000311 | 0.0000478 |
| 316 | Quartair | Quartair | 0.0000561 | 0.0000153 | 0.0000014 | 0.0000074 | 0.0000050 | 0.0000244 | 0.0000153 | 0.0001373 | 0.0000202 | 0.0000105 | 0.0000254 | 0.0007076 | 0.0015821 | 0.9973715 | 0.0000143 | 0.0000051 | 0.0000009 |
| 316 | Quartair | Quartair | 0.0002236 | 0.0000127 | 0.0000010 | 0.0000108 | 0.0000023 | 0.0000224 | 0.0000191 | 0.0002150 | 0.0000331 | 0.0000194 | 0.0000566 | 0.0011703 | 0.0011319 | 0.9970576 | 0.0000107 | 0.0000080 | 0.0000055 |
| 316 | Quartair | Quartair | 0.0345086 | 0.0000139 | 0.0000223 | 0.0000133 | 0.0000029 | 0.0000175 | 0.0004638 | 0.0000103 | 0.0000573 | 0.0012153 | 0.0001845 | 0.0004419 | 0.0005647 | 0.9623100 | 0.0000162 | 0.0001553 | 0.0000022 |
| 316 | Quartair | Quartair | 0.0070702 | 0.0000130 | 0.0000314 | 0.0000068 | 0.0000034 | 0.0000139 | 0.0022903 | 0.0000237 | 0.0000171 | 0.0000077 | 0.0002801 | 0.0009240 | 0.0009342 | 0.9883544 | 0.0000110 | 0.0000180 | 0.0000010 |
| 316 | Quartair | Quartair | 0.0293236 | 0.0000075 | 0.0000269 | 0.0000052 | 0.0000019 | 0.0000090 | 0.0003159 | 0.0000176 | 0.0000148 | 0.0000106 | 0.0001935 | 0.0002881 | 0.0005229 | 0.9692330 | 0.0000068 | 0.0000221 | 0.0000005 |
| 316 | Quartair | Quartair | 0.0165233 | 0.0000352 | 0.0000235 | 0.0000093 | 0.0000105 | 0.0000196 | 0.0001293 | 0.0001139 | 0.0001232 | 0.0003014 | 0.0002193 | 0.0003094 | 0.0007179 | 0.9812296 | 0.0000208 | 0.0002052 | 0.0000084 |
| 316 | Quartair | Quartair | 0.0037115 | 0.0000522 | 0.0000241 | 0.0000462 | 0.0000132 | 0.0000615 | 0.0004644 | 0.0001416 | 0.0001022 | 0.0000659 | 0.0006304 | 0.0080505 | 0.0024383 | 0.9840953 | 0.0000716 | 0.0000225 | 0.0000088 |
| 316 | Quartair | Quartair | 0.0500095 | 0.0005228 | 0.0000789 | 0.0002064 | 0.0002201 | 0.0003905 | 0.0002284 | 0.0037998 | 0.0024351 | 0.0005016 | 0.0783146 | 0.0051364 | 0.0060337 | 0.8508718 | 0.0010659 | 0.0000608 | 0.0001239 |
| 316 | Quartair | Quartair | 0.0034045 | 0.0003383 | 0.0000367 | 0.0003508 | 0.0005242 | 0.0003286 | 0.0002692 | 0.0106392 | 0.0012067 | 0.0003128 | 0.0114619 | 0.0052046 | 0.0076786 | 0.9575136 | 0.0005872 | 0.0000492 | 0.0000940 |
| 316 | Quartair | Quartair | 0.0010029 | 0.0001396 | 0.0000181 | 0.0003235 | 0.0002850 | 0.0001436 | 0.0001632 | 0.0177765 | 0.0002887 | 0.0001061 | 0.0030714 | 0.0142229 | 0.0038200 | 0.9584195 | 0.0001829 | 0.0000188 | 0.0000172 |
| 316 | Quartair | Quartair | 0.0010958 | 0.0002668 | 0.0000188 | 0.0002141 | 0.0001447 | 0.0001074 | 0.0000933 | 0.0153103 | 0.0005881 | 0.0001263 | 0.0046678 | 0.0027650 | 0.0017492 | 0.9726022 | 0.0002040 | 0.0000172 | 0.0000289 |
| 316 | Quartair | Quartair | 0.0046080 | 0.0003765 | 0.0000349 | 0.0004689 | 0.0002990 | 0.0002102 | 0.0003212 | 0.0064621 | 0.0015983 | 0.0002096 | 0.0135717 | 0.0024582 | 0.0026177 | 0.9662536 | 0.0004313 | 0.0000295 | 0.0000494 |
| 316 | Quartair | Quartair | 0.0014685 | 0.0002943 | 0.0000267 | 0.0004893 | 0.0003072 | 0.0001641 | 0.0002077 | 0.0078781 | 0.0010799 | 0.0001843 | 0.0052349 | 0.0032146 | 0.0014258 | 0.9775595 | 0.0004005 | 0.0000234 | 0.0000412 |
