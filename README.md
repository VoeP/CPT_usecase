CPT USECASE GitHub Document
================

## CPT Usecase - Lithostratigraphy Modelling

modeling/ contains notebooks and scripts for model building

dashbord/ contains streamlit dashboard scripts

You need to add your aboslute paths to the paths_cpt_file for it to run.
You need to define PATH_TO_PARQUET = “” and PATH_TO_MODEL =““. The idea
there is that we can change this logic later to use environment
variables instead to run this in different environments. For the
dashboard to work, you need to also create the pickle of the model by
running the EDA.ipynb (or whatever other method).

This README explains:

- how the data is processed (binning pipeline),
- how to run the preprocessing scripts,
- how to train / evaluate models,
- how to run the dashboard.

## Repository structure

- [**dashboard/**](./dashboard) - Streamlit dashboard files
- [**Documentation/**](./Documentation) - Reports, slides, background
  docs
- [**exploratory/**](./exploratory) - EDA notebooks and quick
  experiments
- [**modeling/**](./modeling) - Reusable scripts + modelling notebooks
- [**results/**](./results) - Processed data, splits, model outputs
- [**README.md**](./README.md) - This file
- [**README.Rmd**](./README.Rmd) - RMarkdown source for README.md
- [**requirements.txt**](./requirements.txt) - Python dependencies
- [**run_dashboard.bat**](./run_dashboard.bat) - Helper script to start
  the dashboard on Windows

## Setup

Create a virtual environment and install dependencies: pip install -r
requirements.txt The path configuration for CPT data is in path_cpt.py
(This can can be replaced by environment variables)

### Data Processing

#### [Binning pipeline code](./modeling/data_processing.py)

The goal is to turn noisy CPT measurements along depth into clean,
comparable slices (bins) with summary features.

- What we bin

  - For each borehole (sondering_id), measurements are taken at depths
    (diepte) for signals like qc, fs, rf, qtn, fr.

- How bins are built

  - We create fixed-depth intervals from 0 to the maximum depth using a
    bin width bin_w (default 0.6 m).
  - Each row is assigned to one depth_bin (include_lowest=True,
    ordered).

- What we compute per bin

  - For each borehole × bin, we summarize available signals summary
    stats:

    - mean, standard deviation, interquartile range, median, MAD
    - percentiles (10th, 50th, 90th)
    - coefficient of variation (std/mean, guarded when mean≈0)

  - We also compute QC “spike” features for qc:

    - fraction and count above 20 and 40 (TH_QC20, TH_QC40)
    - 99th percentile (p99)

- Lithostrat label per bin

  - Each bin gets a single lithostrat_id based on the records that fall
    inside that interval (one label per borehole × bin).

- Optional trend cleaning before binning

  - If enabled, each signal is de-trended per borehole using seasonal
    decomposition (additive or multiplicative), with a fallback to a
    rolling mean. Missing and infinite values are forward/backward
    filled.

- Why binning

  - Reduces noise,

  Run the preprocessing from the repo root with optional flags to
  control behavior:

Inport the script as a module and in your modelling script run it as
shown below

``` python
train_processed = dp.process_test_train(
    cpt_df=cpt_data,
    sondering_ids=train_ids,
    bin_w=BIN_W,
    do_extract_trend=EXTRACT_TREND,
    trend_type=TREND_TYPE
)

print("Processing test data...")
test_processed = dp.process_test_train(
    cpt_df=cpt_data,
    sondering_ids=test_ids,
    bin_w=BIN_W,
    do_extract_trend=EXTRACT_TREND,
    trend_type=TREND_TYPE
)
```

Arguments:

- extract_trend toggles per-borehole detrending;
- bin_w sets bin size in meters;
- seed controls the reproducible train/test split;
- trend_type selects the decomposition model (“additive” or
  “multiplicative”);
- data_folder points to the input folder containing the parquet (expects
  vw_cpt_brussels_params_completeset_20250318_remapped.parquet);
- results_folder sets the output folder for processed CSVs.

#### Train/test split reproduction

If you prefer to work on raw (unbinned) CPT rows, you can still
reproduce the exact train/test split by loading results/split_res.json
(pickled dict with keys train_ids and test_ids) and filtering your
dataset by sondering_id accordingly. This guarantees consistency with
the split used to generate the binned outputs.

``` python
import json
with open("results/split_res.json", "r") as f:
    test_train = json.load(f)
#print("IDS", test_train)
print("Train IDs:", test_train["train_ids"])
print("Test   IDs:", test_train["test_ids"])
```

#### [Data Modules](./modeling/data_modules.py)

### [Models Folder](./modeling)

This folder contains the model training and evaluation notebooks and
scripts.

#### [KNN Model & HMM Experimentation](./modeling/modeling_voep.ipynb)

modeling_voep.ipynb implements a K‑Nearest Neighbors (KNN) model for local, feature‑based prediction
s and a Hidden Markov Model (HMM) to capture sequential structure and enforce geologically plausible
state transitions. The KNN provides pointwise estimates by averaging nearby samples in feature space
(tunable k and distance metric) to capture local variability and produce probabilistic or continuous outputs;
the HMM models depth-wise or profile-wise state dynamics via learned transition and emission parameters,
using Viterbi/posterior decoding to segment profiles, smooth noisy predictions, and quantify sequence uncertainty.
Combined, the KNN gives accurate local predictions while the HMM adds context-aware smoothing and state-segmentation,
improving interpretability and robustness for subsurface profiling

#### [Geaospatial interpolation Model](./modeling/geospatial_model.py)

This model builds continuous spatial prediction surfaces by combining spatial feature engineering with spatially-aware modeling.
It converts spatial locations and auxiliary layers into predictive features, fits models that capture spatial structure (for example local regressors, kriging/Gaussian-process style interpolators, or ensemble learners that incorporate spatial covariates),
interpolates model outputs to a regular grid, and estimates per-location uncertainty (prediction intervals or kriging variance).
The module also applies spatial validation and smoothing/post-processing to produce geologically plausible, ready-to-map rasters and
saved model artifacts for downstream visualization and analysis.



#### [Fit Models](./modeling/fit_models.py)

This model coordinates training, selection, and evaluation across the project’s feature pipelines.
It constructs end-to-end training workflows: builds preprocessing/feature pipelines, fits configurable estimators (e.g., tree ensembles, linear models, or other scikit-learn-compatible learners),
runs hyperparameter search and cross-validation, and evaluates models with standard metrics and calibration checks.
The script supports model comparison and ensembling, persists trained model artifacts and metadata, and produces evaluation summaries and diagnostics (learning curves, feature importances, confusion/misclassification analyses).
Its role is orchestration—turning prepared features into validated, production-ready models that downstream modules can load for prediction and analysis.

#### [Binning Models](./modeling/binn_method_modelling.ipynb)

To handle the high-resolution nature of Cone Penetration Test (CPT)
data, this workflow implements a feature engineering strategy based on
depth binning. Raw sensor measurements are aggregated into fixed-width
intervals (0.6m), extracting statistical summaries and trend features to
reduce noise and dimensionality.

We trained and tuned three tree-based ensemble classifiers—XGBoost,
Random Forest, and LightGBM—using RandomizedSearchCV to optimize

performance. Evaluation is conducted in two stages: first on the
aggregated test bins, and subsequently by propagating predictions back
to the raw, unbinned dataset to assess the model’s granular accuracy
across specific lithostratigraphic units.

#### [CRM Model](./modeling/model_CRM.py)

Sequence-aware Conditional Random Field (CRF) model predicts lithostratigraphic units from CPT depth profiles.
Each CPT has feature vectors ordered by depth and are treated as a sequence, with the corresponding lithostratigraphic units
as the sequence of target labels. This yields one feature sequence and one label sequence per borehole, which the CRF then models
jointly along depth.

These sequences are then used to train a linear-chain CRF that models both the relationship between features and labels at each depth
and the dependencies between neighbouring lithostrat labels along depth.
The model is implemented with the L-BFGS optimiser and all_possible_transitions=True, and its regularisation hyperparameters are tuned with RandomizedSearchCV
using a sequence-aware F1 scorer and cross-validation on the training CPTs.
Entire CPTs are kept either in the training or in the test set to avoid leakage along depth.
The final CRF reaches about 58% test accuracy (weighted F1 ≈ 0.57) on held-out CPTs, and its trained weights are saved as a reusable model that can be loaded for further analysis and visualisation in the dashboard.

### Exploratory Analysis

#### [Folder with EDA notebooks and comparison of model results](./exploratory)

Initial data checks can be found here some of it is also within the
modeling notebooks.

### Dashboard

#### Streamlit dashboard code

dashboard_internals.py — Provides the app's reusable building blocks: cached data loaders, data-filtering and feature‑extraction utilities, plotting and mapping helpers (for Plotly/folium/GeoPandas), and model‑inference wrappers that load trained models and produce profile or spatial predictions. It encapsulates business logic and presentation helpers so the Streamlit UI stays thin and responsive

dashboard_scripts.py — The Streamlit application entrypoint: defines layout and UI controls (sidebars, selectors, model/parameter choices), reacts to user inputs, invokes dashboard_internals to run inference and generate visualizations, and renders interactive maps, profiles, and metrics. It manages session state and caching to keep interactions fast and handles user-driven exports/downloads.

dashboard_preparation.py — Prepares and caches data and artifacts used by the dashboard: runs preprocessing pipelines, computes derived spatial layers or aggregated summaries, and optionally precomputes model output tiles or profile caches to accelerate the interactive app. Intended to be run ahead of launching the dashboard so the UI serves preprocessed, ready-to-render content.
- [Dashboard internals](./dashboard/dashboard_internals.py)
- [Dashboard main script](./dashboard/dashboard_scripts.py)
- [Dashboard preparation script](./dashboard/dashboard_preparation.py)

##### How to run streamlit dashboard

Give more details on how to Run the streamlit dashboard

- [Running the Dashboard script windows](./run_dashboard.bat)
- [Running the Dashboard script linux](./run_dashboard.sh)

#### Other Tried Frameworks (Plotly Dash)

- [Dash App](./dashboard/app.py)

##### How to run

- From terminal run: python dashboard/app.py
- You will see the link to open in the terminal once the server is
  running.
- Open the link in your browser to access the dashboard.
- Data upload use csv file with same structure as data provided by VITO
- Example data set [test data](data/test_raw_data.csv)
- required packages
- [Model used is the RandomForest model pickle provided in the
  repo](./results/models/best_rf_model.pkl)
- [Labeler used is the label encoder pickle provided in the
  repo](./results/models/label_encoder.pkl)

``` text
numpy
pandas
dash
plotly
joblib
matplotlib
fastparquet
scikit-learn
```

- copy packages in a text file text_file_name.txt and run pip install -r
  text_file_name.txt to install all packages at once.
