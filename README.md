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

say something about this notebook

#### [Geaospatial interpolation Model](./modeling/geospatial_model.py)

say something about this script

#### [Fit Models](./modeling/fit_models.py)

say something about this script

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

#### [CRM Model](./modeling/model_CRM_Dorothy.ipynb)

say something about this notebook

### Exploratory Analysis

#### [Folder with EDA notebooks and comparison of model results](./exploratory)

Initial data checks can be found here some of it is also within the
modeling notebooks.

### Dashboard

#### Streamlit dashboard code

say something about the dashboard

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
