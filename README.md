CPT USECASE GitHub Document
================

## CPT usecase

modeling/ contains notebooks and scripts for model building

dashbord/ contains streamlit dashboard scripts

You need to add your aboslute paths to the paths_cpt_file for it to run.
You need to define PATH_TO_PARQUET = “” and PATH_TO_MODEL =““. The idea is that we can change this logic later to use environment variables instead to run this in different environments. For the dashboard to work, you need to also create the pickle of the model by running the EDA.ipynb (or whatever other method).

This README explains:

- how the data is processed (binning pipeline),
- how to run the preprocessing scripts,
- how to train / evaluate models,
- how to run the dashboard.

## Repository structure

├─ dashboard/            # Streamlit dashboard files
├─ Documentation/        # Reports, slides, background docs
├─ exploratory/          # EDA notebooks and quick experiments
├─ modeling/             # Reusable scripts + modelling notebooks
├─ results/              # Processed data, splits, model outputs
├─ README.md             # This file
├─ README.Rmd            # RMarkdown source for README.html
├─ README.html           # Rendered README (for sharing)
├─ requirements.txt      # Python dependencies
└─ run_dashboard.bat     # Helper script to start the dashboard on Windows

## Setup
Create a virtual environment and install dependencies:
pip install -r requirements.txt
The path configuration for CPT data is in path_cpt.py (This can be replaced by environment variables)

## Data Processing

### The binning method

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

``` bash
python modeling/data_processing.py --extract_trend True --bin_w 0.6 --seed 42 --trend_type additive --data_folder data --results_folder results
```

- To run with default parameters, simply run:

``` bash
python modeling/data_processing.py
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
```

    ## Train IDs: [4875, 4476, 4877, 4551, 1135, 4574, 4617, 4869, 3648, 4674, 316, 13184, 4678, 13081, 4833, 4560, 3749, 4643, 4417, 4868, 3745, 2924, 3658, 4747, 4445, 3386, 3724, 3804, 4659, 3663, 4582, 4767, 4807, 3892, 3646, 14010, 13295, 496, 3399, 13298, 13288, 1931, 4660, 4449, 4594, 4437, 4918, 4665, 552, 13140, 4866, 495, 1580, 377, 3678, 4425, 4383, 13289, 4524, 13167, 3667, 3666, 4811, 4629, 13203, 3674, 375, 1928, 4415, 3662, 14008, 4775, 1933, 14001, 4440, 4912, 3712, 4876, 13285, 319, 13180, 3676, 4614, 3806, 3670, 14007, 4768, 3387, 4640, 4873, 4579, 3644, 4624, 4865, 4810, 13066, 3681, 3891, 554, 3661, 4578, 4759, 3748, 3400, 13300, 555, 13125, 13267, 4429, 3682, 2699, 4835, 2752, 14013, 14006, 3643, 4510, 4407, 4472, 4917, 494, 13113, 3655, 14004, 4646, 4677, 4806, 3389, 13211, 13057, 4809, 4815, 4919, 4647, 3675, 3651, 4481, 13257, 1578, 4442, 376, 2923, 3398, 550, 4740, 4913, 13063, 13117, 13067, 551, 13297, 4648, 13145, 14000, 4517, 4886, 4408, 553, 1930, 4473, 4564, 4878, 1775, 13010, 14003, 4547, 4605, 1579, 4808, 13131, 4794, 4409, 3660, 497, 1848, 3808, 4776, 1929, 4482, 4441, 4915, 3746, 4515, 4843, 13095, 3656, 13001, 4464, 4382, 13094, 3893, 4910, 4557, 4455, 4874, 3647, 3652, 1932, 4761]

``` python
print("Test   IDs:", test_train["test_ids"])
```

    ## Test   IDs: [3807, 3665, 4569, 4521, 315, 1581, 4887, 3809, 4828, 4424, 4814, 4785, 4411, 4772, 4460, 13110, 13138, 3671, 1156, 1834, 3805, 4410, 14011, 3385, 374, 3397, 4537, 4518, 13143, 4737, 3388, 13068, 13062, 13286, 14009, 13159, 1158, 4916, 14005, 3653, 13091, 4821, 13206, 3679, 4514, 13006, 4540, 3664, 13105, 4448, 13166, 2705, 3672, 3654, 3669, 4619, 3894, 3683, 3720, 4889, 3747, 13004, 4736, 4858, 1823, 3722, 3739, 13160, 3677, 3680, 4438, 4860, 2702, 2916, 4820, 493, 4745, 13142, 4658, 4902, 314, 3723, 3657, 4867, 1927, 4596]
