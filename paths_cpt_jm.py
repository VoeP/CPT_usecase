from pathlib import Path

# Project root = folder where this file lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder structure according to our repo
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "modeling"

# Full CPT parquet file
PATH_TO_PARQUET = DATA_DIR / "vw_cpt_brussels_params_completeset_20250318_remapped.parquet"

# for now, rf pkl
PATH_TO_MODEL = MODEL_DIR / "rf_model_export.pkl"

# Baseline geospatial interpolation model (see fit_models_jm.py)
PATH_TO_GEOSPATIAL = MODEL_DIR / "geospatial_interpolation.pkl"
