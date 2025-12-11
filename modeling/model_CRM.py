# ======================================================
# CONDITIONAL RANDOM FIELD (CRF) MODEL FOR CPT DATA
# ======================================================
# Purpose: Predict lithostratigraphic units from CPT data using CRF
# ======================================================

from pathlib import Path
from data_module import segments_oi, invalid_labels, DataSet
folder = Path(r"C:\Users\dorothy.chepkoech\Documents\MSC_2026\Project_DataScience\Data")

for p in folder.glob("*.parquet"):
    print(p.name)


import pandas as pd
import numpy as np
from pathlib import Path

home = Path.home()
print("Home folder:", home)

# Correct path with correct spelling and folder name
path_to_parquet = home / "Documents" / "MSC_2026" / "Project_DataScience" / "Data" / "remapped.parquet"

print("Exists?", path_to_parquet.exists())
print("Path:", path_to_parquet)

df = pd.read_parquet(path_to_parquet)

print(df.shape)
print(df.columns)
df.head()

# ==== Global config for segments & labels ====
segments_oi = [  # segments of interest
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
    "Mons_en_Pevele",
]

invalid_labels = {"", "none", "nan", "onbekend"}

default_num_vars = ["qc", "fs", "rf", "qtn", "fr", "icn", "sbt", "ksbt"]

segment_order = [
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
    "Mons_en_Pevele",
]

def impute_params(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    - Impute icn, sbt, ksbt based on qtn/fr/icn.
    - overwrite = False (default): only fill where values are NA.
    - overwrite = True: recompute for all rows.
    """
    df = df.copy()

    # icn
    mask_icn = df["icn"].isna() if not overwrite else np.ones(len(df), dtype=bool)
    valid_icn = mask_icn & df["qtn"].gt(0) & df["fr"].gt(0)

    icn_new = np.sqrt(
        (3.47 - np.log10(df.loc[valid_icn, "qtn"])) ** 2
        + (np.log10(df.loc[valid_icn, "fr"]) + 1.22) ** 2
    )

    df.loc[valid_icn, "icn"] = icn_new

    # sbt
    def sbt_from_icn(icn):
        if pd.isna(icn):
            return np.nan
        if icn < 1.31:
            return 1
        elif icn < 2.05:
            return 2
        elif icn < 2.60:
            return 3
        elif icn < 2.95:
            return 4
        elif icn < 3.60:
            return 5
        else:
            return 6

    mask_sbt = df["sbt"].isna() if not overwrite else np.ones(len(df), dtype=bool)
    df.loc[mask_sbt, "sbt"] = df.loc[mask_sbt, "icn"].apply(sbt_from_icn)

    # ksbt
    df["ksbt"] = pd.to_numeric(df["ksbt"], errors="coerce")

    def ksbt_from_icn(icn):
        if pd.isna(icn):
            return np.nan

        if 1.0 < icn <= 3.27:
            return 10 ** (0.952 - 3.04 * icn)
        elif icn > 3.27:
            return 10 ** (-4.52 - 1.37 * icn)

    mask_ksbt = df["ksbt"].isna() if not overwrite else np.ones(len(df), dtype=bool)
    df.loc[mask_ksbt, "ksbt"] = df.loc[mask_ksbt, "icn"].apply(ksbt_from_icn)

    return df

# =========================================
# Create labeled subset and impute
# =========================================

# 1) Keep only rows whose lithostrat_id is one of your segments of interest
df_known = df[df["lithostrat_id"].isin(segments_oi)].copy()
print("Known before imputation:", df_known.shape)

# 2) Impute icn, sbt, ksbt
df_known = impute_params(df_known, overwrite=False)
print("Known after imputation:", df_known.shape)

# 3) Remove invalid / empty labels
df_known = df_known[~df_known["lithostrat_id"].isin(invalid_labels)]
df_known = df_known.dropna(subset=["lithostrat_id"])
print("Known after label cleaning:", df_known.shape)
print(df_known["lithostrat_id"].value_counts())

cpt_col   = "sondeernummer"
depth_col = "diepte"

df_known = df_known.sort_values([cpt_col, depth_col])

print(df_known[[cpt_col, depth_col, "lithostrat_id"]].head())

import pandas as pd

def safe_float(x, default=0.0):
    return float(x) if pd.notna(x) else default

def row_to_features(row):
    """Features for one depth point."""
    return {
        # numeric CPT variables (use .get to avoid KeyError if some are missing)
        "qc":   safe_float(row.get("qc")),
        "fs":   safe_float(row.get("fs")),
        "rf":   safe_float(row.get("rf")),
        "qtn":  safe_float(row.get("qtn")),
        "fr":   safe_float(row.get("fr")),
        "icn":  safe_float(row.get("icn")),
        "ksbt": safe_float(row.get("ksbt")),
        # categorical soil behavior type
        "sbt":  str(int(row["sbt"])) if pd.notna(row.get("sbt")) else "nan",
        # depth as a feature
        "depth": safe_float(row.get(depth_col)),
    }

def row_to_label(row):
    return str(row["lithostrat_id"])

# Use binned-feature pipeline from data_processing
import data_processing as dp
from pathlib import Path as _Path

# Run preprocessing to get binned features
results_folder = _Path(__file__).parent.parent / "results"
res = dp.process_cpt_data(
    data_folder=path_to_parquet.parent,
    results_folder=results_folder,
    parquet_filename=path_to_parquet.name,
    do_extract_trend=True,
    bin_w=0.6,
    trend_type="additive",
)

features = res["features"]
train_ids = set(res["train_ids"])
test_ids = set(res["test_ids"])

print("Feature rows:", features.shape)

# Select columns for features (exclude id/labels/bin metadata)
exclude = {"sondering_id", "depth_bin", "lithostrat_id", "depth_bin_left", "depth_bin_right"}
feat_cols = [c for c in features.columns if c not in exclude]

def bin_row_to_features(row):
    d = {}
    for c in feat_cols:
        v = row.get(c)
        if pd.isna(v):
            d[c] = 0.0
        else:
            try:
                d[c] = float(v)
            except Exception:
                d[c] = str(v)
    return d

# Build training sequences from binned features
X_all = []
y_all = []
cpt_ids = []

for sid, g in features[features["sondering_id"].isin(train_ids)].groupby("sondering_id", observed=False):
    g = g.sort_values("depth_bin")
    X_seq = [bin_row_to_features(r) for _, r in g.iterrows()]
    y_seq = [str(r["lithostrat_id"]) for _, r in g.iterrows()]
    if len(X_seq) == 0:
        continue
    X_all.append(X_seq)
    y_all.append(y_seq)
    cpt_ids.append(sid)

print("Number of CPT sequences (binned, train):", len(X_all))

# Build test sequences
X_test = []
y_test = []
for sid, g in features[features["sondering_id"].isin(test_ids)].groupby("sondering_id", observed=False):
    g = g.sort_values("depth_bin")
    X_seq = [bin_row_to_features(r) for _, r in g.iterrows()]
    y_seq = [str(r["lithostrat_id"]) for _, r in g.iterrows()]
    if len(X_seq) == 0:
        continue
    X_test.append(X_seq)
    y_test.append(y_seq)

X_train = X_all
y_train = y_all

print("Train sequences:", len(X_train))
print("Test sequences:", len(X_test))

from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import CRF, metrics
from sklearn_crfsuite.metrics import flat_f1_score
import scipy.stats
from sklearn.metrics import make_scorer
import random

crf = CRF(
    algorithm="lbfgs",
    max_iterations=200,
    all_possible_transitions=True,
)

params_space = {
    "c1": scipy.stats.expon(scale=0.1),  # try around 0.1
    "c2": scipy.stats.expon(scale=0.1),
}

rs = RandomizedSearchCV(
    crf,
    params_space,
    cv=3,
    verbose=1,
    n_jobs=-1,
    n_iter=20,
    scoring=make_scorer(flat_f1_score, average="macro"),
    random_state=100,
)

# set seeds for reproducibility of sampling and any randomness
np.random.seed(100)
random.seed(100)
rs.fit(X_train, y_train)

print("Best params:", rs.best_params_)
print("Best CV score:", rs.best_score_)

best_crf = rs.best_estimator_
y_pred = best_crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred, digits=3))

# Save the trained model
import pickle
models_dir = _Path(__file__).parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "best_crf.pkl"
with open(model_path, "wb") as mf:
    pickle.dump(best_crf, mf)
print(f"Saved trained CRF to: {model_path}")

def predict_cpt_profile(cpt_id, df_source, model, cpt_col, depth_col):
    """
    cpt_id: value of sondeernummer for the CPT you want to plot
    df_source: dataframe with all engineered features (df_known or df_unlab)
    model: best_crf or loaded_crf
    """
    g = df_source[df_source[cpt_col] == cpt_id].copy()
    g = g.sort_values(depth_col)

    # build feature sequence
    X_seq = [row_to_features(r) for _, r in g.iterrows()]
    if len(X_seq) == 0:
        raise ValueError("No rows for this CPT")

    y_pred_seq = model.predict_single(X_seq)

    g["lithostrat_pred"] = y_pred_seq

    # if true labels exist
    if "lithostrat_id" in g.columns:
        g["lithostrat_true"] = g["lithostrat_id"]

    return g

# Example: choose first CPT in df_known
example_cpt = df_known[cpt_col].iloc[0]
print("Example CPT:", example_cpt)

profile_df = predict_cpt_profile(example_cpt, df_known, best_crf, cpt_col, depth_col)
profile_df[[cpt_col, depth_col, "lithostrat_true", "lithostrat_pred"]].head()

import matplotlib.pyplot as plt
import numpy as np

def plot_cpt_true_vs_pred_with_legend(profile_df, cpt_id, depth_col="diepte"):
    # sort by depth
    profile_df = profile_df.sort_values(depth_col)

    depths = profile_df[depth_col]
    qc = profile_df["qc"]

    # get all labels that appear (true + predicted)
    true_labels = profile_df["lithostrat_true"].dropna().astype(str)
    pred_labels = profile_df["lithostrat_pred"].dropna().astype(str)

    all_labels = sorted(set(true_labels.unique()) | set(pred_labels.unique()))

    # choose a discrete colormap and assign a color to each lithostrat
    cmap = plt.get_cmap("tab10")   # tab10 has 10 distinct colors
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(all_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # ---------- LEFT: TRUE ----------
    ax = axes[0]
    for lab in all_labels:
        mask = (profile_df["lithostrat_true"].astype(str) == lab)
        if mask.any():
            ax.scatter(qc[mask], depths[mask], s=15, color=color_map[lab], label=lab)

    ax.set_title("True lithostratigraphy")
    ax.set_xlabel("qc (MPa)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()

    # ---------- RIGHT: PRED ----------
    ax = axes[1]
    for lab in all_labels:
        mask = (profile_df["lithostrat_pred"].astype(str) == lab)
        if mask.any():
            ax.scatter(qc[mask], depths[mask], s=15, color=color_map[lab], label=lab)

    ax.set_title("Predicted lithostratigraphy")
    ax.set_xlabel("qc (MPa)")
    ax.invert_yaxis()

    # ---------- ONE shared legend ----------
    # build legend handles from the color_map
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[lab], label=lab)
        for lab in all_labels
    ]
    fig.legend(handles=handles,
               labels=all_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.03),  # move if you want it elsewhere
               ncol=3)

    fig.suptitle(f"CPT {cpt_id}: true vs predicted layers", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for legend + title
    plt.show()

example_cpt = df_known[cpt_col].iloc[0]
profile_df = predict_cpt_profile(example_cpt, df_known, best_crf, cpt_col, depth_col)

plot_cpt_true_vs_pred_with_legend(profile_df, example_cpt, depth_col=depth_col)
