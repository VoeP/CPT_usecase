from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors

from data_module import (
    DataSet,
    extract_features,
    tile_split,
    segments_oi,
    segment_order,
    loo_cv,
    print_loocv,
)

# =============================================================================
# PATHS AND BASIC SETTINGS
# =============================================================================

try:
    THIS_FILE = Path(__file__).resolve()
except NameError:
    THIS_FILE = Path.cwd()  # note to self think about this: adjust if running in notebook
BASE_DIR = THIS_FILE.parent.parent   # repo root

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

PARQUET_PATH = DATA_DIR / "vw_cpt_brussels_params_completeset_20250318_remapped.parquet"
SPLIT_JSON_PATH = RESULTS_DIR / "split_res.json"

LABEL = "lithostrat_id"

# =============================================================================
# GLOBAL SWITCHES
# =============================================================================

# If True: run feature discriminability, feature selection and LOOCV tuning.
# If False: skip tuning and use manually specified features + hyperparameters.
MODEL_SELECTION_MODE = False  # <-- set to False for final model only.

# These are only used when MODEL_SELECTION_MODE is False.
# They were filled with the results from the previous run.
FINAL_FEAT_COLS = [
    "icn_sq_mean", "log_qc_mean", "rf_top3_mean_depth_rel", "qtn_top3_mean_depth_rel", "rf_mean"
]

FINAL_MODEL_TYPE = "row"

FINAL_PARAMS = {
    # based on best LOOCV config but note: alpha (spatial weight) has more influence than beta (feature weight)
    "k": 3,
    "tau": 20.0,
    "alpha": 2.5,
    "beta": 2.0,
}

# =============================================================================
# 1. LOAD DATA
# =============================================================================

ds = DataSet(
    path_to_parquet=PARQUET_PATH,
    segments_of_interest=segments_oi,
)

if "pkey_sondering" in ds.raw_df.columns:
    ds.raw_df = ds.raw_df.drop(columns=["pkey_sondering"])

df_work = ds.impute_params(
    overwrite=False,
    scope="known",
    use_imputation=True,
    drop_na_if_not_imputed=True,
    na_cols=["qc", "fs", "rf", "qtn", "fr", "icn", "sbt", "ksbt"],
)

df_target = ds.unlabeled_data.copy()

for d in (df_work, df_target):
    if LABEL in d.columns:
        d[LABEL] = d[LABEL].astype("category")

# print("Frequency table per label (lithostrat_id)")
# print(df_work[LABEL].value_counts(dropna=False))

# =============================================================================
# 1a. DUPLICATE ANALYSIS (COMMENTED; KEEP FOR LATER)
# =============================================================================
"""
# note to self think about this: only run if duplicate structure needs to be inspected in detail
df = df_work.copy()

ID_COL = "index"          # adjust if needed
KEY = ["sondeernummer", "diepte"]
LABEL = "lithostrat_id"

index_mask = df[ID_COL].duplicated(keep=False)
key_mask = df.duplicated(subset=KEY, keep=False)
rows_mask = df.duplicated(keep=False)

n_index_dup = int(index_mask.sum())
n_key_dup = int(key_mask.sum())
n_rows_dup = int(rows_mask.sum())

n_labels_per_key = (
    df.loc[key_mask, [*KEY, LABEL]]
      .groupby(KEY)[LABEL]
      .nunique(dropna=False)
      .rename("n_labels")
)

multiple_label_keys = n_labels_per_key.index[n_labels_per_key > 1]
single_label_keys   = n_labels_per_key.index[n_labels_per_key == 1]

rows_in_multiple_label_keys = df.set_index(KEY).index.isin(multiple_label_keys)
rows_in_single_label_keys   = df.set_index(KEY).index.isin(single_label_keys)

index_dup_explained = (index_mask & rows_in_multiple_label_keys)
n_index_dup_explained = int(index_dup_explained.sum())

index_dup_unexplained = (index_mask & ~rows_in_multiple_label_keys)
n_index_dup_unexplained = int(index_dup_unexplained.sum())

key_dup_not_index_dup = (key_mask & ~index_mask)
n_key_dup_not_index_dup = int(key_dup_not_index_dup.sum())

def index_group_has_label_conflict(g):
    same_key = (g[KEY].nunique(dropna=False) == 1).all()
    lab_conf = g[LABEL].nunique(dropna=False) > 1
    return bool(same_key and lab_conf)

index_conflict_flags = (
    df.loc[index_mask]
      .groupby(ID_COL)
      .apply(index_group_has_label_conflict)
)

n_index_values_explained = int(index_conflict_flags.sum())

index_unexplained_df = df.loc[index_dup_unexplained].copy()

def unique_values_cols(g):
    return (g.nunique(dropna=False) <= 1).all()

index_unexplained_identical = (
    index_unexplained_df
    .groupby(ID_COL)
    .apply(unique_values_cols)
)
n_index_unexplained_identical_ids = int(index_unexplained_identical.sum())
n_index_unexplained_nonidentical_ids = int((~index_unexplained_identical).sum())

singlelabel_rows = df.set_index(KEY).index.isin(single_label_keys)
df_singlelabel = df.loc[singlelabel_rows].copy()

def key_group_identical(g):
    non_key_cols = [c for c in df.columns if c not in KEY]
    varying = (g[non_key_cols].nunique(dropna=False) > 1).sum()
    return varying == 0

singlelabel_flag = (
    df_singlelabel
    .groupby(KEY)
    .apply(key_group_identical)
)

n_singlelabel_identical_keys    = int(singlelabel_flag.sum())
n_singlelabel_nonidentical_keys = int((~singlelabel_flag).sum())

summary = {
    "INDEX duplicates rows": n_index_dup,
    "KEY (sondeernummer+diepte) duplicates rows ": n_key_dup,
    "Exact duplicates rows": n_rows_dup,
    "INDEX duplicates rows EXPLAINED by multiple-labels": n_index_dup_explained,
    "INDEX duplicates rows UNEXPLAINED by multiple-labels": n_index_dup_unexplained,
    "KEY duplicates rows that are NOT INDEX duplicates when they should be": n_key_dup_not_index_dup,
    "KEY duplicates with different indices due to multiple-labels": n_index_values_explained,
    "UNEXPLAINED INDEX duplicates but IDENTICAL column values, likely safe to reduce to drop duplicates": n_index_unexplained_identical_ids,
    "UNEXPLAINED INDEX duplicates which also have DIFFERING column values": n_index_unexplained_nonidentical_ids,
    "Single-label KEY-duplicates but IDENTICAL column values, likely safe to drop duplicates": n_singlelabel_identical_keys,
    "Single-label KEY-duplicates but DIFFERING column values": n_singlelabel_nonidentical_keys,
}
print("Duplicate Summary")
for k, v in summary.items():
    print(f"{k}: {v}")
"""

# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

vars_num = ["qc", "fs", "rf", "qtn", "fr", "icn", "ksbt"]

feat_all = extract_features(
    df_work,
    vars_num=vars_num,
    depth_col="diepte",
    depth_mtaw_col="diepte_mtaw",
    label_col="lithostrat_id",
    cpt_col="sondeernummer",
    min_n=5,
)

# print("Feature table (head):")
# print(feat_all.head())

# =============================================================================
# 2a. MAP sondering_id ONTO FEATURES
# =============================================================================

if "sondering_id" not in df_work.columns:
    raise ValueError("Column 'sondering_id' not found in df_work, but JSON split uses it.")

id_map = (
    df_work[["sondeernummer", "sondering_id"]]
    .drop_duplicates("sondeernummer")
)

feat_all = feat_all.merge(id_map, on="sondeernummer", how="left")

if feat_all["sondering_id"].isna().any():
    n_missing = int(feat_all["sondering_id"].isna().sum())
    print(f"WARNING: sondering_id missing for {n_missing} feature rows after merge")

# =============================================================================
# 3. TILES FOR SPATIAL CONTEXT
# =============================================================================

(
    feat_with_tiles,   # all rows with tile/xbin/ybin
    feat_train_tile,   # TILE-BASED train set
    feat_test_tile,    # TILE-BASED test set
    X_train_unused,
    X_test_unused,
    y_train_unused,
    y_test_unused,
) = tile_split(
    feat_all,
    Gx=5,
    Gy=5,
    train_frac=0.70,
    random_state=22,
    x_col="x",
    y_col="y",
    cpt_col="sondeernummer",
    label_col="layer_label",
    extra_id_cols=[
        "sondering_id",
        "start_depth",
        "end_depth",
        "start_depth_mtaw",
        "end_depth_mtaw",
        "thickness",
        "mean_depth_mtaw",
        "n_samples_used",
    ],
)

for df_ in (feat_train_tile, feat_test_tile):
    if "layer_label" in df_.columns:
        df_["layer_label"] = df_["layer_label"].astype("category")

# tile_counts = feat_with_tiles.groupby("tile")["sondeernummer"].nunique()
# print("Tiles and CPT counts per tile:")
# print(tile_counts)
# print("Total number of unique tiles:", tile_counts.index.nunique())

# =============================================================================
# 4. LOAD JSON SPLIT AND BUILD TRAIN / TEST BY sondering_id (for comparison with other models, only needed for reporting)
# =============================================================================

with open(SPLIT_JSON_PATH, "r") as f:
    split = json.load(f)

train_ids_json = {str(i) for i in split["train_ids"]}
test_ids_json = {str(i) for i in split["test_ids"]}

CPT_ID_COL = "sondering_id"
if CPT_ID_COL not in feat_with_tiles.columns:
    raise ValueError(f"{CPT_ID_COL!r} not present in feat_with_tiles")

feat_with_tiles["cpt_id_str"] = feat_with_tiles[CPT_ID_COL].astype(str)

feat_train_json = feat_with_tiles[feat_with_tiles["cpt_id_str"].isin(train_ids_json)].copy()
feat_test_json  = feat_with_tiles[feat_with_tiles["cpt_id_str"].isin(test_ids_json)].copy()

print(f"feat_train_json shape (JSON split via sondering_id): {feat_train_json.shape}")
print(f"feat_test_json  shape (JSON split via sondering_id): {feat_test_json.shape}")

for df_ in (feat_train_json, feat_test_json):
    if "layer_label" in df_.columns:
        df_["layer_label"] = df_["layer_label"].astype("category")

# === HYBRID MODEL: USE TILE-BASED SPLIT ===
feat_train = feat_train_tile.copy()
feat_test  = feat_test_tile.copy()

# training set used for LOOCV (tile-based train CPTs)
train_with_tiles = feat_train.copy()

# =============================================================================
# 5. FEATURE ENGINEERING FOR HYBRID MODELS
# =============================================================================

eps = 1e-6

for df_ in (feat_train, feat_test, train_with_tiles):
    df_["rf_mean"] = 100.0 * (df_["fs_mean"] / (df_["qc_mean"] + eps))
    df_["log_qc_mean"] = np.log(np.maximum(df_["qc_mean"], eps))
    df_["log_rf_mean"] = np.log(np.maximum(df_["rf_mean"], eps))

feat_cols = [
    "qc_mean", "fs_mean",
    "rf_mean",
    "qc_d_dz_mean", "fs_d_dz_mean",
    "thickness",
    "log_qc_mean", "log_rf_mean",
]

min_thickness_for_stats = 0.5 # as suggested in Q&A, influences JSON acc results

feat_train_f = feat_train[feat_train["thickness"] >= min_thickness_for_stats].copy()
mu_global = feat_train_f[feat_cols].mean(numeric_only=True)
sd_global = feat_train_f[feat_cols].std(ddof=1, numeric_only=True).replace(0, 1.0)

z_scores_train_global = ((feat_train[feat_cols].fillna(mu_global)) - mu_global) / sd_global
z_scores_test_global = ((feat_test[feat_cols].fillna(mu_global)) - mu_global) / sd_global

certainty_weight = None

# ============================================================================
# 6. FEATURE DIAGNOSTICS: DISCRIMINABILITY AND CORRELATION
# ============================================================================

def feature_discriminability_table(df, feat_cols, label_col="layer_label"):
    """
    Computes ANOVA F-statistic and p-value for each feature vs the label.
    Higher F -> stronger between-class separation relative to within-class variance.
    """
    sub = df.dropna(subset=feat_cols + [label_col]).copy()

    # encode categories as integers for f_classif
    if isinstance(sub[label_col].dtype, pd.CategoricalDtype):
        y = sub[label_col].cat.codes.to_numpy()
    else:
        y = sub[label_col].astype("category").cat.codes.to_numpy()

    X = sub[feat_cols].to_numpy(dtype=float)

    F_vals, p_vals = f_classif(X, y)

    tab = pd.DataFrame({
        "feature": feat_cols,
        "F_stat": F_vals,
        "p_value": p_vals,
    }).sort_values("F_stat", ascending=False)

    return tab.reset_index(drop=True)

# columns that are NOT candidate features
id_like_cols = [
    "sondeernummer",
    "sondering_id",
    "cpt_id_str",
    "layer_label",
    "x",
    "y",
    "tile",
    "xbin",  # note to self think about this: keep spatial info in spatial distance, not in features
    "start_depth",
    "end_depth",
    "start_depth_mtaw",
    "end_depth_mtaw",
    "mean_depth_mtaw",
    "n_samples_used",
]

id_like_cols = [c for c in id_like_cols if c in train_with_tiles.columns]

numeric_cols = train_with_tiles.select_dtypes(include=[np.number]).columns.tolist()

candidate_feat_cols = [c for c in numeric_cols if c not in id_like_cols]

def select_features_by_f_and_corr(
    df,
    disc_table,
    max_features=20,
    corr_threshold=0.95,
):
    """
    Takes a discriminability table (feature, F_stat, p_value),
    and selects at most `max_features` features by:
      1) sorting by F_stat (desc),
      2) iteratively adding features that have |corr| < corr_threshold
         to all already selected ones.
    """
    disc_sorted = disc_table.sort_values("F_stat", ascending=False).reset_index(drop=True)
    feat_list = disc_sorted["feature"].tolist()

    corr = df[feat_list].corr().abs()

    selected = []

    for feat in feat_list:
        if len(selected) == 0:
            selected.append(feat)
        else:
            too_correlated = any(
                corr.loc[feat, s] >= corr_threshold
                for s in selected
                if feat in corr.index and s in corr.columns
            )
            if not too_correlated:
                selected.append(feat)

        if len(selected) >= max_features:
            break

    return selected

if MODEL_SELECTION_MODE:
    print("\nNumber of candidate numeric features:", len(candidate_feat_cols))
    print("Example candidate features:", candidate_feat_cols[:15])

    disc_all = feature_discriminability_table(
        train_with_tiles,
        feat_cols=candidate_feat_cols,
        label_col="layer_label",
    )

    print("\n=== FEATURE DISCRIMINABILITY (ANOVA F-test) ON TRAIN-WITH-TILES (ALL NUMERIC FEATURES) ===")
    print(disc_all.head(30).to_string(index=False))

    selected_feat_cols = select_features_by_f_and_corr(
        df=train_with_tiles,
        disc_table=disc_all,
        max_features=5,      # after running LOOCV on sets of max_features, 5 is best
        corr_threshold=0.95,
    )

    print("\n=== SELECTED FEATURES AFTER F + CORRELATION PRUNING ===")
    print("Number of selected features:", len(selected_feat_cols))
    for f in selected_feat_cols:
        row = disc_all[disc_all["feature"] == f].iloc[0]
        print(f"{f:35s}  F={row['F_stat']:.2f}, p={row['p_value']:.2e}")

    # override base feat_cols with selected ones
    feat_cols = selected_feat_cols
else:
    # Directly use manually specified features from the top of the file
    if not FINAL_FEAT_COLS:
        raise ValueError(
            "MODEL_SELECTION_MODE is False but FINAL_FEAT_COLS is empty. "
            "Fill FINAL_FEAT_COLS with best feature set."
        )
    feat_cols = FINAL_FEAT_COLS
    print("\nSkipping feature selection — using FINAL_FEAT_COLS:")
    print(feat_cols)

# =============================================================================
# 7. HELPER: WEIGHTING AND HYBRID PREDICTORS
# =============================================================================

def inv_weight(d, labels=None, certainty_weights=None):
    d = np.asarray(d, dtype=float)
    w = 1.0 / (d + 1.0)
    if labels is not None and certainty_weights is not None:
        cw = np.array([float(certainty_weights.get(lab, 1.0)) for lab in labels], dtype=float)
        w *= cw
    return w

def predict_label_hybrid_row(
    row,
    feat_train,
    z_scores_train,
    feat_cols,
    mu,
    sd,
    k=7,
    tau=10.0,
    alpha=1.0,
    beta=1.0,
    x_sd=None,
    y_sd=None,
    certainty_weights=None,
    tau_max=None,
    tau_multi=1.5,
):
    if x_sd is None:
        x_sd = float(feat_train["x"].std(ddof=1)) or 1.0
    if y_sd is None:
        y_sd = float(feat_train["y"].std(ddof=1)) or 1.0

    x0 = float(row["x"])
    y0 = float(row["y"])

    dx = (feat_train["x"].to_numpy(dtype=float) - x0) / x_sd
    dy = (feat_train["y"].to_numpy(dtype=float) - y0) / y_sd
    d_spatial = np.hypot(dx, dy)

    mean_mtaw_q = float(row["mean_depth_mtaw"])
    elev_diff_all = np.abs(feat_train["mean_depth_mtaw"].to_numpy(float) - mean_mtaw_q)

    if tau is None:
        tau_current = 0.0
    else:
        tau_current = float(tau)

    if tau_max is None:
        tau_cap = (tau_current if tau_current > 0 else 1.0) * 5.0
    else:
        tau_cap = float(tau_max)

    while True:
        mask = elev_diff_all <= tau_current
        if np.count_nonzero(mask) >= max(1, k) or tau_current >= tau_cap:
            break
        tau_current *= tau_multi

    if not np.any(mask):
        k_all = min(int(k), len(feat_train))
        nn_local_index = np.argpartition(elev_diff_all, kth=k_all - 1)[:k_all]
        mask = np.zeros(len(feat_train), dtype=bool)
        mask[nn_local_index] = True

    row_vals = row.reindex(feat_cols).to_numpy(dtype=float, copy=False)
    row_vec = (row_vals - mu.to_numpy(dtype=float)) / sd.to_numpy(dtype=float)

    train_mat_sub = z_scores_train[mask]
    d_feat_sub = np.linalg.norm(train_mat_sub - row_vec, axis=1)

    d_spatial_sub = d_spatial[mask]
    d_combo_sub = np.sqrt(alpha * (d_spatial_sub ** 2) + beta * (d_feat_sub ** 2))

    k_eff = min(int(k), len(d_combo_sub))
    nn_local_index = np.argpartition(d_combo_sub, kth=k_eff - 1)[:k_eff]
    train_indices = np.flatnonzero(mask)[nn_local_index]

    labels = feat_train.iloc[train_indices]["layer_label"].to_numpy()
    d_only_nn = d_combo_sub[nn_local_index]
    w = inv_weight(d_only_nn, labels=labels, certainty_weights=certainty_weights)

    weights_by_label = {}
    for lab, wi in zip(labels, w):
        weights_by_label[lab] = weights_by_label.get(lab, 0.0) + float(wi)

    sorted_labels = sorted(weights_by_label.items(), key=lambda x: x[1], reverse=True)

    best_label, best_weight = sorted_labels[0]
    second_label, second_weight = (None, 0.0)
    if len(sorted_labels) > 1:
        second_label, second_weight = sorted_labels[1]

    return {
        "best_label": best_label,
        "best_weight": best_weight,
        "second_label": second_label,
        "second_weight": second_weight,
        "weights_by_label": weights_by_label,
    }

def predict_label_hybrid_cpt(
    row,
    feat_train,
    z_scores_train,
    feat_cols,
    mu,
    sd,
    k_cpt=7,
    tau=10.0,
    alpha=1.0,
    beta=1.0,
    x_sd=None,
    y_sd=None,
    certainty_weights=None,
):
    cpt_xy = (
        feat_train.groupby("sondeernummer")[["x", "y"]]
        .first()
        .reset_index()
    )

    if x_sd is None:
        x_sd = float(cpt_xy["x"].std(ddof=1)) or 1.0
    if y_sd is None:
        y_sd = float(cpt_xy["y"].std(ddof=1)) or 1.0

    x0 = float(row["x"])
    y0 = float(row["y"])

    dx = (cpt_xy["x"].to_numpy(dtype=float) - x0) / x_sd
    dy = (cpt_xy["y"].to_numpy(dtype=float) - y0) / y_sd
    d_spatial = np.hypot(dx, dy)

    k_eff = min(int(k_cpt), len(cpt_xy))
    nn_index = np.argpartition(d_spatial, kth=k_eff - 1)[:k_eff]
    cand_cpts = cpt_xy.iloc[nn_index]["sondeernummer"].to_numpy()
    d_spatial_nn = d_spatial[nn_index]

    mean_mtaw_q = float(row["mean_depth_mtaw"])

    cand = (
        feat_train[feat_train["sondeernummer"].isin(cand_cpts)]
        .assign(_elev_diff=lambda df: (df["mean_depth_mtaw"] - mean_mtaw_q).abs())
    )
    cand = (
        cand.sort_values(["sondeernummer", "_elev_diff"])
        .groupby("sondeernummer")
        .head(1)
        .copy()
    )

    if tau is not None:
        keep = cand["_elev_diff"] <= tau
        if keep.any():
            cand = cand[keep]

    dmap = dict(zip(cpt_xy.iloc[nn_index]["sondeernummer"], d_spatial_nn))
    cand["_d_spatial"] = cand["sondeernummer"].map(dmap).astype(float)

    row_vec = ((row.reindex(feat_cols).to_numpy(float) - mu.to_numpy(float)) / sd.to_numpy(float))

    train_idx = cand.index
    train_mat = z_scores_train.loc[train_idx, feat_cols].to_numpy(float)
    d_feat = np.linalg.norm(train_mat - row_vec, axis=1)

    d_combo = np.sqrt(alpha * cand["_d_spatial"].to_numpy() ** 2 + beta * d_feat ** 2)

    labels = cand["layer_label"].to_numpy()
    w = inv_weight(d_combo, labels=labels, certainty_weights=certainty_weights)

    weights_by_label = {}
    for lab, wi in zip(labels, w):
        weights_by_label[lab] = weights_by_label.get(lab, 0.0) + float(wi)

    sorted_labels = sorted(weights_by_label.items(), key=lambda x: x[1], reverse=True)

    best_label, best_weight = sorted_labels[0]
    second_label, second_weight = (None, 0.0)
    if len(sorted_labels) > 1:
        second_label, second_weight = sorted_labels[1]

    return {
        "best_label": best_label,
        "best_weight": best_weight,
        "second_label": second_label,
        "second_weight": second_weight,
        "weights_by_label": weights_by_label,
    }

# =============================================================================
# 8. LOOCV-BASED HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparams_loocv(
    train_with_tiles,
    feat_cols,
    model_type="row",
    k_list=(1, 3, 5, 7, 9),
    tau_list=(1.0, 5.0, 10.0, 15.0, 20.0),
    alpha_list=(0, 0.5, 1.0, 1.5, 2.0, 2.5),
    beta_list=(0, 0.5, 1.0, 1.5, 2.0, 2.5),
    min_thickness=0.5,
    certainty_weights=None,
):
    results = []

    for k in k_list:
        for tau in tau_list:
            for alpha in alpha_list:
                for beta in beta_list:
                    total_correct = 0
                    total_n = 0

                    for t, train_df, test_df, X_tr, X_te, y_tr, y_te in loo_cv(
                        train_with_tiles,
                        label_col="layer_label",
                        cpt_col="sondeernummer",
                    ):
                        if len(train_df) == 0 or len(test_df) == 0:
                            continue

                        train = train_df.copy()
                        test = test_df.copy()

                        for df_ in (feat_train, feat_test, train_with_tiles, feat_train_json, feat_test_json):
                            df_["rf_mean"] = 100.0 * (df_["fs_mean"] / (df_["qc_mean"] + eps))
                            df_["log_qc_mean"] = np.log(np.maximum(df_["qc_mean"], eps))
                            df_["log_rf_mean"] = np.log(np.maximum(df_["rf_mean"], eps))

                        train_f = train[train["thickness"] >= min_thickness].copy()
                        mu_fold = train_f[feat_cols].mean(numeric_only=True)
                        sd_fold = train_f[feat_cols].std(ddof=1, numeric_only=True).replace(0, 1.0)

                        z_scores_train_fold = ((train[feat_cols].fillna(mu_fold)) - mu_fold) / sd_fold

                        x_sd_fold = float(train["x"].std(ddof=1)) or 1.0
                        y_sd_fold = float(train["y"].std(ddof=1)) or 1.0

                        if model_type == "row":
                            preds_dict = test.apply(
                                lambda r: predict_label_hybrid_row(
                                    r,
                                    feat_train=train,
                                    z_scores_train=z_scores_train_fold,
                                    feat_cols=feat_cols,
                                    mu=mu_fold,
                                    sd=sd_fold,
                                    k=k,
                                    tau=tau,
                                    alpha=alpha,
                                    beta=beta,
                                    x_sd=x_sd_fold,
                                    y_sd=y_sd_fold,
                                    certainty_weights=certainty_weights,
                                ),
                                axis=1,
                            )
                        elif model_type == "cpt":
                            preds_dict = test.apply(
                                lambda r: predict_label_hybrid_cpt(
                                    r,
                                    feat_train=train,
                                    z_scores_train=z_scores_train_fold,
                                    feat_cols=feat_cols,
                                    mu=mu_fold,
                                    sd=sd_fold,
                                    k_cpt=k,
                                    tau=tau,
                                    alpha=alpha,
                                    beta=beta,
                                    x_sd=x_sd_fold,
                                    y_sd=y_sd_fold,
                                    certainty_weights=certainty_weights,
                                ),
                                axis=1,
                            )
                        else:
                            raise ValueError(f"Unknown model_type {model_type!r}")

                        test["pred_label1"] = preds_dict.apply(lambda d: d["best_label"])

                        seen_labels_fold = set(train["layer_label"])
                        mask_seen_fold = test["layer_label"].isin(seen_labels_fold)

                        n_seen = int(mask_seen_fold.sum())
                        if n_seen == 0:
                            continue

                        correct = int(
                            (test.loc[mask_seen_fold, "pred_label1"] ==
                             test.loc[mask_seen_fold, "layer_label"]).sum()
                        )

                        total_correct += correct
                        total_n += n_seen

                    if total_n == 0:
                        mean_acc = np.nan
                    else:
                        mean_acc = total_correct / total_n

                    results.append(
                        {
                            "model_type": model_type,
                            "k": k,
                            "tau": tau,
                            "alpha": alpha,
                            "beta": beta,
                            "loocv_acc": mean_acc,
                        }
                    )

                    print(
                        f"LOOCV {model_type}: k={k}, tau={tau}, alpha={alpha}, beta={beta}, "
                        f"acc={mean_acc:.3f}" if not np.isnan(mean_acc) else
                        f"LOOCV {model_type}: k={k}, tau={tau}, alpha={alpha}, beta={beta}, acc=nan"
                    )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("loocv_acc", ascending=False).reset_index(drop=True)
    return res_df

if MODEL_SELECTION_MODE:
    print("Starting LOOCV hyperparameter tuning for hybrid ROW-based model")
    loocv_row_results = tune_hyperparams_loocv(
        train_with_tiles=train_with_tiles,
        feat_cols=feat_cols,
        model_type="row",
        k_list=(2, 3, 4),
        tau_list=(10.0, 20.0, 30.0),
        alpha_list=(1.0, 2.5, 4.0),
        beta_list=(1.0, 2.0, 3.0),
        min_thickness=0.5,
        certainty_weights=certainty_weight,
    )

    print("Top 5 LOOCV configs for ROW-based hybrid model:")
    print(loocv_row_results.head(5).to_string(index=False))

    print("Starting LOOCV hyperparameter tuning for hybrid CPT-based model")
    loocv_cpt_results = tune_hyperparams_loocv(
        train_with_tiles=train_with_tiles,
        feat_cols=feat_cols,
        model_type="cpt",
        k_list=(2, 3, 4),
        tau_list=(10.0, 20.0, 30.0),
        alpha_list=(1.0, 2.5, 4.0),
        beta_list=(1.0, 2.0, 3.0),
        min_thickness=0.5,
        certainty_weights=certainty_weight,
    )

    print("Top 5 LOOCV configs for CPT-based hybrid model:")
    print(loocv_cpt_results.head(5).to_string(index=False))

    best_row = loocv_row_results.iloc[0]
    best_cpt = loocv_cpt_results.iloc[0]

    print("\nBest ROW-based config from LOOCV:")
    print(best_row)

    print("\nBest CPT-based config from LOOCV:")
    print(best_cpt)

    if best_row["loocv_acc"] >= best_cpt["loocv_acc"]:
        chosen_model_type = "row"
        chosen_params = best_row
    else:
        chosen_model_type = "cpt"
        chosen_params = best_cpt

    print(f"\nChosen model type based on LOOCV: {chosen_model_type}")
    print("Chosen parameters:")
    print(chosen_params)
else:
    # Use manually specified final model
    chosen_model_type = FINAL_MODEL_TYPE
    chosen_params = FINAL_PARAMS

    # sanity checks
    required_keys = {"k", "tau", "alpha", "beta"}
    if not required_keys.issubset(chosen_params.keys()):
        raise ValueError(
            f"FINAL_PARAMS must contain keys {required_keys}, got {set(chosen_params.keys())}"
        )

    print("\nSkipping LOOCV — using FINAL_MODEL_TYPE and FINAL_PARAMS:")
    print("chosen_model_type:", chosen_model_type)
    print("chosen_params:", chosen_params)

# ============================================================================
#  9. LOOCV SENSITIVITY: NUMBER OF FEATURES (ROW & CPT HYBRID)
# ============================================================================

# def loocv_feature_count_sensitivity(
#     train_with_tiles,
#     selected_feat_cols,
#     model_type="row",
#     N_list=(5, 10, 15, 20),
#     k_list=(5, 7, 9),
#     tau_list=(5.0, 10.0),
#     alpha_list=(0.5, 1.0, 2.0),
#     beta_list=(0.5, 1.0, 2.0),
#     min_thickness=0.5,
#     certainty_weights=None,
# ):
#     """
#     For each N in N_list:
#       - use the first N features from selected_feat_cols
#       - run LOOCV tuning over (k, tau, alpha, beta)
#       - record best LOOCV accuracy and corresponding hyperparams
#     """
#     results = []

#     for N in N_list:
#         feats_N = selected_feat_cols[:N]
#         print(f"\n=== LOOCV for model={model_type}, N_features={N} ===")
#         print("Features:", feats_N)

#         loocv_res = tune_hyperparams_loocv(
#             train_with_tiles=train_with_tiles,
#             feat_cols=feats_N,
#             model_type=model_type,
#             k_list=k_list,
#             tau_list=tau_list,
#             alpha_list=alpha_list,
#             beta_list=beta_list,
#             min_thickness=min_thickness,
#             certainty_weights=certainty_weights,
#         )

#         # best config for this N
#         best_idx = loocv_res["loocv_acc"].idxmax()
#         best_row = loocv_res.loc[best_idx].copy()
#         best_row["N_features"] = N
#         best_row["feature_list"] = feats_N
#         results.append(best_row)

#         print("Best for N =", N)
#         print(best_row)

#     results_df = pd.DataFrame(results).sort_values("loocv_acc", ascending=False).reset_index(drop=True)
#     return results_df


# # run for ROW-based model
# N_list = (5, 10, 15, 20)   # TODO: think about smaller steps

# print("\n>>> LOOCV sensitivity (ROW-based hybrid) vs number of features <<<")
# row_feat_sensitivity = loocv_feature_count_sensitivity(
#     train_with_tiles=train_with_tiles,
#     selected_feat_cols=selected_feat_cols,
#     model_type="row",
#     N_list=N_list,
#     k_list=(5, 7, 9),
#     tau_list=(5.0, 10.0),
#     alpha_list=(0.5, 1.0, 2.0),
#     beta_list=(0.5, 1.0, 2.0),
#     min_thickness=0.5,
#     certainty_weights=certainty_weight,
# )

# print("\nROW-based: best configs by LOOCV accuracy (different N):")
# print(row_feat_sensitivity.to_string(index=False))

# # run for CPT-based model
# print("\n>>> LOOCV sensitivity (CPT-based hybrid) vs number of features <<<")
# cpt_feat_sensitivity = loocv_feature_count_sensitivity(
#     train_with_tiles=train_with_tiles,
#     selected_feat_cols=selected_feat_cols,
#     model_type="cpt",
#     N_list=N_list,
#     k_list=(5, 7, 9),
#     tau_list=(5.0, 10.0),
#     alpha_list=(0.5, 1.0, 2.0),
#     beta_list=(0.5, 1.0, 2.0),
#     min_thickness=0.5,
#     certainty_weights=certainty_weight,
# )

# print("\nCPT-based: best configs by LOOCV accuracy (different N):")
# print(cpt_feat_sensitivity.to_string(index=False))

# =============================================================================
# 10. FIT CHOSEN MODEL ON TILE-BASED TRAIN AND EVALUATE ON TILE-BASED TEST
# =============================================================================

train_f = feat_train[feat_train["thickness"] >= min_thickness_for_stats].copy()
mu_final = train_f[feat_cols].mean(numeric_only=True)
sd_final = train_f[feat_cols].std(ddof=1, numeric_only=True).replace(0, 1.0)

z_scores_train_final = ((feat_train[feat_cols].fillna(mu_final)) - mu_final) / sd_final
z_scores_test_final = ((feat_test[feat_cols].fillna(mu_final)) - mu_final) / sd_final

x_sd_final = float(feat_train["x"].std(ddof=1)) or 1.0
y_sd_final = float(feat_train["y"].std(ddof=1)) or 1.0

feat_test = feat_test.copy()

if chosen_model_type == "row":
    preds_final = feat_test.apply(
        lambda r: predict_label_hybrid_row(
            r,
            feat_train=feat_train,
            z_scores_train=z_scores_train_final,
            feat_cols=feat_cols,
            mu=mu_final,
            sd=sd_final,
            k=int(chosen_params["k"]),
            tau=float(chosen_params["tau"]),
            alpha=float(chosen_params["alpha"]),
            beta=float(chosen_params["beta"]),
            x_sd=x_sd_final,
            y_sd=y_sd_final,
            certainty_weights=certainty_weight,
        ),
        axis=1,
    )
elif chosen_model_type == "cpt":
    preds_final = feat_test.apply(
        lambda r: predict_label_hybrid_cpt(
            r,
            feat_train=feat_train,
            z_scores_train=z_scores_train_final,
            feat_cols=feat_cols,
            mu=mu_final,
            sd=sd_final,
            k_cpt=int(chosen_params["k"]),
            tau=float(chosen_params["tau"]),
            alpha=float(chosen_params["alpha"]),
            beta=float(chosen_params["beta"]),
            x_sd=x_sd_final,
            y_sd=y_sd_final,
            certainty_weights=certainty_weight,
        ),
        axis=1,
    )
else:
    raise ValueError(f"Unexpected chosen_model_type {chosen_model_type!r}")

feat_test["pred_hybrid"] = preds_final
feat_test["pred_label1"] = feat_test["pred_hybrid"].apply(lambda d: d["best_label"])
feat_test["pred_w1"] = feat_test["pred_hybrid"].apply(lambda d: d["best_weight"])
feat_test["pred_label2"] = feat_test["pred_hybrid"].apply(lambda d: d["second_label"])
feat_test["pred_w2"] = feat_test["pred_hybrid"].apply(lambda d: d["second_weight"])

mask_seen_final = feat_test["layer_label"].isin(set(feat_train["layer_label"]))

if mask_seen_final.any():
    acc_final = (
        feat_test.loc[mask_seen_final, "pred_label1"]
        == feat_test.loc[mask_seen_final, "layer_label"]
    ).mean()
    print(f"\nAccuracy on TILE TEST (before post-processing, chosen model): {acc_final:.3f}")
else:
    print("No overlapping labels between train and test for final evaluation")

per_class_acc_final = (
    (feat_test.loc[mask_seen_final, "pred_label1"] ==
     feat_test.loc[mask_seen_final, "layer_label"])
    .groupby(feat_test.loc[mask_seen_final, "layer_label"])
    .mean()
    .rename("acc")
)
support_final = feat_test.loc[mask_seen_final, "layer_label"].value_counts().rename("n")

print("\nPer-label accuracy before post-processing (chosen model):")
print(pd.concat([per_class_acc_final, support_final], axis=1).sort_values("n", ascending=False))

# =============================================================================
# COMPARE ACCURACY AMONG MODELS USING JSON FILE SPLIT
# ============================================================================

feat_test_json_eval = feat_test_json.copy()

# reuse same scaling from tile-train:
if chosen_model_type == "row":
    preds_json = feat_test_json_eval.apply(
        lambda r: predict_label_hybrid_row(
            r,
            feat_train=feat_train,  # still train on tile-based set
            z_scores_train=z_scores_train_final,
            feat_cols=feat_cols,
            mu=mu_final,
            sd=sd_final,
            k=int(chosen_params["k"]),
            tau=float(chosen_params["tau"]),
            alpha=float(chosen_params["alpha"]),
            beta=float(chosen_params["beta"]),
            x_sd=x_sd_final,
            y_sd=y_sd_final,
            certainty_weights=certainty_weight,
        ),
        axis=1,
    )
else:
    preds_json = feat_test_json_eval.apply(
        lambda r: predict_label_hybrid_cpt(
            r,
            feat_train=feat_train,
            z_scores_train=z_scores_train_final,
            feat_cols=feat_cols,
            mu=mu_final,
            sd=sd_final,
            k_cpt=int(chosen_params["k"]),
            tau=float(chosen_params["tau"]),
            alpha=float(chosen_params["alpha"]),
            beta=float(chosen_params["beta"]),
            x_sd=x_sd_final,
            y_sd=y_sd_final,
            certainty_weights=certainty_weight,
        ),
        axis=1,
    )

feat_test_json_eval["pred_label1"] = preds_json.apply(lambda d: d["best_label"])

mask_seen_json = feat_test_json_eval["layer_label"].isin(set(feat_train["layer_label"]))
acc_json = (
    feat_test_json_eval.loc[mask_seen_json, "pred_label1"]
    == feat_test_json_eval.loc[mask_seen_json, "layer_label"]
).mean()
print(f"\nAccuracy on JSON TEST (tile-trained model, before post-processing): {acc_json:.3f}")

# =============================================================================
# 11. POST-PROCESSING: Enforcing segment order
# =============================================================================

order_index = {lab: i for i, lab in enumerate(segment_order)}

def is_allowed_label(candidate, prev):
    """
    Only enforce stratigraphic order:

    - If there is no previous label (topmost), anything is allowed.
    - Otherwise, the new label's index must be >= the previous label's index
      in segment_order (no upwards jump).
    """
    if candidate is None:
        return False
    if prev is None:
        return True
    # enforce monotone non-decreasing order
    return order_index[candidate] >= order_index[prev]


def postprocess_cpt(group):
    """
    Enforce only segment order along a CPT:
    - sort by mean_depth_mtaw (top to bottom; currently descending),
    - try best prediction, else second, else fall back to previous label
      (or best if there is no previous).
    """
    g = group.sort_values("mean_depth_mtaw", ascending=False).copy()

    corrected = []
    prev = None
    changes = 0

    for _, row in g.iterrows():
        best = row["pred_label1"]
        second = row["pred_label2"]

        if is_allowed_label(best, prev):
            chosen = best
            used_second = False
        elif is_allowed_label(second, prev):
            chosen = second
            used_second = True
        else:
            # if neither best nor second respects order, keep previous label if possible
            if prev is not None:
                chosen = prev
                used_second = False
            else:
                # at the very top we have to pick something; accept best even if order_index issue
                chosen = best
                used_second = False

        if (chosen != best) or used_second:
            changes += 1

        corrected.append(chosen)
        prev = chosen

    return corrected, changes


def apply_postprocessing(feat_df):
    df = feat_df.copy()
    total_changes = 0

    for cpt, group in df.groupby("sondeernummer"):
        corrected_list, changes = postprocess_cpt(group)
        df.loc[
            group.sort_values("mean_depth_mtaw", ascending=False).index,
            "pred_hybrid_corrected",
        ] = corrected_list
        total_changes += changes

    return df, total_changes


feat_test_corrected, n_corrected = apply_postprocessing(feat_test)
print(f"\nNumber of corrected rows: {n_corrected}")

mask_seen_corr = feat_test_corrected["layer_label"].isin(set(feat_train["layer_label"]))

final_acc_corr = (
    feat_test_corrected.loc[mask_seen_corr, "pred_hybrid_corrected"]
    == feat_test_corrected.loc[mask_seen_corr, "layer_label"]
).mean()
print(f"FINAL post-processed accuracy on TILE TEST (chosen model): {final_acc_corr:.3f}")

# =============================================================================
# Spatial Leakage Eval: No need to re-run
# =============================================================================

def tile_leakage_summary(feat_train, feat_test):
    tiles_train = set(feat_train["tile"].unique())
    tiles_test = set(feat_test["tile"].unique())
    overlap_tiles = tiles_train & tiles_test

    frac_test_in_overlap_tiles = feat_test["tile"].isin(overlap_tiles).mean()

    print("Train tiles:", sorted(tiles_train))
    print("Test tiles:", sorted(tiles_test))
    print("Overlapping tiles:", sorted(overlap_tiles))
    print(f"Fraction of TEST rows in overlapping tiles: {frac_test_in_overlap_tiles:.3f}")

    if "sondeernummer" in feat_test.columns:
        cpt_counts_test = (
            feat_test.groupby("tile")["sondeernummer"]
            .nunique()
            .sort_index()
        )
        print("\nTest CPT counts per tile:")
        print(cpt_counts_test)

    return {
        "tiles_train": tiles_train,
        "tiles_test": tiles_test,
        "overlap_tiles": overlap_tiles,
        "frac_test_in_overlap_tiles": frac_test_in_overlap_tiles,
    }


def add_nearest_train_distance(feat_train, feat_test):
    cpt_train = (
        feat_train.groupby("sondeernummer")[["x", "y"]]
        .first()
        .reset_index()
    )
    cpt_test = (
        feat_test.groupby("sondeernummer")[["x", "y"]]
        .first()
        .reset_index()
    )

    train_xy = cpt_train[["x", "y"]].to_numpy(dtype=float)
    test_xy = cpt_test[["x", "y"]].to_numpy(dtype=float)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_xy)
    dists, indices = nn.kneighbors(test_xy)

    cpt_test["dist_to_nearest_train"] = dists[:, 0]

    print("Distance to nearest TRAIN CPT (test CPTs):")
    print(cpt_test["dist_to_nearest_train"].describe())

    for R in [100, 250, 500, 1000]:
        frac = (cpt_test["dist_to_nearest_train"] <= R).mean()
        print(f"Fraction of test CPTs within {R} units of a train CPT: {frac:.3f}")

    return cpt_test


def tile_coords(tile_id, Gx=5):
    row = tile_id // Gx
    col = tile_id % Gx
    return row, col


def tile_adjacent(t1, t2, Gx=5):
    r1, c1 = tile_coords(t1, Gx)
    r2, c2 = tile_coords(t2, Gx)
    return max(abs(r1 - r2), abs(c1 - c2)) == 1  # 8-neighbourhood


def adjacency_leakage(feat_train, feat_test, Gx=5):
    tiles_train = set(feat_train["tile"].unique())
    tiles_test = set(feat_test["tile"].unique())

    adjacent_test_tiles = set()
    for tt in tiles_test:
        if any(tile_adjacent(tt, tr, Gx=Gx) for tr in tiles_train):
            adjacent_test_tiles.add(tt)

    frac_test_tiles_adjacent = len(adjacent_test_tiles) / max(1, len(tiles_test))
    print("Train tiles:", sorted(tiles_train))
    print("Test tiles:", sorted(tiles_test))
    print("Adjacent test tiles (touching a train tile):", sorted(adjacent_test_tiles))
    print(f"Fraction of test tiles adjacent to at least one train tile: {frac_test_tiles_adjacent:.3f}")

# print("\n=== SPATIAL LEAKAGE DIAGNOSTICS (TILE-BASED FINAL MODEL) ===")

# # 1) Tile-overlap leakage: should be 0
# _ = tile_leakage_summary(feat_train, feat_test)

# # 2) Distance-based leakage
# _ = add_nearest_train_distance(feat_train, feat_test)

# # 3) Tile adjacency leakage (could be avoided if wanted)
# adjacency_leakage(feat_train, feat_test, Gx=5)