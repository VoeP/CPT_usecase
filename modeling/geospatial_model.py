# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("C:/Users/joelle/Documents/Master/ProjectDS/vw_cpt_brussels_params_completeset_20250318_remapped.parquet", engine="pyarrow")
df = df.drop(columns=['pkey_sondering']) # not needed for analysis, links to data portal; not accessible for our team (non-Belgians)

#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isna().sum())

# %%

# duplicates analysis

# ID_COL = "index"
# KEY = ['sondeernummer', 'diepte']
# LABEL = 'lithostrat_id'

# # masks so no alterations to working df
# index_mask  = df[ID_COL].duplicated(keep=False)                 # index has duplicates
# key_mask  = df.duplicated(subset=KEY, keep=False)               # key has duplicates
# rows_mask = df.duplicated(keep=False)                           # fully identical rows

# n_index_dup  = int(index_mask.sum())
# n_key_dup  = int(key_mask.sum())
# n_rows_dup = int(rows_mask.sum())

# # duplicate keys (some are expected due to multiple labels)
# n_labels_per_key = (
#     df.loc[key_mask, [*KEY, LABEL]]
#       .groupby(KEY)[LABEL]
#       .nunique(dropna=False)
#       .rename('n_labels')
# )

# multiple_label_keys = n_labels_per_key.index[n_labels_per_key > 1]    # multi-label keys
# single_label_keys = n_labels_per_key.index[n_labels_per_key == 1]     # duplicated keys but single label

# # map row-wise: is this row in the duplicate-keys count?
# rows_in_multiple_label_keys = df.set_index(KEY).index.isin(multiple_label_keys)
# rows_in_single_label_keys = df.set_index(KEY).index.isin(single_label_keys)

# # index duplicates which are also in label duplicates
# index_dup_explained = (index_mask & rows_in_multiple_label_keys)
# n_index_dup_explained = int(index_dup_explained.sum())

# # unexplained index duplicates which are not in label duplicates
# index_dup_unexplained = (index_mask & ~rows_in_multiple_label_keys)
# n_index_dup_unexplained = int(index_dup_unexplained.sum())

# # key duplicates -> should not happen, some may be explained due to full row duplicates (same index)
# key_dup_not_index_dup = (key_mask & ~index_mask)
# n_key_dup_not_index_dup = int(key_dup_not_index_dup.sum())

# # confirm that the duplicated index actually differs in LABEL for multiple-label rows
# def index_group_has_label_conflict(g):
#     # same KEY?
#     same_key = (g[KEY].nunique(dropna=False) == 1).all()
#     # label conflict?
#     lab_conf = g[LABEL].nunique(dropna=False) > 1
#     return bool(same_key and lab_conf)

# index_conflict_flags = (
#     df.loc[index_mask]
#       .groupby(ID_COL)
#       .apply(index_group_has_label_conflict)
# )

# n_index_values_explained = int(index_conflict_flags.sum())  # number of index values (not rows) with label conflict & same key -> multiple-labels problem

# # remaining index-duplicate rows: are there measurement differences (should if index different but not logical)
# index_unexplained_df = df.loc[index_dup_unexplained].copy()
# exact_dup = int(index_unexplained_df.duplicated(keep=False).sum())

# # true only if all columns have unique values -> would need different index
# def unique_values_cols(g):
#     return (g.nunique(dropna=False) <= 1).all()
# index_unexplained_identical = (
#     index_unexplained_df
#     .groupby(ID_COL)
#     .apply(unique_values_cols)
# )
# n_index_unexplained_identical_ids = int(index_unexplained_identical.sum())
# n_index_unexplained_nonidentical_ids = int((~index_unexplained_identical).sum())

# # keys expected to have one label
# singlelabel_rows = df.set_index(KEY).index.isin(single_label_keys)
# df_singlelabel = df.loc[singlelabel_rows].copy()

# def key_group_identical(g):
#     # identical if all non-key columns identical
#     non_key_cols = [c for c in df.columns if c not in KEY]
#     varying = (g[non_key_cols].nunique(dropna=False) > 1).sum()
#     return varying == 0

# # boolean flag true if identical
# singlelabel_flag = (
#     df_singlelabel
#     .groupby(KEY)
#     .apply(key_group_identical)
# )

# n_singlelabel_identical_keys   = int(singlelabel_flag.sum())
# n_singlelabel_nonidentical_keys = int((~singlelabel_flag).sum())

# # summary to print
# summary = {
#     "INDEX duplicates rows": n_index_dup,
#     "KEY (sondeernummer+diepte) duplicates rows ": n_key_dup,
#     "Exact duplicates rows": n_rows_dup,
#     "INDEX duplicates rows EXPLAINED by multiple-labels": n_index_dup_explained,
#     "INDEX duplicates rows UNEXPLAINED by multiple-labels": n_index_dup_unexplained,
#     "KEY duplicates rows that are NOT INDEX duplicates when they should be": n_key_dup_not_index_dup,
#     "KEY duplicates with different indices due to multiple-labels": n_index_values_explained,
#     "UNEXPLAINED INDEX duplicates but IDENTICAL column values, likely safe to reduce to drop duplicates": n_index_unexplained_identical_ids,
#     "UNEXPLAINED INDEX duplicates which also have DIFFERING column values": n_index_unexplained_nonidentical_ids,
#     "Single-label KEY-duplicates but IDENTICAL column values, likely safe to drop duplicates": n_singlelabel_identical_keys,
#     "Single-label KEY-duplicates but DIFFERING column values": n_singlelabel_nonidentical_keys,
# }
# print("Duplicate Summary")
# for k,v in summary.items():
#     print(f"{k}: {v}")

# # example index-duplicate explained by conflict
# if n_index_dup_explained > 0:
#     ix = df.loc[index_dup_explained, ID_COL].iloc[0]
#     print(f"\nExample INDEX value explained by conflict: {ix!r}")
#     print(df[df[ID_COL] == ix].sort_values(KEY).head())

# # example unexplained index-duplicate
# if n_index_dup_unexplained > 0:
#     ix2 = df.loc[index_dup_unexplained, ID_COL].iloc[0]
#     print(f"\nExample UNEXPLAINED INDEX duplicate: {ix2!r}")
#     print(df[df[ID_COL] == ix2].sort_values(KEY).head())

# # example key-duplicate not in index-duplicates
# if n_key_dup_not_index_dup > 0:
#     ex_key = df.loc[key_dup_not_index_dup, KEY].iloc[0].to_dict()
#     print(f"\nExample KEY duplicate not tied to index-dup: {ex_key}")
#     print(df[(df['sondeernummer'] == ex_key['sondeernummer']) & (df['diepte'] == ex_key['diepte'])].head())

# %%

# define two data sets: working and target
LABEL = "lithostrat_id"
df[LABEL] = df[LABEL].astype("string").str.strip()

# consider onbekend as missing for now # TODO: check later if missing is still best option
invalid_labels = {"", "none", "nan", "null", "onbekend", "na", "n/a"}
valid_mask = ~df[LABEL].str.lower().isin(invalid_labels) & df[LABEL].notna()
df_work = df.loc[valid_mask].copy()     # working dataset (with lithostrat_id)
df_target = df.loc[~valid_mask].copy()  # target dataset
df[LABEL] = df[LABEL].astype("category")
df_work[LABEL] = df_work[LABEL].astype("category")
df_target[LABEL] = df_target[LABEL].astype("category")

# print("Data split summary")
# print(f"Total rows: {len(df)}")
# print(f"Rows WITH lithostrat_id (df_work): {len(df_work)}")
# print(f"Rows WITHOUT lithostrat_id (df_target): {len(df_target)}")

freq = df_work["lithostrat_id"].value_counts(dropna=False)
# print("Frequency table per label (lithostrat_id)")
# print(freq)

# %%

# print(df_work.isna().sum())
# print(df_target.isna().sum())

# imputing missing values, however, dropping NA could be a good alternative due to small proportion of missingness
# only for 3 vars, maybe extend later when developing data import automation (TODO)

# to impute parameters, formulas from ChatGPT were used -> needs adjustment after lit review TODO

def impute_params (df, overwrite = False):
    df = df.copy()

    # icn
    mask_icn = df['icn'].isna() if not overwrite else np.ones(len(df), dtype=bool)
    valid_icn = mask_icn & df['qtn'].gt(0) & df['fr'].gt(0)
    
    icn_new = np.sqrt((3.47 - np.log10(df.loc[valid_icn, 'qtn']))**2 +
                      (np.log10(df.loc[valid_icn, 'fr']) + 1.22)**2)
    
    df.loc[valid_icn, 'icn'] = icn_new
    
    # sbt
    def sbt_from_ic(ic):
        if pd.isna(ic):
            return np.nan
        if ic < 1.31: return 1
        elif ic < 2.05: return 2
        elif ic < 2.60: return 3
        elif ic < 2.95: return 4
        elif ic < 3.60: return 5
        else: return 6
    
    mask_sbt = df['sbt'].isna() if not overwrite else np.ones(len(df), dtype=bool)
    df.loc[mask_sbt, 'sbt'] = df.loc[mask_sbt, 'icn'].apply(sbt_from_ic)
    
    # ksbt
    df['ksbt'] = pd.to_numeric(df['ksbt'], errors='coerce')
    def ksbt_from_ic(ic):
        if pd.isna(ic):
            return np.nan
        if 1.0 < ic <= 3.27:
            return 10 ** (0.952 - 3.04 * ic)
        elif 3.27 > ic: # < 4.0:
            return 10 ** (-4.52 - 1.37 * ic)
        else:
            return np.nan
    
    mask_ksbt = df['ksbt'].isna() if not overwrite else np.ones(len(df), dtype=bool)
    df.loc[mask_ksbt, 'ksbt'] = df.loc[mask_ksbt, 'icn'].apply(ksbt_from_ic)
    
    return df

df_work = impute_params(df_work)
# print(df_work.isna().sum())

# %%

# # are there wrong values?
# # no negative values for qc and fs
# print("qc < 0:", (df_work['qc'] < 0).sum())
# print("fs < 0:", (df_work['fs'] < 0).sum())

# # this section is based on ChatGPT values: what are expected limits for each var -> needs adjustment after review from Vito TODO; are there even impossible values or just unlikely?
# # probably not needed for model development but check (TODO)

# def check_computed_limits(df,
#                           qc_max=50.0,  
#                           fs_max=1.0,   
#                           rf_max=20.0,
#                           fr_max=20.0,
#                           qtn_max=1e4,
#                           icn_max=5.0,
#                           ksbt_min=1e-12,
#                           ksbt_max=1e-2):
#     summary = {}

#     # qc (tip resistance, should be > 0)
#     summary['qc'] = {
#         "min": df["qc"].min(),
#         "max": df["qc"].max(),
#         "outside_exp": ((df["qc"] <= 0) | (df["qc"] > qc_max)).sum()
#     }

#     # fs (sleeve friction, should be >= 0)
#     summary['fs'] = {
#         "min": df["fs"].min(),
#         "max": df["fs"].max(),
#         "outside_exp": ((df["fs"] < 0) | (df["fs"] > fs_max)).sum()
#     }

#     # rf (friction ratio, %)
#     summary['rf'] = {
#         'min': df['rf'].min(),
#         'max': df['rf'].max(),
#         'outside_exp': ((df['rf'] <= 0) | (df['rf'] > rf_max)).sum()
#     }

#     # fr (normalized friction ratio, %)
#     summary['fr'] = {
#         'min': df['fr'].min(),
#         'max': df['fr'].max(),
#         'outside_exp': ((df['fr'] <= 0) | (df['fr'] > fr_max)).sum()
#     }

#     # qtn (normalized tip resistance)
#     summary['qtn'] = {
#         'min': df['qtn'].min(),
#         'max': df['qtn'].max(),
#         'outside_exp': ((df['qtn'] <= 0) | (df['qtn'] > qtn_max)).sum()
#     }

#     # icn (SBT index)
#     summary['icn'] = {
#         'min': df['icn'].min(),
#         'max': df['icn'].max(),
#         'outside_exp': ((df['icn'] < 0) | (df['icn'] > icn_max)).sum()
#     }

#     # ksbt (hydraulic conductivity, m/s)
#     summary['ksbt'] = {
#         'min': df['ksbt'].min(),
#         'max': df['ksbt'].max(),
#         'outside_exp': ((df['ksbt'] > ksbt_max) | (df['ksbt'] < ksbt_min)).sum()
#     }

#     return pd.DataFrame(summary).T

# print(check_computed_limits(df_work))

# %%

# building a feature extractor: from each CPT, the features need to be stored
vars_num = ["qc","fs","rf","qtn","fr","icn","ksbt"]
def extract_features(
    df: pd.DataFrame,
    vars_num,
    depth_col="diepte",
    depth_mtaw_col="diepte_mtaw",
    label_col="lithostrat_id",
    min_n=5,
):
    """
    feature extractor: 
    (1) groups by (sondeernummer, label_col)
    (2) for each group, computes:
    - IDs and position: sondeernummer, x, y, start_depth, end_depth, start_depth_mtaw, end_depth_mtaw, thickness, mean_depth_mtaw, n_samples_used
    - Stats for each var in vars_num: mean, sd, squared values mean, intra-step deltas, intra-layer deltas, intra-layer normalized deltas (mean, sd for all deltas)
    (3) Returns one row per (CPT, layer).
    """
    
    rows = []
    skipped = 0 # for layers with too few samples
    skipped_examples = []
    for (cpt, label), g in df.groupby(["sondeernummer", label_col], observed=False):
        g = g.sort_values(depth_col)
        if len(g) < min_n:
            skipped += 1
            skipped_examples.append((cpt, label, len(g)))
            continue
        
        # probably should have done this when also checking index dup, move? (TODO)
        if g["x"].nunique() != 1 or g["y"].nunique() != 1:
            print(f"Warning: multiple coordinate values in CPT={cpt}, layer={lab}: ")
        
        # id + positional
        start_depth = float(g[depth_col].min())
        end_depth   = float(g[depth_col].max())
        start_depth_mtaw = float(g[depth_mtaw_col].min())
        end_depth_mtaw   = float(g[depth_mtaw_col].max())
        row = {
            "sondeernummer": cpt,
            "layer_label": label,
            "x": float(g["x"].iloc[0]),
            "y": float(g["y"].iloc[0]),
            "start_depth": start_depth,
            "end_depth": end_depth,
            "start_depth_mtaw": start_depth_mtaw,
            "end_depth_mtaw": end_depth_mtaw,
            "thickness": end_depth - start_depth,
            "mean_depth_mtaw": float(g[depth_mtaw_col].mean()),
            "n_samples_used": int(len(g)),
        }
        
        # Numeric summaries
        for var in vars_num:
            if var not in g.columns:
            # included na value handling; not important for initial data set but maybe later when new CPTs are uploaded
                row[f"{var}_mean"] = np.nan
                row[f"{var}_sd"]  = np.nan
                row[f"{var}_sq_mean"] = np.nan
                row[f"{var}_dstep_mean"] = np.nan
                row[f"{var}_dstep_sd"]  = np.nan
                row[f"{var}_dstep_abs_mean"] = np.nan
                row[f"{var}_d_dz_mean"] = np.nan
                row[f"{var}_d_dz_sd"]  = np.nan
                continue
            s = g[var].dropna()
            row[f"{var}_mean"] = float(s.mean()) if len(s) else np.nan              # mean
            row[f"{var}_sd"]  = float(s.std(ddof=1)) if len(s) > 1 else np.nan      # sd     
            s2 = s ** 2                                                             # squared value
            row[f"{var}_sq_mean"] = float(s2.mean()) if len(s2) else np.nan         # mean of squared values

            # deltas
            sub = g[[depth_col, var]].dropna().sort_values(depth_col)

            if len(sub) > 1:
                v = sub[var].to_numpy(dtype=float) # for variable values
                z = sub[depth_col].to_numpy(dtype=float) # for depth values

                # net change end to start
                delta_layer = float(v[-1] - v[0])
                thickness = float(z[-1] - z[0])
                row[f"{var}_end_minus_start"] = delta_layer

                # normalized change: net variable change / thickness -> change in value per meter 
                row[f"{var}_end_minus_start_per_m"] = (delta_layer / thickness) if thickness > 0 else np.nan

                # local changes (derivatives)
                dv = np.diff(v)
                dz = np.diff(z)

                with np.errstate(divide='ignore', invalid='ignore'): # if z=0 or NA
                    deriv = dv / dz

                deriv = deriv[np.isfinite(deriv)] # drop NA or undefined
                if deriv.size:
                    row[f"{var}_d_dz_mean"] = float(deriv.mean()) # unweighted mean of slopes (per group)
                    row[f"{var}_d_dz_sd"]   = float(deriv.std(ddof=1)) if deriv.size > 1 else 0.0
                    # length-weighted mean slope
                    # weighted_mean = float((deriv * dz[np.isfinite(deriv)]).sum() / dz[np.isfinite(deriv)].sum())
                else:
                    row[f"{var}_d_dz_mean"] = np.nan
                    row[f"{var}_d_dz_sd"]   = np.nan
            else:
                # not enough measurements to compute slopes but should we use end/start if exactly one? (TODO)
                row[f"{var}_end_minus_start"] = np.nan
                row[f"{var}_end_minus_start_per_m"] = np.nan
                row[f"{var}_d_dz_mean"] = np.nan
                row[f"{var}_d_dz_sd"]   = np.nan
            
            # mean depth of the 3 largest values
            if len(s) >= 3:
                top3_idx = s.nlargest(3).index
                row[f"{var}_top3_mean_depth_rel"]  = float(g.loc[top3_idx, depth_col].mean())
                row[f"{var}_top3_mean_depth_mtaw"] = float(g.loc[top3_idx, depth_mtaw_col].mean())
            else:
                row[f"{var}_top3_mean_depth_rel"]  = np.nan
                row[f"{var}_top3_mean_depth_mtaw"] = np.nan
        
        rows.append(row)
    
    out = pd.DataFrame(rows)

    if skipped > 0:
        print(f"feature extraction function skipped {skipped} layer group(s) due to insufficient samples (< {min_n}).")
        # for cpt, label, n in skipped_examples:
        #    print(f"Example skipped group: sondeernummer={cpt}, layer_label={label}, n_samples={n}")
    id_cols = ["sondeernummer","layer_label","x","y","start_depth","end_depth", "start_depth_mtaw", "end_depth_mtaw", "thickness","mean_depth_mtaw","n_samples_used"]
    feature_cols = [c for c in out.columns if c not in id_cols]
    return out[id_cols + feature_cols]



feat_all = extract_features(
    df_work,
    vars_num=vars_num,
    depth_col="diepte",
    depth_mtaw_col="diepte_mtaw",
    label_col="lithostrat_id",
    min_n=5,                
)
# print(feat_all.head())
# feat_all.to_excel("features.xlsx", index=False)

# %%

# building the geospatial model
# first, using a grid-based method to create tiles
feat = feat_all.copy()

# 4 * 4 grid
Gx = Gy = 4
feat["xbin"] = pd.qcut(feat["x"], q=Gx, labels=False, duplicates="drop").astype(int)
feat["ybin"] = pd.qcut(feat["y"], q=Gy, labels=False, duplicates="drop").astype(int)
feat["tile"] = feat["xbin"] * Gy + feat["ybin"]

# tile counts
tile_sizes = feat.groupby("tile").size().sort_values(ascending=False)
print("Tiles and layer counts per tile:\n", tile_sizes.to_string())
n_tiles = feat["tile"].nunique()
print(f"Total unique tiles: {n_tiles}")

# randomly choose 20 % of tiles for train - maybe exclude adjacent tiles from test later or increase train
rng = np.random.default_rng(42)
train_tile_count = max(1, int(np.ceil(n_tiles * 0.20))) # 20 %
all_tiles = feat["tile"].unique()
train_tiles = set(rng.choice(all_tiles, size=train_tile_count, replace=False))
test_tiles  = set(all_tiles) - train_tiles

print(f"Chosen TRAIN tiles ({len(train_tiles)}): {sorted(train_tiles)}")
print(f"Chosen TEST  tiles ({len(test_tiles)}):  {sorted(test_tiles)}")

# match tiles by sondeernummer to create train and test df
train_cpts = set(feat.loc[feat["tile"].isin(train_tiles), "sondeernummer"].unique())
test_cpts  = set(feat.loc[feat["tile"].isin(test_tiles),  "sondeernummer"].unique())

feat_train = feat[feat["sondeernummer"].isin(train_cpts)].copy()
feat_test  = feat[feat["sondeernummer"].isin(test_cpts)].copy()

# tile sizes 
print("Tile sizes")
print(f"Tiles:   train={len(train_tiles):>3} | test={len(test_tiles):>3} | total={n_tiles}")
print(f"CPTs:    train={feat_train['sondeernummer'].nunique():>3} | test={feat_test['sondeernummer'].nunique():>3} | total={feat['sondeernummer'].nunique()}")
print(f"Layers:  train={len(feat_train):>5} | test={len(feat_test):>5} | total={len(feat)}")

# label counts, due to small train, some labels may not be part of train and can't be modelled
# print("Train label counts:")
# print(feat_train["layer_label"].value_counts().to_string())

# print("Test label counts:")
# print(feat_test["layer_label"].value_counts().to_string())

# %%

# features for train and test
id_cols = ["sondeernummer","layer_label","x","y","start_depth","end_depth", "start_depth_mtaw", "end_depth_mtaw", "thickness","mean_depth_mtaw","n_samples_used"]
X_train = feat_train.drop(columns=id_cols) # only predictors
y_train = feat_train["layer_label"].copy() # label to predict
X_test  = feat_test.drop(columns=id_cols)
y_test  = feat_test["layer_label"].copy()

# print("X shapes:", X_train.shape, X_test.shape)

seen_labels = set(feat_train["layer_label"].unique())
mask_seen = feat_test["layer_label"].isin(seen_labels)
print(f"Test rows with labels seen in train: {mask_seen.sum()} / {len(feat_test)} "
      f"({mask_seen.mean():.1%})")

# using k nearest ROWS (but not in same tile) and mtaw as elevation feature tau
def predict_label(row, feat_train, k=5, tau=None):
    dx = feat_train["x"].to_numpy() - float(row["x"]) # x coord distances
    dy = feat_train["y"].to_numpy() - float(row["y"]) # y coord distances
    # euclidian distances
    d = np.hypot(dx, dy)

    k = min(k, len(feat_train)) # k < n train samples
    nn_index = np.argpartition(d, kth=k-1)[:k] # argpartition: k smallest distances d in first k positions
    cand = feat_train.iloc[nn_index].copy()
    cand["_dist"] = d[nn_index] # adds distances to candidate neighbour rows

    if tau is not None:
        mean_mtaw = float(row["mean_depth_mtaw"])
        aligned = cand[np.abs(cand["mean_depth_mtaw"] - mean_mtaw) <= tau] # only accept candidates with elevation similar to labelled train data
        if not aligned.empty:
            cand = aligned

    weights = {}
    for label, dist in zip(cand["layer_label"], cand["_dist"]):
        w = 1.0 / (dist + 1.0)   # distance-weighted, could use other methods? Gaussian kernel? TODO: think about it
        weights[label] = weights.get(label, 0.0) + w

    best_label, best_weight = None, -1.0
    for label, w in weights.items():
        if w > best_weight:
            best_label, best_weight = label, w
    return best_label

# %%

# using k nearest unique CPTs, again mtaw as elevation feature tau
def predict_label_by_cpt(row, feat_train, k_cpt=5, tau=None):
    cpt_xy = (feat_train.groupby("sondeernummer")[["x","y"]]
              .first().reset_index())
    dx = cpt_xy["x"].to_numpy() - float(row["x"])
    dy = cpt_xy["y"].to_numpy() - float(row["y"])
    d = np.hypot(dx, dy) # euclidian distances

    k = min(k_cpt, len(cpt_xy)) # k < n train samples
    nn_index = np.argpartition(d, kth=k-1)[:k]
    cand_cpts = set(cpt_xy.iloc[nn_index]["sondeernummer"])

    # per CPT, choose labelled groups closest in absolute depth mtaw
    mean_mtaw = float(row["mean_depth_mtaw"])
    cand = (feat_train[feat_train["sondeernummer"].isin(cand_cpts)]
            .assign(_elev_diff=lambda df: (df["mean_depth_mtaw"] - mean_mtaw).abs()))
    cand = cand.sort_values(["sondeernummer","_elev_diff"]).groupby("sondeernummer").head(1).copy()

    # maybe remove this for small train?  too far in elevation
    if tau is not None:
        keep = cand["_elev_diff"] <= tau
        if keep.any():
            cand = cand[keep]

    # distance from test point to each CPT
    dmap = dict(zip(cpt_xy.iloc[nn_index]["sondeernummer"], d[nn_index]))
    cand["_dist"] = cand["sondeernummer"].map(dmap).astype(float)

    # distance-weighted, again maybe change
    weights = {}
    for label, dist in zip(cand["layer_label"], cand["_dist"]):
        w = 1.0 / (dist + 1.0)
        weights[label] = weights.get(label, 0.0) + w

    best_label, best_weight = None, -1.0
    for label, w in weights.items():
        if w > best_weight:
            best_label, best_weight = label, w
    return best_label

# %%
# evaluate

def eval_nearest_row_pred(feat_train, feat_test, k_list=(3,5,7,10), tau_list=(2.0,3.0,5.0)): # hyperparameter tuning
    seen = set(feat_train["layer_label"])
    test_set = feat_test[feat_test["layer_label"].isin(seen)].copy()
    rows = []
    for k in k_list:
        for tau in tau_list:
            preds = test_set.apply(lambda r: predict_label(r, feat_train, k=k, tau=tau), axis=1)
            acc = (preds == test_set["layer_label"]).mean()
            rows.append(("rows_based", k, tau, acc))
    return pd.DataFrame(rows, columns=["model","k","tau","acc"]).sort_values("acc", ascending=False)

def eval_nearest_cpt_pred(feat_train, feat_test, k_list=(3,5,7,10), tau_list=(2.0,3.0,5.0)):
    seen = set(feat_train["layer_label"])
    te = feat_test[feat_test["layer_label"].isin(seen)].copy()
    rows = []
    for k in k_list:
        for tau in tau_list:
            preds = te.apply(lambda r: predict_label_by_cpt(r, feat_train, k_cpt=k, tau=tau), axis=1)
            acc = (preds == te["layer_label"]).mean()
            rows.append(("cpt_based", k, tau, acc))
    return pd.DataFrame(rows, columns=["model","k","tau","acc"]).sort_values("acc", ascending=False)

print("Nearest row evaluation")
print(eval_nearest_row_pred(feat_train, feat_test).to_string(index=False))
print("Nearest CPT evaluation")
print(eval_nearest_cpt_pred(feat_train, feat_test).to_string(index=False))

# evaluation for row-based spatial model with k = 5, tau = 2.0
feat_test = feat_test.copy()
feat_test["pred_knn"] = feat_test.apply(
    lambda r: predict_label(r, feat_train, k=5, tau=2.0), axis=1
)

# accuracy only labels present in train
if mask_seen.any():
    acc = (feat_test.loc[mask_seen, "pred_knn"] == feat_test.loc[mask_seen, "layer_label"]).mean()
    print(f"Row-based spatial kNN (k=5, tau=2m) accuracy on seen-label subset: {acc:.3f}")
else:
    print("None of the test labels are present in train.")

# unseen labels
unseen_counts = feat_test.loc[~mask_seen, "layer_label"].value_counts()
if len(unseen_counts):
    print("Test labels unseen in TRAIN (not predictable):")
    print(unseen_counts.to_string())

# per-class accuracy used for model assessment/improvement

per_class_acc = (
    (feat_test.loc[mask_seen, "pred_knn"] == feat_test.loc[mask_seen, "layer_label"])
    .groupby(feat_test.loc[mask_seen, "layer_label"]).mean()
    .rename("acc")
)
support = feat_test.loc[mask_seen, "layer_label"].value_counts().rename("n")
print("Per-class (top 10 by support):")
print(pd.concat([per_class_acc, support], axis=1).sort_values("n", ascending=False).head(10).to_string())


# %%

# features for kNN-feature-based model (TODO)
# choose subset, modify later
feat_cols = [
    "qc_mean","fs_mean","qc_d_dz_mean","fs_d_dz_mean",
    "thickness","mean_depth_mtaw"
]

# per feature stats for train data set
mu = feat_train[feat_cols].mean()
sd = feat_train[feat_cols].std(ddof=1).replace(0, 1.0)