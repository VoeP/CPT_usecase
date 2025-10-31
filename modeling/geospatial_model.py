# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx

df = pd.read_parquet("C:/Users/joelle/Documents/Master/ProjectDS/vw_cpt_brussels_params_completeset_20250318_remapped.parquet", engine = "pyarrow")
df = df.drop(columns = ['pkey_sondering']) # not needed for analysis, links to data portal; not accessible for our team (non-Belgians)

#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isna().sum())

# %%

# # duplicates analysis

# ID_COL = "index"
# KEY = ['sondeernummer', 'diepte']
# LABEL = 'lithostrat_id'

# # masks so no alterations to working df
# index_mask  = df[ID_COL].duplicated(keep = False)                 # index has duplicates
# key_mask  = df.duplicated(subset = KEY, keep = False)               # key has duplicates
# rows_mask = df.duplicated(keep = False)                           # fully identical rows

# n_index_dup  = int(index_mask.sum())
# n_key_dup  = int(key_mask.sum())
# n_rows_dup = int(rows_mask.sum())

# # duplicate keys (some are expected due to multiple labels)
# n_labels_per_key = (
#     df.loc[key_mask, [*KEY, LABEL]]
#       .groupby(KEY)[LABEL]
#       .nunique(dropna = False)
#       .rename('n_labels')
# )

# multiple_label_keys = n_labels_per_key.index[n_labels_per_key > 1]    # multi-label keys
# single_label_keys = n_labels_per_key.index[n_labels_per_key ==  1]     # duplicated keys but single label

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
#     same_key = (g[KEY].nunique(dropna = False) ==  1).all()
#     # label conflict?
#     lab_conf = g[LABEL].nunique(dropna = False) > 1
#     return bool(same_key and lab_conf)

# index_conflict_flags = (
#     df.loc[index_mask]
#       .groupby(ID_COL)
#       .apply(index_group_has_label_conflict)
# )

# n_index_values_explained = int(index_conflict_flags.sum())  # number of index values (not rows) with label conflict & same key -> multiple-labels problem

# # remaining index-duplicate rows: are there measurement differences (should if index different but not logical)
# index_unexplained_df = df.loc[index_dup_unexplained].copy()
# exact_dup = int(index_unexplained_df.duplicated(keep = False).sum())

# # true only if all columns have unique values -> would need different index
# def unique_values_cols(g):
#     return (g.nunique(dropna = False) <=  1).all()
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
#     varying = (g[non_key_cols].nunique(dropna = False) > 1).sum()
#     return varying ==  0

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
#     print(df[df[ID_COL] ==  ix].sort_values(KEY).head())

# # example unexplained index-duplicate
# if n_index_dup_unexplained > 0:
#     ix2 = df.loc[index_dup_unexplained, ID_COL].iloc[0]
#     print(f"\nExample UNEXPLAINED INDEX duplicate: {ix2!r}")
#     print(df[df[ID_COL] ==  ix2].sort_values(KEY).head())

# # example key-duplicate not in index-duplicates
# if n_key_dup_not_index_dup > 0:
#     ex_key = df.loc[key_dup_not_index_dup, KEY].iloc[0].to_dict()
#     print(f"\nExample KEY duplicate not tied to index-dup: {ex_key}")
#     print(df[(df['sondeernummer'] ==  ex_key['sondeernummer']) & (df['diepte'] ==  ex_key['diepte'])].head())

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

freq = df_work["lithostrat_id"].value_counts(dropna = False)
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
    mask_icn = df['icn'].isna() if not overwrite else np.ones(len(df), dtype = bool)
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
    
    mask_sbt = df['sbt'].isna() if not overwrite else np.ones(len(df), dtype = bool)
    df.loc[mask_sbt, 'sbt'] = df.loc[mask_sbt, 'icn'].apply(sbt_from_ic)
    
    # ksbt
    df['ksbt'] = pd.to_numeric(df['ksbt'], errors = 'coerce')
    def ksbt_from_ic(ic):
        if pd.isna(ic):
            return np.nan
        if 1.0 < ic <=  3.27:
            return 10 ** (0.952 - 3.04 * ic)
        elif 3.27 > ic: # < 4.0:
            return 10 ** (-4.52 - 1.37 * ic)
        else:
            return np.nan
    
    mask_ksbt = df['ksbt'].isna() if not overwrite else np.ones(len(df), dtype = bool)
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
#                           qc_max = 50.0,  
#                           fs_max = 1.0,   
#                           rf_max = 20.0,
#                           fr_max = 20.0,
#                           qtn_max = 1e4,
#                           icn_max = 5.0,
#                           ksbt_min = 1e-12,
#                           ksbt_max = 1e-2):
#     summary = {}

#     # qc (tip resistance, should be > 0)
#     summary['qc'] = {
#         "min": df["qc"].min(),
#         "max": df["qc"].max(),
#         "outside_exp": ((df["qc"] < =  0) | (df["qc"] > qc_max)).sum()
#     }

#     # fs (sleeve friction, should be > =  0)
#     summary['fs'] = {
#         "min": df["fs"].min(),
#         "max": df["fs"].max(),
#         "outside_exp": ((df["fs"] < 0) | (df["fs"] > fs_max)).sum()
#     }

#     # rf (friction ratio, %)
#     summary['rf'] = {
#         'min': df['rf'].min(),
#         'max': df['rf'].max(),
#         'outside_exp': ((df['rf'] < =  0) | (df['rf'] > rf_max)).sum()
#     }

#     # fr (normalized friction ratio, %)
#     summary['fr'] = {
#         'min': df['fr'].min(),
#         'max': df['fr'].max(),
#         'outside_exp': ((df['fr'] < =  0) | (df['fr'] > fr_max)).sum()
#     }

#     # qtn (normalized tip resistance)
#     summary['qtn'] = {
#         'min': df['qtn'].min(),
#         'max': df['qtn'].max(),
#         'outside_exp': ((df['qtn'] < =  0) | (df['qtn'] > qtn_max)).sum()
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
    depth_col = "diepte",
    depth_mtaw_col = "diepte_mtaw",
    label_col = "lithostrat_id",
    min_n = 5,
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
    for (cpt, label), g in df.groupby(["sondeernummer", label_col], observed = False):
        g = g.sort_values(depth_col)
        if len(g) < min_n:
            skipped +=  1
            skipped_examples.append((cpt, label, len(g)))
            continue
        
        # probably should have done this when also checking index dup, move? (TODO)
        if g["x"].nunique() !=  1 or g["y"].nunique() !=  1:
            print(f"Warning: multiple coordinate values in CPT = {cpt}, layer = {label}: ")
        
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
                row[f"{var}_dstep_mean"] = np.nan       # average change per step (ca 2cm)
                row[f"{var}_dstep_sd"]  = np.nan        # variation per step
                row[f"{var}_dstep_abs_mean"] = np.nan   # absolute value
                row[f"{var}_d_dz_mean"] = np.nan        # average change per meter depth
                row[f"{var}_d_dz_sd"]  = np.nan         # variation per meter depth
                continue
            s = g[var].dropna()
            row[f"{var}_mean"] = float(s.mean()) if len(s) else np.nan              # mean
            row[f"{var}_sd"]  = float(s.std(ddof = 1)) if len(s) > 1 else np.nan      # sd     
            s2 = s ** 2                                                             # squared value
            row[f"{var}_sq_mean"] = float(s2.mean()) if len(s2) else np.nan         # mean of squared values

            # deltas
            sub = g[[depth_col, var]].dropna().sort_values(depth_col)

            if len(sub) > 1:
                v = sub[var].to_numpy(dtype = float) # for variable values
                z = sub[depth_col].to_numpy(dtype = float) # for depth values

                # net change end to start
                delta_layer = float(v[-1] - v[0])
                thickness = float(z[-1] - z[0])
                row[f"{var}_end_minus_start"] = delta_layer

                # normalized change: net variable change / thickness -> change in value per meter 
                row[f"{var}_end_minus_start_per_m"] = (delta_layer / thickness) if thickness > 0 else np.nan

                # local changes (derivatives)
                dv = np.diff(v)
                dz = np.diff(z)

                with np.errstate(divide = 'ignore', invalid = 'ignore'): # if z = 0 or NA
                    deriv = dv / dz

                deriv = deriv[np.isfinite(deriv)] # drop NA or undefined
                if deriv.size:
                    row[f"{var}_d_dz_mean"] = float(deriv.mean()) # unweighted mean of slopes (per group)
                    row[f"{var}_d_dz_sd"]   = float(deriv.std(ddof = 1)) if deriv.size > 1 else 0.0
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
            if len(s) >=  3:
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
        #    print(f"Example skipped group: sondeernummer = {cpt}, layer_label = {label}, n_samples = {n}")
    id_cols = ["sondeernummer","layer_label","x","y","start_depth","end_depth", "start_depth_mtaw", "end_depth_mtaw", "thickness","mean_depth_mtaw","n_samples_used"]
    feature_cols = [c for c in out.columns if c not in id_cols]
    return out[id_cols + feature_cols]



feat_all = extract_features(
    df_work,
    vars_num = vars_num,
    depth_col = "diepte",
    depth_mtaw_col = "diepte_mtaw",
    label_col = "lithostrat_id",
    min_n = 5,                
)
print(feat_all.head())
# feat_all.to_excel("features.xlsx", index = False)

# %%

# building the geospatial model
# first, using a grid-based method to create tiles
feat = feat_all.copy()

# 4 * 4 grid
Gx = Gy = 4
feat["xbin"] = pd.qcut(feat["x"], q = Gx, labels = False, duplicates = "drop").astype(int)
feat["ybin"] = pd.qcut(feat["y"], q = Gy, labels = False, duplicates = "drop").astype(int)
feat["tile"] = feat["xbin"] * Gy + feat["ybin"]

# tile counts
tile_sizes = feat.groupby("tile").size().sort_values(ascending = False)
print("Tiles and layer counts per tile:\n", tile_sizes.to_string())
n_tiles = feat["tile"].nunique()
print(f"Total unique tiles: {n_tiles}")

# randomly choose 20 % of tiles for train - maybe exclude adjacent tiles from test later or increase train
rng = np.random.default_rng(42)
train_tile_count = max(1, int(np.ceil(n_tiles * 0.30))) # 30 %
all_tiles = feat["tile"].unique()
train_tiles = set(rng.choice(all_tiles, size = train_tile_count, replace = False))
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
print(f"Tiles:   train = {len(train_tiles):>3} | test = {len(test_tiles):>3} | total = {n_tiles}")
print(f"CPTs:    train = {feat_train['sondeernummer'].nunique():>3} | test = {feat_test['sondeernummer'].nunique():>3} | total = {feat['sondeernummer'].nunique()}")
print(f"Layers:  train = {len(feat_train):>5} | test = {len(feat_test):>5} | total = {len(feat)}")

# label counts, due to small train, some labels may not be part of train and can't be modelled
# print("Train label counts:")
# print(feat_train["layer_label"].value_counts().to_string())

# print("Test label counts:")
# print(feat_test["layer_label"].value_counts().to_string())

# %%

# features for train and test
id_cols = ["sondeernummer","layer_label","x","y","start_depth","end_depth", "start_depth_mtaw", "end_depth_mtaw", "thickness","mean_depth_mtaw","n_samples_used"]
X_train = feat_train.drop(columns = id_cols) # only predictors
y_train = feat_train["layer_label"].copy() # label to predict
X_test  = feat_test.drop(columns = id_cols)
y_test  = feat_test["layer_label"].copy()

# print("X shapes:", X_train.shape, X_test.shape)

seen_labels = set(feat_train["layer_label"].unique())
mask_seen = feat_test["layer_label"].isin(seen_labels)
print(f"Test rows with labels seen in train: {mask_seen.sum()} / {len(feat_test)} "
      f"({mask_seen.mean():.1%})")

# using k nearest ROWS (but not in same tile) and mtaw as elevation feature tau
def predict_label(row, feat_train, k = 5, tau = None):
    dx = feat_train["x"].to_numpy() - float(row["x"]) # x coord distances
    dy = feat_train["y"].to_numpy() - float(row["y"]) # y coord distances
    # euclidian distances
    d = np.hypot(dx, dy)

    k = min(k, len(feat_train)) # k < n train samples
    nn_index = np.argpartition(d, kth = k-1)[:k] # argpartition: k smallest distances d in first k positions
    cand = feat_train.iloc[nn_index].copy()
    cand["_dist"] = d[nn_index] # adds distances to candidate neighbour rows

    mean_mtaw = float(row["mean_depth_mtaw"])

    cand = cand.copy()
    cand["_elev_diff"] = (cand["mean_depth_mtaw"] - mean_mtaw).abs()
    aligned = cand[cand["_elev_diff"] <=  tau]
    if aligned.empty:
        aligned = cand.nsmallest(k, "_elev_diff")
    cand = aligned.drop(columns = "_elev_diff")

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
def predict_label_by_cpt(row, feat_train, k_cpt = 5, tau = None):
    cpt_xy = (feat_train.groupby("sondeernummer")[["x","y"]]
              .first().reset_index())
    dx = cpt_xy["x"].to_numpy() - float(row["x"])
    dy = cpt_xy["y"].to_numpy() - float(row["y"])
    d = np.hypot(dx, dy) # euclidian distances

    k = min(k_cpt, len(cpt_xy)) # k < n train samples
    nn_index = np.argpartition(d, kth = k-1)[:k]
    cand_cpts = set(cpt_xy.iloc[nn_index]["sondeernummer"])

    # per CPT, choose labelled groups closest in absolute depth mtaw
    mean_mtaw = float(row["mean_depth_mtaw"])
    cand = (feat_train[feat_train["sondeernummer"].isin(cand_cpts)]
            .assign(_elev_diff = lambda df: (df["mean_depth_mtaw"] - mean_mtaw).abs()))
    cand = cand.sort_values(["sondeernummer","_elev_diff"]).groupby("sondeernummer").head(1).copy()

    # maybe remove this for small train?  too far in elevation
    if tau is not None:
        keep = cand["_elev_diff"] <=  tau
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

def eval_nearest_row_pred(feat_train, feat_test, k_list = (3,5,7,10), tau_list = (2.0,3.0,5.0)): # hyperparameter tuning
    seen = set(feat_train["layer_label"])
    test_set = feat_test[feat_test["layer_label"].isin(seen)].copy()
    rows = []
    for k in k_list:
        for tau in tau_list:
            preds = test_set.apply(lambda r: predict_label(r, feat_train, k = k, tau = tau), axis = 1)
            acc = (preds == test_set["layer_label"]).mean()
            rows.append(("rows_based", k, tau, acc))
    return pd.DataFrame(rows, columns = ["model","k","tau","acc"]).sort_values("acc", ascending = False)

def eval_nearest_cpt_pred(feat_train, feat_test, k_list = (3,5,7,10), tau_list = (2.0,3.0,5.0)):
    seen = set(feat_train["layer_label"])
    test_set = feat_test[feat_test["layer_label"].isin(seen)].copy()
    rows = []
    for k in k_list:
        for tau in tau_list:
            preds = test_set.apply(lambda r: predict_label_by_cpt(r, feat_train, k_cpt = k, tau = tau), axis = 1)
            acc = (preds == test_set["layer_label"]).mean()
            rows.append(("cpt_based", k, tau, acc))
    return pd.DataFrame(rows, columns = ["model","k","tau","acc"]).sort_values("acc", ascending = False)

print("Nearest row evaluation")
print(eval_nearest_row_pred(feat_train, feat_test).to_string(index = False))
print("Nearest CPT evaluation")
print(eval_nearest_cpt_pred(feat_train, feat_test).to_string(index = False))

# evaluation for row-based spatial model with k = 5, tau = 2.0
feat_test = feat_test.copy()
feat_test["pred_knn"] = feat_test.apply(lambda r: predict_label(r, feat_train, k = 5, tau = 2.0), axis = 1)

# accuracy only labels present in train
if mask_seen.any():
    acc = (feat_test.loc[mask_seen, "pred_knn"] == feat_test.loc[mask_seen, "layer_label"]).mean()
    print(f"Row-based spatial kNN (k = 5, tau = 2m) accuracy on seen-label subset: {acc:.3f}")
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
print(pd.concat([per_class_acc, support], axis = 1).sort_values("n", ascending = False).head(10).to_string())


# %%

# combine spatial model with features
# for now, choose drilling measurements (qc, fs - cone and sleeve resistance), the respective vertical changes, geological data (thickness, elevation)

# added this part after the csv was uploaded
eps = 1e-6
for df in (feat_train, feat_test):
    df["rf_mean"] = 100.0 * (df["fs_mean"] / (df["qc_mean"] + eps))
    df["log_qc_mean"] = np.log(np.maximum(df["qc_mean"], eps))
    df["log_rf_mean"] = np.log(np.maximum(df["rf_mean"], eps))

# modification after csv: add rf_mean, logs
feat_cols = [
    "qc_mean", "fs_mean",
    "rf_mean",
    "qc_d_dz_mean", "fs_d_dz_mean",
    "thickness",
    "log_qc_mean", "log_rf_mean"
]

min_thickness = 0.5  # meters
feat_train_f = feat_train[feat_train["thickness"] >= min_thickness].copy()
feat_test_f  = feat_test[ feat_test["thickness"]  >= min_thickness].copy()

# modified the mu, sd calc after the Q&A: option to drop thin layers
mu = feat_train_f[feat_cols].mean(numeric_only=True)
sd = feat_train_f[feat_cols].std(ddof=1, numeric_only=True).replace(0, 1.0)
z_scores_train = ((feat_train_f[feat_cols] - mu) / sd)

# calculating z-scores
# # standard calc (without removing thin layers)   
# mu = feat_train[feat_cols].mean(numeric_only = True)
# sd = feat_train[feat_cols].std(ddof = 1, numeric_only = True).replace(0,1)
z_scores_train = ((feat_train[feat_cols].fillna(mu)) - mu) / sd
z_scores_test  = ((feat_test[feat_cols].fillna(mu)) - mu) / sd

# reset index, maybe not needed if we clean indexes in initial data handling (TODO)
feat_train_reset = feat_train.reset_index(drop = True)
feat_test_reset  = feat_test.reset_index(drop = True)

# numpy matrices should be better for distance calculations
train_feat_mat = z_scores_train.to_numpy(dtype = float)
test_feat_mat  = z_scores_test.to_numpy(dtype = float)

# adding label certainty as weight
# certainty_scale = {"low": 0.5, "medium": 1.0, "high": 2.0, "very high": 6.0}
# label_certainty = {
#     "Quartair": "low",
#     "Diest": "medium",
#     "Bolderberg": "medium",
#     "Sint_Huibrechts_Hern": "medium",
#     "Ursel": "high",
#     "Asse": "high",
#     "Wemmel": "high",
#     "Lede": "high",
#     "Brussel": "high",
#     "Merelbeke": "very high",
#     "Kwatrecht": "high",
#     "Mont_Panisel": "high",
#     "Aalbeke": "very high",
#     "Mons_en_Pevele": "very high",
# }
# certainty_weight = {lab: certainty_scale[label_certainty[lab]]
#     for lab in label_certainty}

certainty_weight = None # not using label certainty weighting

def inv_weight(d, labels = None, certainty_weights = None):
    d = np.asarray(d, dtype = float)
    w = 1.0 / (d + 1.0)
    # additional for label weighting, turns out that label weighting did not improve the model
    # made it into a conditional statement instead
    if labels is not None and certainty_weights is not None:
        cw = np.array([float(certainty_weights.get(lab, 1.0)) for lab in labels], dtype=float)
        w *= cw
    return w

def predict_label_hybrid(
    row,
    feat_train,
    z_scores_train,
    feat_cols, mu, sd,
    k = 5, tau = 5.0, alpha = 1.0, beta = 1.0,
    x_sd = None, y_sd = None,
    certainty_weights = None, # addition for label certainty weighting
    tau_max = None, tau_multi = 1.5 # addition for adaptive tau:
    # instead of removing tau filter entirely if no data with similar tau is available, rather expand tau
):

    # using spatial scaling (with sds from train and fallback to 1.0) - introduced after initial model assessment
    if x_sd is None:
        x_sd = float(feat_train["x"].std(ddof = 1)) or 1.0
    if y_sd is None:
        y_sd = float(feat_train["y"].std(ddof = 1)) or 1.0

    # from here on, initial model
    x0 = float(row["x"]); y0 = float(row["y"])
    dx = (feat_train["x"].to_numpy(dtype = float) - x0) / x_sd
    dy = (feat_train["y"].to_numpy(dtype = float) - y0) / y_sd
    d_spatial = np.hypot(dx, dy)

    # similar elevation requirement
    mean_mtaw_q = float(row["mean_depth_mtaw"])
    elev_diff_all = np.abs(feat_train["mean_depth_mtaw"].to_numpy(float) - mean_mtaw_q)

    if tau is None:
        tau_current = 0.0
    else:
        tau_current = float(tau)
    tau_cap  = float(tau_max) if tau_max is not None else (tau_current if tau_current > 0 else 1.0) * 5.0

    while True:
        mask = elev_diff_all <= tau_current
        if np.count_nonzero(mask) >= max(1, k) or tau_current >= tau_cap:
            break
        tau_current *= tau_multi  # expand window
    
    if not np.any(mask):
        k_all = min(int(k), len(feat_train))
        nn_local_index = np.argpartition(elev_diff_all, kth=k_all-1)[:k_all]
        mask = np.zeros(len(feat_train), dtype=bool)
        mask[nn_local_index] = True

    # row vectors
    row_vals = row.reindex(feat_cols).to_numpy(dtype = float, copy = False)
    row_vec  = (row_vals - mu.to_numpy(dtype = float)) / sd.to_numpy(dtype = float)
    
    # feature distances
    train_mat_sub = z_scores_train[mask]
    d_feat_sub = np.linalg.norm(train_mat_sub - row_vec, axis = 1)

    # spatial distances
    d_spatial_sub = d_spatial[mask]
    d_combo_sub = np.sqrt(alpha * (d_spatial_sub ** 2) + beta * (d_feat_sub ** 2))

    k_eff = min(int(k), len(d_combo_sub))
    nn_local_index = np.argpartition(d_combo_sub, kth = k_eff - 1)[:k_eff]
    train_indices = np.flatnonzero(mask)[nn_local_index]

    labels = feat_train.iloc[train_indices]["layer_label"].to_numpy()
    d_only_nn = d_combo_sub[nn_local_index]
    w = inv_weight(d_only_nn, labels=labels, certainty_weights=certainty_weights)

    weights_by_label = {}
    for lab, wi in zip(labels, w):
        weights_by_label[lab] = weights_by_label.get(lab, 0.0) + float(wi)

    return max(weights_by_label.items(), key = lambda kv: kv[1])[0]

# %% eval hybrid model
def eval_hybrid_row_pred(
    feat_train,
    feat_test,
    z_scores_train,
    feat_cols, mu, sd,
    k_list = (3,5,7,10),
    tau_list = (2.0,3.0,5.0),
    alpha_list = (0.5,1.0,2.0),
    beta_list = (0.5,1.0,2.0),
    x_sd = None, y_sd = None,
    certainty_weights = None # addition for label certainty weighting
):
    seen = set(feat_train["layer_label"])
    test_set = feat_test[feat_test["layer_label"].isin(seen)].copy()
    rows = []

    for k in k_list:
        for tau in tau_list:
            for alpha in alpha_list:
                for beta in beta_list:
                    preds = test_set.apply(
                        lambda r: predict_label_hybrid(
                            r,
                            feat_train = feat_train,
                            z_scores_train = z_scores_train,   # DF or ndarray OK
                            feat_cols = feat_cols, mu = mu, sd = sd,
                            k = k, tau = tau, alpha = alpha, beta = beta,
                            x_sd = x_sd, y_sd = y_sd,
                            certainty_weights = certainty_weights
                        ),
                        axis = 1
                    )
                    acc = (preds ==  test_set["layer_label"]).mean()
                    rows.append(("hybrid_row_based", k, tau, alpha, beta, acc))

    return (pd.DataFrame(rows, columns = ["model","k","tau","alpha","beta","acc"])
              .sort_values("acc", ascending = False)
              .reset_index(drop = True))

x_sd = float(feat_train["x"].std(ddof = 1)) or 1.0
y_sd = float(feat_train["y"].std(ddof = 1)) or 1.0

print("Hybrid row evaluation")
print(eval_hybrid_row_pred(
        feat_train = feat_train,
        feat_test = feat_test,
        z_scores_train = z_scores_train,
        feat_cols = feat_cols, mu = mu, sd = sd,
        k_list = (3,5,7,10),
        tau_list = (1.0,2.0,3.0,5.0),
        alpha_list = (0.0,1.0,2.0),
        beta_list = (0.0,1.0,2.0),
        x_sd = x_sd, y_sd = y_sd, 
        certainty_weights = certainty_weight
    ).to_string(index = False)
)

# evaluation for hybrid-row-based spatial model with k = 5, tau = 3.0
feat_test = feat_test.copy()
feat_test["pred_hybrid"] = feat_test.apply(
    lambda r: predict_label_hybrid(
        r,
        feat_train = feat_train,
        z_scores_train = z_scores_train,
        feat_cols = feat_cols, mu = mu, sd = sd,
        k = 3, tau = 5.0,
        alpha = 1.0, beta = 1.0,
        x_sd = x_sd, y_sd = y_sd,
        certainty_weights = certainty_weight
    ),
    axis = 1
)

# accuracy only labels present in train
mask_seen = feat_test["layer_label"].isin(set(feat_train["layer_label"]))
if mask_seen.any():
    acc = (feat_test.loc[mask_seen, "pred_hybrid"] ==  feat_test.loc[mask_seen, "layer_label"]).mean()
    print(f"Hybrid kNN (k = 5, tau = 2m, alpha = 1, beta = 1) accuracy on seen-label subset: {acc:.3f}")
else:
    print("None of the test labels are present in train.")

# unseen labels
unseen_counts = feat_test.loc[~mask_seen, "layer_label"].value_counts()
if len(unseen_counts):
    print("Test labels unseen in TRAIN (not predictable):")
    print(unseen_counts.to_string())

# per-class accuracy used for model assessment/improvement

per_class_acc = (
    (feat_test.loc[mask_seen, "pred_hybrid"] == feat_test.loc[mask_seen, "layer_label"])
    .groupby(feat_test.loc[mask_seen, "layer_label"]).mean()
    .rename("acc")
)
support = feat_test.loc[mask_seen, "layer_label"].value_counts().rename("n")
print("Per-class (top 10 by support):")
print(pd.concat([per_class_acc, support], axis = 1).sort_values("n", ascending = False).head(10).to_string())

# %%
# used to assess, why in the inital model, change in alpha and beta did not change the acc
# then, spatial scaling was introduced
# r = feat_test.iloc[0]
# x0, y0 = r["x"], r["y"]

# dx = feat_train["x"] - x0
# dy = feat_train["y"] - y0
# d_spatial = np.hypot(dx, dy)

# row_vals = r.reindex(feat_cols).to_numpy(dtype = float)
# row_vec  = (row_vals - mu.to_numpy()) / sd.to_numpy()
# train_mat = ((feat_train[feat_cols] - mu) / sd).to_numpy()
# d_feat = np.linalg.norm(train_mat - row_vec, axis = 1)

# print("mean spatial distance:", np.mean(d_spatial))
# print("mean feature distance:", np.mean(d_feat))
# print("ratio spatial/feature:", np.mean(d_spatial)/np.mean(d_feat))

# res = eval_hybrid_row_pred(
#     feat_train, feat_test, z_scores_train,
#     feat_cols, mu, sd,
#     k_list = (3,5,7,10),
#     tau_list = (1.0,2.0,3.0,5.0),
#     alpha_list = (0.5,1.0,2.0),
#     beta_list = (0.5,1.0,2.0),
#     x_sd = x_sd, y_sd = y_sd
# )
# print("rows:", len(res))
# print(res.groupby(["tau","k"]).size())

# r = feat_test.iloc[0]
# dx = (feat_train["x"].to_numpy(float) - float(r["x"])) / x_sd
# dy = (feat_train["y"].to_numpy(float) - float(r["y"])) / y_sd
# d_spatial_scaled = np.hypot(dx, dy)

# row_vals = r.reindex(feat_cols).to_numpy(float, copy = False)
# row_vec  = (row_vals - mu.to_numpy(float)) / sd.to_numpy(float)
# train_mat = ((feat_train[feat_cols] - mu)/sd).to_numpy(float)

# d_feat = np.linalg.norm(train_mat - row_vec, axis = 1)
# print("mean d_spatial_scaled:", d_spatial_scaled.mean())
# print("mean d_feat:", d_feat.mean())

# neighbours should stay constant for varying params check
# def neighbor_ids(row, alpha, beta, k=5, tau=1.0):
#     x0, y0 = float(row["x"]), float(row["y"])
#     dx = (feat_train["x"].to_numpy(float)-x0)/x_sd
#     dy = (feat_train["y"].to_numpy(float)-y0)/y_sd
#     d_spatial = np.hypot(dx, dy)

#     mean_mtaw_q = float(row["mean_depth_mtaw"])
#     elev_diff = np.abs(feat_train["mean_depth_mtaw"].to_numpy(float) - mean_mtaw_q)
#     mask = elev_diff <= tau
#     if not np.any(mask):
#         k_elev = min(k, len(feat_train))
#         idx = np.argpartition(elev_diff, k_elev-1)[:k_elev]
#         mask = np.zeros(len(feat_train), bool); mask[idx] = True

#     row_vec = ((row.reindex(feat_cols).to_numpy(float) - mu.to_numpy(float)) / sd.to_numpy(float))
#     train_mat = np.asarray(z_scores_train, float)[mask]
#     d_feat = np.linalg.norm(train_mat - row_vec, axis=1)

#     d = np.sqrt(alpha*(d_spatial[mask]**2) + beta*(d_feat**2))
#     k_eff = min(k, len(d))
#     nn_local = np.argpartition(d, k_eff-1)[:k_eff]
#     return np.flatnonzero(mask)[nn_local]

# row0 = feat_test.iloc[0]
# for a,b in [(0.5,0.5),(1,1),(2,2)]:
#     print((a,b), np.sort(neighbor_ids(row0, a, b)))

# # fraction of test rows whose prediction varies across any α/β
# stack = np.column_stack(list(preds.values()))
# varies = (stack != stack[:, [0]]).any(axis=1).mean()
# print("share of rows that change label across α/β:", varies)

# %%

# i got 99 plots

def prepare_df(df):
    plot_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry = gpd.points_from_xy(df["x"], df["y"]),
        crs = "EPSG:31370"
    )
    return plot_gdf.to_crs(epsg = 3857)

def fixed_window_size(bounds, pad_ratio = 0.08, min_span = 1500):
    xmin, ymin, xmax, ymax = bounds # create window using bounds
    dx, dy = xmax - xmin, ymax - ymin #width, height
    # padding
    pad_x, pad_y = dx * pad_ratio, dy * pad_ratio
    xmin_, xmax_ = xmin - pad_x, xmax + pad_x
    ymin_, ymax_ = ymin - pad_y, ymax + pad_y
    # span
    span_x = max(xmax_ - xmin_, min_span)
    span_y = max(ymax_ - ymin_, min_span)
    cx0, cy0 = (xmin_ + xmax_) / 2, (ymin_ + ymax_) / 2
    return (cx0 - span_x/2, cy0 - span_y/2, cx0 + span_x/2, cy0 + span_y/2)

def colourscale(gdf, feature, vmin = None, vmax = None):
    if vmin is None:
        vmin = gdf[feature].min() # use global minimum
    if vmax is None:
        vmax = gdf[feature].max() # use the global maximum
    return vmin, vmax 

def plot_feature_map(
    df,
    feature,
    groupby = None, # grouping column
    feature_label = None, # label for the colourbar
    cmap = "viridis", # default colour scheme (intended for scales)
    markersize = 50,
    order_desc = True, # default: groups with most data points first
    pad_ratio = 0.08,
    min_span = 1500,
    figsize = (9, 6), # in inches
):
    
    g = prepare_df(df)

    window_size = fixed_window_size(g.total_bounds, pad_ratio, min_span)

    vmin, vmax = colourscale(g, feature)

    if groupby:
        counts = g[groupby].value_counts()
        if order_desc:
            keys = counts.index.tolist() # most data to fewest data
        else:
            keys = counts.sort_values(ascending = True).index.tolist() # fewest to most
    else:
        keys = [None] 

    for k in keys:
        # use subset for grouping, loop through groups
        sub = g if k is None else g[g[groupby] == k]
        if sub.empty:
            continue

        title = feature_label

        if groupby and k is not None:
            side_label = f"{feature}  |  {groupby} = {k} (n = {len(sub)})"
        else:
            side_label = feature

        fig, ax = plt.subplots(figsize = figsize)
        sub.plot(
            column = feature,
            cmap = cmap,
            legend = True,
            ax = ax,
            markersize = markersize,
            alpha = 0.95, edgecolor = "k", linewidth = 0.25,
            vmin = vmin, vmax = vmax,
            legend_kwds = {"label": side_label}
        )

        # fix window size
        ax.set_xlim(window_size[0], window_size[2])
        ax.set_ylim(window_size[1], window_size[3])

        # add basemap
        cx.add_basemap(
            ax, source = cx.providers.CartoDB.Positron,
            crs = g.crs, alpha = 1.0
        )

        # fix format, show plot
        ax.set_title(title)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()

cpt_idx = feat_all.groupby("sondeernummer")["start_depth_mtaw"].idxmin()
cpt_df  = feat_all.loc[cpt_idx].copy()
layers_df = feat_all.drop_duplicates(["sondeernummer", "layer_label"]).copy()

cpt_df["start_depth_mtaw"].hist(bins=30)
plt.title("Starting elevation (mtaw) – CPT-level")
plt.xlabel("mtaw")
plt.ylabel("count")
plt.show()

plot_feature_map(
    cpt_df,
    feature = "start_depth_mtaw",
    feature_label = "Starting elevation (mtaw)"
)

plot_feature_map(
    layers_df,
    feature = "thickness",
    groupby = "layer_label",
    feature_label = "Layer thickness (m)"
)

plot_feature_map(
    layers_df,
    feature = "qc_mean",
    groupby = "layer_label",
    feature_label = "Cone resistance qc (MPa)"
)

layer_count = (
    feat_all.groupby("sondeernummer")["layer_label"]
            .nunique()
            .reset_index(name="n_layers")
)

# merge the count onto CPT points (one point per CPT)
cpt_points = cpt_df[["sondeernummer", "x", "y"]].copy()
df_layers = cpt_points.merge(layer_count, on="sondeernummer", how="left")

plot_feature_map(
    df_layers,
    feature = "n_layers",
    feature_label = "Distinct layers per CPT (count)"
)

# %%
# homogeneity per layer across all train CPTs
# def layer_homogeneity_table(feat_train, layer_col="layer_label"):
#     agg = (
#         feat_train.groupby(layer_col).agg(
#             qc_mean_std=("qc_mean", "std"),
#             fs_mean_std=("fs_mean", "std"),
#             qc_d_dz_mean_std=("qc_d_dz_mean", "std"),
#             fs_d_dz_mean_std=("fs_d_dz_mean", "std"),
#             thickness_std=("thickness", "std"),
#             mean_depth_mtaw_std=("mean_depth_mtaw", "std"),
#             n_layers=("sondeernummer", "size"),
#             n_cpts=("sondeernummer", "nunique"),
#         )
#         .fillna(0.0)
#     )

#     # average of z-score sds
#     std_cols = [
#         "qc_mean_std", "fs_mean_std",
#         "qc_d_dz_mean_std", "fs_d_dz_mean_std",
#         "thickness_std", "mean_depth_mtaw_std"
#     ]
#     Z = (agg[std_cols] - agg[std_cols].mean()) / agg[std_cols].std(ddof=1).replace(0, 1)
#     agg["homogeneity_score"] = Z.mean(axis=1)

#     return agg.sort_values("homogeneity_score")

# homog_table = layer_homogeneity_table(feat_train)
# print(homog_table.round(3).to_string())
