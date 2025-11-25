import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import random

#############################
# Global defaults
#############################
seed = 22 #current date
val_fraction: float = 0.3
segments_oi = [  # example segments of interest (same as the ones from the company's excel sheet)
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

#############################
# Class definition
#############################

class DataSet():

    ##############################
    # 1st step: load data
    ##############################

    def __init__(self, path_to_parquet, segments_of_interest):
        """Takes path to data, segments of interest to filter on"""
        self.raw_df = pd.read_parquet(path_to_parquet)
        self.known_data = self.raw_df[self.raw_df["lithostrat_id"].isin(segments_of_interest)]
        self.unlabeled_data = self.raw_df[~self.raw_df["lithostrat_id"].isin(segments_of_interest)]

    ##############################
    # 2nd step: impute missing values
    ##############################

    def impute_params(
            self,
            overwrite: bool = False,
            *,
            scope: str = "all",
            use_imputation: bool = True,
            drop_na_if_not_imputed: bool = True,
            na_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Default pre-processing for data:
        - Use_imputation = True (Default): Impute missing values according to global logic
        in impute_params function.
        - Drop_na_if_not_imputed = True (Default): Drop na in the specified columns when
        these are not imputed with use_imputation.
        - If the above are overwritten, return the df depending on the scope (full dataset),
        dataset with labeled CPTs only, dataset with unlabeled CPTs only.

        We assume no duplicates in index, sondeernummer, coordinates after discussion with VITO.
        """
        if scope == "known":
            base = self.known_data
        elif scope == "all":
            base = self.raw_df
        elif scope == "unlabeled":
            base = self.unlabeled
        df = base.copy()
        if use_imputation:
            return impute_params(df, overwrite = overwrite)
        if drop_na_if_not_imputed and na_cols is not None:
            return df.dropna(subset = list(na_cols))
        return df
    
    ##############################
    # optional: simple data split: random by groups (sondering_id)
    ##############################
    
    def simple_split(
        self,
        val_frac: float = val_fraction,
        group_col: str = "sondering_id",
        random_state: int | None = seed,
        *,
        use_imputation: bool = True,
        drop_na_if_not_imputed: bool = True,
        na_cols: list[str] | None = None,
    ):
        """
        Group-based train/validation split.
        - Uses imputation as default.
        - Groups by group_col (here: sondering_id).
        - Randomly selects groups for the validation set so that
          val_frac (here: 0.2) of all rows end up in validation.
        - Returns: train_df, val_df
        """
        df = self.impute_params(
            overwrite = False,
            use_imputation = use_imputation,
            drop_na_if_not_imputed = drop_na_if_not_imputed,
            na_cols = na_cols,
        )

        return simple_split(
            df,
            group_col = group_col,
            val_frac = val_frac,
            random_state = random_state,
        )
    
    ##############################
    # model assessment by LOOCV 
    ##############################
    def loo_cv(
            self,
            df_with_tiles: pd.DataFrame,
            *,
            label_col: str = "lithostrat_id",
            cpt_col: str = "sondeernummer",
            extra_id_cols: list[str] | None = None,
    ):
        """
        """
        return loo_cv(
            df_with_tiles, 
            label_col = label_col, 
            cpt_col = cpt_col, 
            extra_id_cols = extra_id_cols,
            )

##################################
# Global functions
##################################

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

def extract_features(
    df: pd.DataFrame,
    vars_num: list[str] = default_num_vars,
    depth_col: str = "diepte",
    depth_mtaw_col: str = "diepte_mtaw",
    label_col: str = "lithostrat_id",
    cpt_col: str = "sondeernummer",
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Feature extractor: one row per (CPT, layer_label).
    Groups by (cpt_col, label_col) and computes:
      - IDs & position: cpt_col, x, y, start/end depth, thickness.
      - For each numeric variable in vars_num:
        mean, sd, squared mean, end-start, end-start per meter,
        depth-derivative stats, depth of top-3 values.
    """
    rows = []
    skipped = 0
    skipped_examples = [] # for layers with too few samples to compute features

    for (cpt, label), g in df.groupby([cpt_col, label_col], observed=False):
        g = g.sort_values(depth_col)
        if len(g) < min_n:
            skipped += 1
            skipped_examples.append((cpt, label, len(g)))
            continue

        # positional
        start_depth = float(g[depth_col].min())
        end_depth = float(g[depth_col].max())
        start_depth_mtaw = float(g[depth_mtaw_col].max())
        end_depth_mtaw = float(g[depth_mtaw_col].min())

        row = {
            cpt_col: cpt,
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

        # numeric summaries
        for var in vars_num:
            if var not in g.columns:
            # included na value handling; not important for initial data set
            # but maybe later when new CPTs are uploaded
                row[f"{var}_mean"] = np.nan
                row[f"{var}_sd"] = np.nan
                row[f"{var}_sq_mean"] = np.nan
                row[f"{var}_end_minus_start"] = np.nan
                row[f"{var}_end_minus_start_per_m"] = np.nan
                row[f"{var}_d_dz_mean"] = np.nan
                row[f"{var}_d_dz_sd"] = np.nan
                row[f"{var}_top3_mean_depth_rel"] = np.nan
                row[f"{var}_top3_mean_depth_mtaw"] = np.nan
                continue

            s = g[var].dropna()
            # mean
            row[f"{var}_mean"] = float(s.mean()) if len(s) else np.nan
            # standard deviation
            row[f"{var}_sd"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
            # mean of squared values
            s2 = s ** 2
            row[f"{var}_sq_mean"] = float(s2.mean()) if len(s2) else np.nan

            # local depth behaviour
            sub = g[[depth_col, var]].dropna().sort_values(depth_col)

            if len(sub) > 1:
                # v is used for variable values
                v = sub[var].to_numpy(dtype=float)
                # z is used for depth values
                z = sub[depth_col].to_numpy(dtype=float)
                # delta from end to start of a layer -> net change
                delta_layer = float(v[-1] - v[0])
                thickness = float(z[-1] - z[0])
                row[f"{var}_end_minus_start"] = delta_layer
                # normalized change: net variable change / thickness
                # -> change in value per meter 
                row[f"{var}_end_minus_start_per_m"] = (
                    delta_layer / thickness if thickness > 0 else np.nan
                )

                # local changes: derivatives
                dv = np.diff(v)
                dz = np.diff(z)
                with np.errstate(divide="ignore", invalid="ignore"):
                    deriv = dv / dz
                deriv = deriv[np.isfinite(deriv)]
                if deriv.size:
                    # mean and standard deviation of slopes (per group)
                    row[f"{var}_d_dz_mean"] = float(deriv.mean())
                    row[f"{var}_d_dz_sd"] = (
                        float(deriv.std(ddof=1)) if deriv.size > 1 else 0.0
                    )
                else:
                    row[f"{var}_d_dz_mean"] = np.nan
                    row[f"{var}_d_dz_sd"] = np.nan
            else:
                row[f"{var}_end_minus_start"] = np.nan
                row[f"{var}_end_minus_start_per_m"] = np.nan
                row[f"{var}_d_dz_mean"] = np.nan
                row[f"{var}_d_dz_sd"] = np.nan

            # mean depth of the 3 largest values
            if len(s) >= 3:
                top3_idx = s.nlargest(3).index
                row[f"{var}_top3_mean_depth_rel"] = float(
                    g.loc[top3_idx, depth_col].mean()
                )
                row[f"{var}_top3_mean_depth_mtaw"] = float(
                    g.loc[top3_idx, depth_mtaw_col].mean()
                )
            else:
                row[f"{var}_top3_mean_depth_rel"] = np.nan
                row[f"{var}_top3_mean_depth_mtaw"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if skipped > 0:
        print(
            f"Feature extractor skipped {skipped} (CPT, layer) group(s) "
            f"with fewer than {min_n} samples."
        )

    id_cols = [
        cpt_col,
        "layer_label",
        "x",
        "y",
        "start_depth",
        "end_depth",
        "start_depth_mtaw",
        "end_depth_mtaw",
        "thickness",
        "mean_depth_mtaw",
        "n_samples_used",
    ]
    feature_cols = [c for c in out.columns if c not in id_cols]
    return out[id_cols + feature_cols]


def simple_split(
    df: pd.DataFrame,
    group_col: str = "sondering_id",
    val_frac: float = val_fraction,
    random_state: int | None = seed,
):
    """
    Group-based train/validation split.
    - Uses imputation as default.
    - Groups by group_col (here: sondering_id).
    - Randomly selects groups for the validation set so that
        val_frac (here: 0.2) of all rows end up in validation.
    - Returns: train_df, val_df
    """
    if random_state is None:
        random_state = seed

    rng = np.random.default_rng(random_state)
    groups = df[group_col].dropna().unique()
    rng.shuffle(groups)

    n_val_groups = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val_groups])

    val_df = df[df[group_col].isin(val_groups)].copy()
    train_df = df[~df[group_col].isin(val_groups)].copy()

    return train_df, val_df


def tile_split(
    df: pd.DataFrame,
    Gx: int = 4,
    Gy: int = 4,
    train_frac: float = val_fraction,
    random_state: int | None = seed,
    *,
    x_col: str = "x",
    y_col: str = "y",
    cpt_col: str = "sondeernummer",
    label_col: str = "lithostrat_id",
    extra_id_cols = None,
):
    """
    Tile-based spatial split.
    Assumes df has columns:
      - x_col, y_col: coordinates
      - cpt_col: CPT sondeernummer or sondering ID
      - label_col: target label
    Returns:
      df_with_tiles, train_df, test_df, X_train, X_test, y_train, y_test
    """

    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_state)

    feat = df.copy()

    # build grid
    feat["xbin"] = pd.qcut(feat[x_col], q = Gx, labels = False, duplicates="drop").astype(int)
    feat["ybin"] = pd.qcut(feat[y_col], q = Gy, labels = False, duplicates="drop").astype(int)
    feat["tile"] = feat["xbin"] * Gy + feat["ybin"]

    # tile stats (you can later add a verbose flag if you want less output)
    tile_sizes = feat.groupby("tile").size().sort_values(ascending=False)
    print("Tiles and CPT counts per tile: ", tile_sizes.to_string())
    n_tiles = feat["tile"].nunique()
    print(f"Total number of unique tiles: {n_tiles}")

    # choose train tiles
    all_tiles = feat["tile"].unique()
    train_tile_count = max(1, int(np.ceil(n_tiles * train_frac)))
    train_tiles = set(rng.choice(all_tiles, size = train_tile_count, replace = False))
    test_tiles = set(all_tiles) - train_tiles

    print(f"Training tiles ({len(train_tiles)}): {sorted(train_tiles)}")
    print(f"Test tiles ({len(test_tiles)}): {sorted(test_tiles)}")

    # match tiles -> CPTs -> rows
    train_cpts = set(feat.loc[feat["tile"].isin(train_tiles), cpt_col].unique())
    test_cpts = set(feat.loc[feat["tile"].isin(test_tiles), cpt_col].unique())

    train_df = feat[feat[cpt_col].isin(train_cpts)].copy()
    test_df = feat[feat[cpt_col].isin(test_cpts)].copy()

    # build id_cols depending on what df actually has
    base_id_cols = [cpt_col, label_col, x_col, y_col, "xbin", "ybin", "tile"]
    if extra_id_cols is None:
        extra_id_cols = []
    id_cols = [c for c in base_id_cols + list(extra_id_cols) if c in feat.columns]

    X_train = train_df.drop(columns=id_cols)
    y_train = train_df[label_col].copy()
    X_test = test_df.drop(columns=id_cols)
    y_test = test_df[label_col].copy()

    # label coverage
    seen_labels = set(y_train.unique())
    mask_seen = test_df[label_col].isin(seen_labels)
    print(
        f"Test rows with labels seen in train: "
        f"{mask_seen.sum()} / {len(test_df)} ({mask_seen.mean():.1%})"
    )

    return feat, train_df, test_df, X_train, X_test, y_train, y_test


def loo_cv(
    df_with_tiles: pd.DataFrame,
    *,
    label_col: str = "layer_label",
    cpt_col: str = "sondeernummer",
    extra_id_cols = None,
):
    """
    Basis for leave-one-tile-out cross-validation.
    Assumes df_with_tiles (output of tile_split)
    - 'tile' column
    - coordinate columns x, y
    - label column (lithostrat_id)
    - optional extra columns depending on the model
    Yields for each tile t:
    - t, train_df, test_df, X_train, X_test, y_train, y_test
    """
    if "tile" not in df_with_tiles.columns:
        raise ValueError("df_with_tiles must contain a 'tile' column.")

    # base ID/meta columns we don't want as features
    base_id_cols = [cpt_col, label_col, "x", "y", "xbin", "ybin", "tile"]
    if extra_id_cols is None:
        extra_id_cols = []
    all_id_cols = [
        c for c in base_id_cols + list(extra_id_cols)
        if c in df_with_tiles.columns
    ]

    tiles = sorted(df_with_tiles["tile"].unique())

    for t in tiles:
        test_df = df_with_tiles[df_with_tiles["tile"] == t].copy()
        train_df = df_with_tiles[df_with_tiles["tile"] != t].copy()

        X_train = train_df.drop(columns=all_id_cols)
        y_train = train_df[label_col].copy()
        X_test = test_df.drop(columns=all_id_cols)
        y_test = test_df[label_col].copy()

        yield t, train_df, test_df, X_train, X_test, y_train, y_test

def print_loocv(
    loocv_df: pd.DataFrame,
    *,
    acc_col: str = "acc",
    model_name: str | None = None,
) -> None:
    """
    Unified print statements for LOOCV results.
    - Loocv_df : DataFrame
    - Expected columns: 'tile', 'n_train', 'n_test', and an accuracy column.
    acc_col : str: Name of column with accuracy values (Default = 'acc')
    - Model_name : str or None (e. g. Hybrid KNN).
    """
    if loocv_df.empty:
        header = f"LOOCV results{f' for {model_name}' if model_name else ''}:"
        print(header, "no tiles / empty result.")
        return

    header = f"LOOCV results for {model_name} (per tile):" if model_name else "LOOCV results (per tile):"
    print("\n" + header)

    cols_to_show = ["tile", "n_train", "n_test"]
    if acc_col in loocv_df.columns:
        cols_to_show.append(acc_col)

    print(loocv_df[cols_to_show].to_string(index=False))

    if acc_col in loocv_df.columns and loocv_df[acc_col].notna().any():
        mean_acc = float(loocv_df[acc_col].mean())
        print(f"\nMean LOOCV {acc_col}: {mean_acc:.3f}")
