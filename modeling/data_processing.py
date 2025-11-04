

"""
PDS Model — Python conversion focused on preprocessing only:
- Parameters (EXTRACT_TREND, BIN_W, SET_SEED, size_tune, TH_QC20, TH_QC40, EXTRACT_TREND_TYPE)
- Utilities: agg_features, extract_trend (with decomp_type), freq_from_depth, group_strat_split
- Load & clean parquet
- Trend extraction per sondering_id
- Feature engineering (bin + whole, QC spikes)
- Train/test split of sondering_id (stratified by dominant class)
- Output: data modeling_features_df.csv
"""

from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

# Parameters  used 

# python modeling/data_processing.py --extract_trend True --bin_w 0.6 --seed 42 --trend_type additive --data_folder data --results_folder results
# extract trend from signals
EXTRACT_TREND = True

# binning width in meters
BIN_W = 0.6
# random seed for reproducibility
SET_SEED = 42
#size_tune = 10
# QC picks thresholds
TH_QC20 = 20
TH_QC40 = 40

EXTRACT_TREND_TYPE = "additive"

print("Parameters:")
print("  EXTRACT_TREND:", EXTRACT_TREND)
print("  BIN_W:", BIN_W)
print("  SET_SEED:", SET_SEED)
#print("  size_tune:", size_tune)
print("  EXTRACT_TREND_TYPE:", EXTRACT_TREND_TYPE)

# Agregate function
def agg_features(df: pd.DataFrame, feat_cols, by_cols, stats, suffix=""):
    """
    Aggregate features by  binned groups with multiple statistics.
    :param df: input dataframe
    :param feat_cols: list of feature columns to aggregate
    :param by_cols: list of columns to group by
    :param stats: dict of {stat_name: function} to apply
    :param suffix: suffix to append to feature names
    :return: aggregated dataframe
    """
    if isinstance(by_cols, str):
        by_cols = [by_cols]
    grouped = df.groupby(by_cols, dropna=False, observed=False)
    out_frames = []
    for stat_name, fn in stats.items():
        tmp = grouped[feat_cols].agg(lambda s: fn(s.to_numpy(dtype=float)))
        tmp = tmp.rename(columns={v: f"{v}{suffix}_{stat_name}" for v in feat_cols})
        out_frames.append(tmp)
    res = pd.concat(out_frames, axis=1).reset_index()
    return res


def extract_trend(x, freq=5, decomp_type="additive"):
    """
    Decompose a 1D series and return the trend component.
    """
    x = pd.Series(pd.to_numeric(x, errors="coerce"))
    if not np.isfinite(freq) or freq < 2 or x.dropna().shape[0] < 2 or len(x) < 2 * int(freq):
        return x.values

    freq = int(freq)

    try:
        res = seasonal_decompose(
            x, model="multiplicative" if decomp_type == "multiplicative" else "additive",
            period=freq, extrapolate_trend=freq
        )
        trend = res.trend
        trend = trend.where(~trend.isna(), x)
        return trend.to_numpy()
    except Exception:
        # Fallback to rolling mean if decomposition fails
        tr = x.rolling(window=freq, center=True, min_periods=max(1, freq // 2)).mean()
        tr = tr.where(~tr.isna(), x)
        return tr.values


def freq_from_depth(z, default=25):
    """
    Estimate sampling frequency from depth measurements for time series decomposition.
    
    Calculates the number of observations per unit depth by taking the inverse of the 
    median depth step between consecutive measurements. This frequency is used in 
    seasonal decomposition to identify the period of repeating patterns in CPT signals.

    If depth measurements are taken every 0.02m on average, the frequency
    would be 1/0.02 = 50 observations per meter, meaning the decomposition should look
    for patterns that repeat every ~50 data points.
    
    Args:
        z: Array-like of depth values (in meters)
        default: Fallback frequency if calculation fails (default 25)
    
    Returns:
        int: Estimated number of observations per unit depth
    """
    z = pd.Series(z).dropna().sort_values()
    if len(z) < 2:
        return int(default)
    dz = np.median(np.diff(np.array(z.values)))
    if not np.isfinite(dz) or dz <= 0:
        return int(default)
    return int(max(1, round(1.0 / dz)))




def group_strat_split(dt: pd.DataFrame,
                      id_col="sondering_id",
                      y_col="lithostrat_id",
                      prop=0.7,
                      tol=0.05,
                      max_tries=200,
                      seed=SET_SEED):
    """
    Split borehole IDs into train/test sets while maintaining geological layer balance.
    
    Each borehole is labeled by its most common layer type, then IDs are split so both 
    train and test sets preserve the original proportions of each layer (within tolerance).
    Tries multiple random seeds until all layer types are within the specified tolerance
    of the target proportion, ensuring the test set accurately represents real-world lithology.
    
    Args:
        dt: DataFrame with borehole measurements
        id_col: Column name for borehole identifiers
        y_col: Column name for geological layer labels
        prop: Target proportion for training set (default 0.7)
        tol: Acceptable deviation from target proportion (default 0.05)
        max_tries: Maximum split attempts before returning best effort
        seed: Base random seed for reproducibility
    
    Returns:
        dict with train_ids, test_ids arrays and the seed used
    """
    # Calculate mode class for each ID - using agg with lambda to avoid deprecation warning
    ids_lab = (dt.groupby(id_col, dropna=False)[y_col]
                 .agg(lambda x: Counter(x.dropna()).most_common(1)[0][0] if len(x.dropna()) > 0 else np.nan)
                 .reset_index(name="mode_class"))
    full_counts = dt[y_col].value_counts().rename_axis(y_col).reset_index(name="N")

    ids = ids_lab.dropna(subset=["mode_class"])
    
    # Check class distribution to ensure stratified split is possible
    class_counts = ids["mode_class"].value_counts()
    min_class_count = class_counts.min()
    
    # If any class has fewer than 2 samples, we cannot do stratified splitting
    # In that case, filter out classes with only 1 sample or fall back to non-stratified split
    if min_class_count < 2:
        print(f"Warning: Some classes have only {min_class_count} sample(s). Filtering out rare classes for stratified split.")
        valid_classes = class_counts[class_counts >= 2].index
        ids = ids[ids["mode_class"].isin(valid_classes)].copy()
        
        # If after filtering we have too few samples, fall back to non-stratified split
        if len(ids) < 2:
            print("Warning: Too few samples after filtering. Using non-stratified split.")
            all_ids = ids_lab[id_col].dropna().values
            tr_ids, te_ids = train_test_split(
                all_ids,
                train_size=prop,
                random_state=seed
            )
            return {"train_ids": tr_ids, "test_ids": te_ids, "seed": seed}
    
    last_tr, last_te, last_seed = None, None, None

    for s in range(max_tries):
        rs = seed + s
        tr_ids, te_ids = train_test_split(
            ids[id_col].values,
            train_size=prop,
            random_state=rs,
            stratify=np.array(ids["mode_class"].values)
        )
        tr_dt = dt[dt[id_col].isin(tr_ids)]
        tr_counts = tr_dt[y_col].value_counts().rename_axis(y_col).reset_index(name="N_tr")
        merged = full_counts.merge(tr_counts, on=y_col, how="left").fillna({"N_tr": 0})
        merged["pct"] = merged["N_tr"] / merged["N"]
        if np.all(np.abs(merged["pct"] - prop) <= tol):
            return {"train_ids": tr_ids, "test_ids": te_ids, "seed": rs}
        last_tr, last_te, last_seed = tr_ids, te_ids, rs

    return {"train_ids": last_tr, "test_ids": last_te, "seed": last_seed}



# expects these to be defined elsewhere in your module:
# - agg_features(df, feat_cols, by_cols, stats, suffix="")
# - group_strat_split(dt, id_col="sondering_id", y_col="lithostrat_id", prop=0.7, tol=0.05, max_tries=200, seed=42)
# - freq_from_depth(z, default=25)
# - extract_trend(x, freq=5, decomp_type="additive")
# - TH_QC20, TH_QC40

def process_cpt_data(
    data_folder: Path,
    results_folder: Path,
    parquet_filename: str = "vw_cpt_brussels_params_completeset_20250318_remapped.parquet",
    do_extract_trend: bool = EXTRACT_TREND,
    bin_w: float = BIN_W,
    seed: int = SET_SEED,
    trend_type: str = "additive"
):
    """
    Complete CPT preprocessing pipeline (preprocessing + features + split only).

    Returns:
        dict(features=DataFrame, train_ids=list, test_ids=list, split_seed=int)
    """
    # checks
    if trend_type not in {"additive", "multiplicative"}:
        raise ValueError(f"trend_type must be 'additive' or 'multiplicative', got {trend_type!r}")
    results_folder.mkdir(parents=True, exist_ok=True)

    parquet_path = data_folder / parquet_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # load data
    cpt_df = pd.read_parquet(parquet_path, engine="fastparquet")

    # required columns used below
    required = {"sondering_id", "lithostrat_id", "diepte"}
    missing = required - set(cpt_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # filter / clean
    cpt_df = cpt_df[~cpt_df["lithostrat_id"].isna()].copy()

    # rare litho removal

    litho_counts = (
        cpt_df.drop_duplicates(subset=["sondering_id", "lithostrat_id"])
              .groupby("lithostrat_id", dropna=False)
              .size()
              .reset_index(name="N")
    )
    rare_litho = set(litho_counts.loc[litho_counts["N"] < 5, "lithostrat_id"])
    if rare_litho:
        cpt_df = cpt_df[~cpt_df["lithostrat_id"].isin(rare_litho)].copy()

    # Find Unique litho per sondering for stratified split
    cpt_unique = cpt_df.drop_duplicates(subset=["sondering_id", "lithostrat_id"]).copy()
    split_res = group_strat_split(cpt_unique, prop=0.7, tol=0.05, seed=seed)
    train_ids = list(split_res["train_ids"])
    test_ids  = list(split_res["test_ids"])

    print("Train IDs:", len(train_ids))
    print("Test   IDs:", len(test_ids))

    # optional trend extraction
    if do_extract_trend:
        feat_cols_trend = [c for c in ["qc", "fs", "rf", "qtn", "fr"] if c in cpt_df.columns]
        if feat_cols_trend:
            cpt_df = cpt_df.sort_values(["sondering_id", "diepte"]).copy()

            def _trend_and_fill(group: pd.DataFrame) -> pd.DataFrame:
                # Ensure the group identifier column exists even when include_groups=False
                if "sondering_id" not in group.columns:
                    group = group.copy()
                    group["sondering_id"] = group.name

                freq = freq_from_depth(group["diepte"].values)
                for col in feat_cols_trend:
                    vals = extract_trend(group[col].values, freq=freq, decomp_type=trend_type)
                    # Convert to series first for proper handling
                    v = pd.Series(pd.to_numeric(vals, errors="coerce"))
                    v = v.replace([np.inf, -np.inf], np.nan)
                    # mirror R: forward once, backward twice
                    v = v.ffill().bfill().bfill().to_numpy()
                    group[col] = v
                return group

            cpt_df = (cpt_df.groupby("sondering_id", group_keys=False, observed=False)
                           .apply(_trend_and_fill))

    #feature engineering
    id_col = "sondering_id"
    depth_col = "diepte"
    feat_cols = [c for c in ["qc", "fs", "rf", "qtn", "fr"] if c in cpt_df.columns]
    geog_cols = [c for c in ["diepte", "diepte_mtaw"] if c in cpt_df.columns]

    # np stats with na handling
    def mean_na(x): 
        return float(np.nanmean(x))
    def sd_na(x):     
        return float(np.nanstd(x, ddof=1))
    def iqr_na(x):   
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
    def median_na(x):
        return float(np.nanmedian(x))
    def mad_na(x):
        med = np.nanmedian(x)
        return float(np.nanmedian(np.abs(x - med)))
    def q10(x): 
        return float(np.nanpercentile(x, 10))
    def q50(x): 
        return float(np.nanpercentile(x, 50))
    def q90(x): 
        return float(np.nanpercentile(x, 90))
    def cv(x):
        m = np.nanmean(x)
        s = np.nanstd(x, ddof=1)
        return float(np.nan) if (not np.isfinite(m) or m == 0) else float(s / m)

    stats = {
        "sd": sd_na, "mean": mean_na,
        "iqr": iqr_na, "median": median_na, "mad": mad_na,
        "q10": q10, "q50": q50, "q90": q90, "cv": cv
    }

    # order + bin
    cpt_df = cpt_df.sort_values([id_col, depth_col]).copy()
    max_depth = float(cpt_df[depth_col].max())
    # BUGFIX: use the function argument `bin_w`, not a global BIN_W
    bins = np.arange(0, max_depth + bin_w, bin_w) if bin_w > 0 else np.array([0, max_depth or 1.0])
    if len(bins) < 2:
        bins = np.array([0, max_depth + (bin_w if bin_w > 0 else 1.0)])
    cpt_df["depth_bin"] = pd.cut(cpt_df["diepte"], bins=bins, include_lowest=True, ordered=True)
    # save cpt_df with cols sondering_id, diepte, depth_bin, lithostrat_id
   
    # unique lithostrat per bin
    litho_dept = (
        cpt_df.loc[:, ["sondering_id", "lithostrat_id", "depth_bin"]]
              .drop_duplicates(subset=["sondering_id", "depth_bin"])
              .copy()
    )
    

    # aggregations
    summaries_bin = agg_features(cpt_df, feat_cols, [id_col, "depth_bin"], stats) if feat_cols else pd.DataFrame()
    summaries_whole = agg_features(cpt_df, geog_cols, [id_col], stats, suffix="_whole") if geog_cols else pd.DataFrame()

    # merge (careful with empty frames)
    if summaries_bin.empty and not summaries_whole.empty:
        dt = summaries_whole.copy()
    elif summaries_whole.empty and not summaries_bin.empty:
        dt = summaries_bin.copy()
    else:
        dt = summaries_bin.merge(summaries_whole, on=id_col, how="left")

    dt = dt.merge(
        litho_dept,
        left_on=[id_col, "depth_bin"],
        right_on=["sondering_id", "depth_bin"],
        how="left"
    )
    ## remove missing lithostrat_id
    dt = dt[~dt["lithostrat_id"].isna()].copy()

    # QC spikes
    if "qc" in cpt_df.columns and not cpt_df["qc"].isna().all():
        qc_ok = cpt_df.dropna(subset=["qc"]).copy()
        qc_spikes = (
            qc_ok.groupby([id_col, "depth_bin"], dropna=False, observed=False)["qc"]
                 .agg(
                     qc_frac_gt20=lambda s: np.mean(s.to_numpy(dtype=float) > TH_QC20),
                     qc_frac_gt40=lambda s: np.mean(s.to_numpy(dtype=float) > TH_QC40),
                     qc_count_gt20=lambda s: np.sum(s.to_numpy(dtype=float) > TH_QC20),
                     qc_count_gt40=lambda s: np.sum(s.to_numpy(dtype=float) > TH_QC40),
                     qc_p99=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 99)),
                 )
                 .reset_index()
        )
        dt = dt.merge(qc_spikes, on=[id_col, "depth_bin"], how="left")

    # Create descriptive filename suffix with parameters
    extract_str = "true" if extract_trend else "false"
    suffix = f"{extract_str}_{bin_w}_{seed}_{trend_type}"
    
    # write out all features
    out_path = results_folder / f"cpt_features_{suffix}.csv"
    dt.to_csv(out_path, index=False)
    print(f"Wrote all features: {out_path}")
    
    # write training features
    train_data = dt[dt[id_col].isin(train_ids)].copy()
    out_path_train = results_folder / f"train_binned_{suffix}.csv"
    train_data.to_csv(out_path_train, index=False)
    print(f"Wrote training data: {out_path_train}")
    
    # write test features
    test_data = dt[dt[id_col].isin(test_ids)].copy()
    out_path_test = results_folder / f"test_binned_{suffix}.csv"
    test_data.to_csv(out_path_test, index=False)
    print(f"Wrote test data: {out_path_test}")

    # save cpt_df with cols sondering_id, diepte, depth_bin, lithostrat_id
    cpt_df[["sondering_id", "diepte", "depth_bin", "lithostrat_id"]].to_csv(
        results_folder / f"cpt_ids_{suffix}.csv", index=False
    )



    return {
        "features": dt,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "split_seed": split_res["seed"],
    }


def main():
    """
    Command-line entry point with argument parsing.
    
    Usage:
        python data_processing.py
        python data_processing.py --extract_trend False --bin_w 0.5 --seed 123
        python data_processing.py --trend_type multiplicative
    """
    parser = argparse.ArgumentParser(
        description="CPT data preprocessing pipeline for geological feature extraction"
    )
    
    parser.add_argument(
        "--extract_trend",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=EXTRACT_TREND,
        help=f"Extract trend from signals (default: {EXTRACT_TREND})"
    )
    
    parser.add_argument(
        "--bin_w",
        type=float,
        default=BIN_W,
        help=f"Binning width in meters (default: {BIN_W})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SET_SEED,
        help=f"Random seed for reproducibility (default: {SET_SEED})"
    )
    
    parser.add_argument(
        "--trend_type",
        type=str,
        choices=["additive", "multiplicative"],
        default=EXTRACT_TREND_TYPE,
        help=f"Trend decomposition type (default: {EXTRACT_TREND_TYPE})"
    )
    
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Path to data folder (default: data)"
    )
    
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Path to results folder (default: results)"
    )
    
    args = parser.parse_args()
    
    # Print parameters being used
    print("Running with parameters:")
    print(f"  EXTRACT_TREND: {args.extract_trend}")
    print(f"  BIN_W: {args.bin_w}")
    print(f"  SET_SEED: {args.seed}")
    print(f"  EXTRACT_TREND_TYPE: {args.trend_type}")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Results folder: {args.results_folder}")
    print()

    result = process_cpt_data(
        data_folder=Path(args.data_folder),
        results_folder=Path(args.results_folder),
        do_extract_trend=args.extract_trend,
        bin_w=args.bin_w,
        seed=args.seed,
        trend_type=args.trend_type
    )
    
    print("\nSplit Summary:")
    print("Train IDs:", len(result["train_ids"]))
    print("Test  IDs:", len(result["test_ids"]))
    print("Seed used:", result["split_seed"])


if __name__ == "__main__":
    main()
