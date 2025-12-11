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
import json
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
SET_SEED = 100
#size_tune = 10
# QC picks thresholds
TH_QC20 = 20
TH_QC40 = 40

EXTRACT_TREND_TYPE = "multiplicative"

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

def process_test_train(
    cpt_df: pd.DataFrame,
    sondering_ids: list,
    do_extract_trend: bool = EXTRACT_TREND,
    bin_w: float = BIN_W,
    trend_type: str = "additive"
) -> pd.DataFrame:
    """
    Process a subset of CPT data (e.g., train or test set) defined by sondering_ids.
    Performs binning, optional trend extraction, feature aggregation, and QC spike calculation.
    
    Args:
        cpt_df: The complete or subset dataframe containing CPT measurements.
        sondering_ids: List of sondering_ids to process.
        do_extract_trend: Whether to perform trend extraction.
        bin_w: Bin width in meters.
        trend_type: Type of trend decomposition ('additive' or 'multiplicative').
        
    Returns:
        pd.DataFrame: The processed and binned features for the given IDs.
    """
    # Filter data based on provided IDs
    df = cpt_df[cpt_df["sondering_id"].isin(sondering_ids)].copy()
    
    if df.empty:
        print("Warning: No data found for the provided sondering_ids.")
        return pd.DataFrame()

    # order + bin
    id_col = "sondering_id"
    depth_col = "diepte"
    df = df.sort_values([id_col, depth_col]).copy()
    max_depth = float(df[depth_col].max())
    
    # create bins
    bins = np.arange(0, max_depth + bin_w, bin_w) if bin_w > 0 else np.array([0, max_depth or 1.0])
    if len(bins) < 2:
        bins = np.array([0, max_depth + (bin_w if bin_w > 0 else 1.0)])
    
    df["depth_bin"] = pd.cut(df["diepte"], bins=bins, include_lowest=True, ordered=True)
    
    # create raw QC column if it exists in source, else assume 'qc' is raw
    if "qc" in df.columns:
        df["QC_raw"] = df["qc"]

    # optional trend extraction
    if do_extract_trend:
        feat_cols_trend = [c for c in ["qc", "fs", "qtn"] if c in df.columns]
        if feat_cols_trend:
            # Sort again just to be safe for rolling/decomposition
            df = df.sort_values(["sondering_id", "diepte"]).copy()

            def _trend_and_fill(group: pd.DataFrame) -> pd.DataFrame:
                # Ensure the group identifier column exists
                if "sondering_id" not in group.columns:
                    group = group.copy()
                    group["sondering_id"] = group.name

                freq = freq_from_depth(group["diepte"].values)
                for col in feat_cols_trend:
                    vals = extract_trend(group[col].values, freq=freq, decomp_type=trend_type)
                    v = pd.Series(pd.to_numeric(vals, errors="coerce"))
                    v = v.replace([np.inf, -np.inf], np.nan)
                    v = v.ffill().bfill().bfill().to_numpy()
                    group[col] = v
                return group

            df = (df.groupby("sondering_id", group_keys=False, observed=False)
                           .apply(_trend_and_fill))

    # Stats for binning + whole
    feat_cols = [c for c in ["qc", "fs", "rf", "qtn", "fr", "diepte", "diepte_mtaw"] if c in df.columns]
    geog_cols = [c for c in ["diepte", "diepte_mtaw"] if c in df.columns]

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
    def q10(x): return float(np.nanpercentile(x, 10))
    def q50(x): return float(np.nanpercentile(x, 50))
    def q90(x): return float(np.nanpercentile(x, 90))

    def cv(x):
        m = np.nanmean(x)
        s = np.nanstd(x, ddof=1)
        return float(np.nan) if (not np.isfinite(m) or m == 0) else float(s / m)
    # max
    def max_na(x):
        return float(np.nanmax(x)) if np.size(x) else float(np.nan)

    # Additional NaN-safe stats
    def min_na(x):
        return float(np.nanmin(x)) if np.size(x) else float(np.nan)

    def range_na(x):
        try:
            return float(np.nanmax(x) - np.nanmin(x))
        except Exception:
            return float(np.nan)

    def trimmed_mean_na(x, trim=0.1):
        v = np.array(x, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(np.nan)
        v.sort()
        k = int(np.floor(trim * v.size))
        v = v[k: v.size - k] if v.size - 2 * k > 0 else v
        return float(np.mean(v))

    def skew_na(x):
        v = np.array(x, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 3:
            return float(np.nan)
        m = np.mean(v); s = np.std(v, ddof=1)
        if s == 0:
            return float(0.0)
        return float(np.mean(((v - m) / s) ** 3))

    def kurtosis_na(x):
        v = np.array(x, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 4:
            return float(np.nan)
        m = np.mean(v); s = np.std(v, ddof=1)
        if s == 0:
            return float(0.0)
        return float(np.mean(((v - m) / s) ** 4) - 3.0)  # excess kurtosis

    def q01(x):
        return float(np.nanpercentile(x, 1))

    def q99(x):
        return float(np.nanpercentile(x, 99))

    def entropy_na(x, bins=20):
        v = np.array(x, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(np.nan)
        # histogram-based entropy
        hist, _ = np.histogram(v, bins=bins)
        p = hist.astype(float)
        p = p / (p.sum() if p.sum() > 0 else 1.0)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def valid_n(x):
        v = np.array(x, dtype=float)
        return float(np.sum(np.isfinite(v)))

    def frac_missing(x):
        v = np.array(x, dtype=float)
        return float(np.mean(~np.isfinite(v))) if v.size else float(np.nan)

    stats = {
        "sd": sd_na, "mean": mean_na,
        "iqr": iqr_na, "median": median_na, "mad": mad_na,
        "q01": q01, "q10": q10, "q50": q50, "q90": q90, "q99": q99,
        "min": min_na, "max": max_na, "range": range_na,
        "cv": cv, "trimmed_mean": trimmed_mean_na,
        "skew": skew_na, "kurtosis": kurtosis_na,
        "entropy": entropy_na, "valid_n": valid_n, "frac_missing": frac_missing
    }

    # unique lithostrat per bin
    litho_dept = (
        df.loc[:, ["sondering_id", "lithostrat_id", "depth_bin"]]
              .drop_duplicates(subset=["sondering_id", "depth_bin", "lithostrat_id"])
              .copy()
    )

    # aggregations
    summaries_bin = agg_features(df, feat_cols, [id_col, "depth_bin"], stats) if feat_cols else pd.DataFrame()
    summaries_whole = agg_features(df, geog_cols, [id_col], stats, suffix="_whole") if geog_cols else pd.DataFrame()

    # merge
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
    
    # remove missing lithostrat_id
    dt = dt[~dt["lithostrat_id"].isna()].copy()

    # QC spikes
    if "QC_raw" in df.columns and not df["QC_raw"].isna().all():
        qc_ok = df.dropna(subset=["QC_raw"]).copy()
        qc_spikes = (
            qc_ok.groupby([id_col, "depth_bin"], dropna=False, observed=False)["QC_raw"]
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

    # Fill missing values with the mean of that column grouped by sondering_id
    # This handles cases where specific bins have missing features (NaN) by using the sondering average
    numeric_cols = dt.select_dtypes(include=[np.number]).columns
    cols_to_fill = [c for c in numeric_cols if c not in [id_col, "depth_bin"] and dt[c].isna().any()]

    if cols_to_fill:
        dt[cols_to_fill] = dt.groupby(id_col)[cols_to_fill].transform(
            lambda g: g.fillna(g.mean())
        )

    return dt

