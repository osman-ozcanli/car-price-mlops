"""Drift detection via two-sample Kolmogorov-Smirnov test.

Runs after data quality and before training. By design, drift findings are
logged but never block the pipeline — a drifted feedback batch is *more*
informative for the next retraining, not less.
"""
import pandas as pd
from scipy.stats import ks_2samp

DRIFT_FEATURES = ["age", "odometer", "condition", "sellingprice"]


def detect_drift(df_original: pd.DataFrame, df_new: pd.DataFrame,
                 threshold: float = 0.05) -> dict:
    results = {}
    for col in DRIFT_FEATURES:
        stat, p_value = ks_2samp(df_original[col], df_new[col])
        drifted = p_value < threshold
        results[col] = {
            "drifted": bool(drifted),
            "p_value": round(p_value, 4),
            "statistic": round(stat, 4),
        }
        status = "DRIFT" if drifted else "normal"
        print(f"[drift] {col}: {status} (p={p_value:.4f})")

    drifted_cols = [c for c, r in results.items() if r["drifted"]]
    if drifted_cols:
        print(f"[drift] Warning: drift detected in {drifted_cols}. "
              f"Pipeline continues.")
    else:
        print("[drift] No drift detected.")
    return results
