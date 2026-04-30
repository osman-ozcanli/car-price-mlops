"""Agent 1 — Data Quality.

Deterministic statistical checks on the feedback batch before it enters the
training pipeline. This is the first of three audit layers; survives all
checks then proceeds to drift detection and the performance agent.
"""
import pandas as pd

from common.constants import PRICE_MIN, PRICE_MAX, ODOMETER_MAX


def run(df_new: pd.DataFrame) -> tuple[bool, str]:
    if len(df_new) == 0:
        return False, "Feedback dataset is empty."

    missing_total = df_new.isnull().sum().sum()
    if missing_total > 0:
        missing = df_new.isnull().sum()
        missing = missing[missing > 0].to_dict()
        return False, f"Missing values found: {missing}"

    # Z-score outlier check is only meaningful with a non-trivial sample.
    # Below ~30 rows, a single legitimate luxury car can trip the 5% gate.
    std = df_new["sellingprice"].std()
    if len(df_new) >= 30 and std and std > 0:
        z = (df_new["sellingprice"] - df_new["sellingprice"].mean()) / std
        outlier_ratio = (z.abs() > 3).mean()
        if outlier_ratio > 0.05:
            return False, f"Outlier ratio too high: {outlier_ratio * 100:.1f}%"

    price_invalid = ((df_new["sellingprice"] < PRICE_MIN) |
                     (df_new["sellingprice"] > PRICE_MAX)).sum()
    if price_invalid > 0:
        return False, f"{price_invalid} rows fall outside the accepted price range."

    odo_invalid = (df_new["odometer"] > ODOMETER_MAX).sum()
    if odo_invalid > 0:
        return False, f"{odo_invalid} rows exceed the odometer limit."

    return True, "Data quality checks passed."
