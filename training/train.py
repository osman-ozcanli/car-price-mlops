"""Retraining pipeline orchestrator.

Runs the three-agent chain:
    1. data_quality_agent  — schema / outlier sanity on feedback
    2. drift detection     — KS test against original_data (logging only)
    3. performance_agent   — train candidate + compare to current production
    4. deploy_agent        — conditional publish if candidate is better
"""
import os
import sys

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_quality_agent import run as data_quality_run  # noqa: E402
from agents.deploy_agent import run as deploy_run  # noqa: E402
from agents.performance_agent import run as performance_run  # noqa: E402
from training.drift import detect_drift  # noqa: E402

ORIGINAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "original_data.parquet")
_HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
FEEDBACK_REPO_ID = f"{_HF_USERNAME}/car_price_prediction_feedback"

REQUIRED_COLS = [
    "make", "model", "trim", "body", "transmission", "state",
    "condition", "odometer", "color", "interior",
    "sellingprice", "age",
]


def load_feedback() -> pd.DataFrame:
    print("[train] Fetching feedback dataset from HuggingFace...")
    try:
        dataset = load_dataset(FEEDBACK_REPO_ID, split="train")
        df = dataset.to_pandas()
        print(f"[train] Loaded {len(df)} feedback rows.")
        return df
    except Exception as e:
        if "EmptyDatasetError" in type(e).__name__ or "doesn't contain any data files" in str(e):
            print("[train] Feedback dataset is empty. Skipping pipeline.")
            return pd.DataFrame()
        raise


def main() -> None:
    print("=" * 60)
    print("[train] Pipeline started.")
    print("=" * 60)

    df_feedback = load_feedback()
    if len(df_feedback) == 0:
        print("[train] No feedback — exiting.")
        return

    print("\n[train] Agent 1: data quality check...")
    ok, msg = data_quality_run(df_feedback)
    if not ok:
        print(f"[train] Agent 1 FAILED: {msg}")
        return
    print(f"[train] Agent 1 passed: {msg}")

    print("\n[train] Drift detection...")
    df_original = pd.read_parquet(ORIGINAL_DATA_PATH)
    drift_results = detect_drift(df_original, df_feedback)  # logs only; never blocks

    print("\n[train] Merging original_data + feedback...")
    df_full = pd.concat(
        [df_original[REQUIRED_COLS], df_feedback[REQUIRED_COLS]],
        ignore_index=True,
    )
    print(f"[train] Total training rows: {len(df_full):,}")

    print("\n[train] Agent 2: train + compare...")
    new_model, new_preprocessor, new_interactions, new_pt, \
        new_rmse, old_rmse, is_better = performance_run(df_full)

    if not is_better:
        print(f"\n[train] Candidate is worse (${new_rmse:,.0f} >= ${old_rmse:,.0f}). "
              f"Keeping the current model.")
        return

    print("\n[train] Agent 3: deploy...")
    deployed, msg = deploy_run(new_model, new_preprocessor, new_interactions,
                                new_pt, new_rmse, old_rmse,
                                drift_results=drift_results,
                                n_feedback=len(df_feedback))
    print(f"[train] Deploy result: {msg}")

    print("\n" + "=" * 60)
    print("[train] Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
