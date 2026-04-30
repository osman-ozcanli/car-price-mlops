"""Generate a quick A/B comparison report from the feedback dataset.

Reads ``feedback.parquet`` from the HF Dataset, splits by ``model_version``
(``v_current`` vs ``v_previous``), and prints a small summary table comparing
how each variant's predictions did against the user-reported sale price.

Usage:
    HF_TOKEN=... python scripts/ab_report.py [--out reports/ab_YYYYMMDD.md]

The ``predicted`` column is reconstructed by re-running each variant of the
model on the recorded inputs — so this script downloads both pickles. It is
NOT a statistical test; sample sizes are small in demo mode and any real
significance test would mislead. It is a sanity dashboard.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Ensure ``common`` is importable when run from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.constants import DATASET_REPO, INFLATION_MULTIPLIER, MODEL_REPO  # noqa: E402
from common.transformers import AddInteractions  # noqa: E402

FEEDBACK_FILE = "feedback.parquet"


def _load(filename: str):
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=filename))


def predict_with(model_filename: str, df: pd.DataFrame, preproc, pt) -> np.ndarray:
    model = _load(model_filename)
    interact = AddInteractions()
    X = interact.transform(df)
    X_proc = preproc.transform(X)
    y_t = model.predict(X_proc)
    y = pt.inverse_transform(y_t.reshape(-1, 1)).ravel()
    return np.clip(y * INFLATION_MULTIPLIER, 500, None)


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"variant": label, "n": 0}
    err = df["sellingprice"].to_numpy() - df["predicted"].to_numpy()
    abs_err = np.abs(err)
    return {
        "variant": label,
        "n": len(df),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_abs_err": float(np.median(abs_err)),
        "bias": float(err.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="Optional output path for a markdown report.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    print(f"Loading feedback from {DATASET_REPO}...")
    try:
        path = hf_hub_download(repo_id=DATASET_REPO, filename=FEEDBACK_FILE,
                               repo_type="dataset", token=token)
    except Exception as e:
        print(f"Could not load feedback: {e}", file=sys.stderr)
        return 1

    df = pd.read_parquet(path)
    if df.empty or "model_version" not in df.columns:
        print("Feedback dataset has no model_version column or is empty.")
        return 0

    print(f"Loaded {len(df)} rows. Running both model variants...")
    preproc = _load("preprocessor.pkl")
    pt = _load("power_transformer.pkl")

    feature_cols = ["age", "odometer", "condition", "body", "transmission",
                    "state", "color", "interior", "make", "model", "trim"]
    df = df.copy()

    rows = []
    for variant_label, model_file in [
        ("v_current", "lgbm_tuned.pkl"),
        ("v_previous", "lgbm_tuned_prev.pkl"),
    ]:
        sub = df[df["model_version"] == variant_label].copy()
        if sub.empty:
            rows.append(summarize(sub, variant_label))
            continue
        try:
            sub["predicted"] = predict_with(model_file, sub[feature_cols], preproc, pt)
        except Exception as e:
            print(f"  Could not predict for {variant_label}: {e}", file=sys.stderr)
            rows.append({"variant": variant_label, "n": len(sub), "error": str(e)})
            continue
        rows.append(summarize(sub, variant_label))

    summary = pd.DataFrame(rows)
    print("\nA/B summary:")
    print(summary.to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# A/B report — {ts}\n\n")
            f.write(summary.to_markdown(index=False))
            f.write("\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
