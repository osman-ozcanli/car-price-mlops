"""Roll back to the previously deployed model.

Downloads ``lgbm_tuned_prev.pkl`` from the HF Model repo and re-uploads it as
``lgbm_tuned.pkl``, then restarts the Space so the change takes effect.

Usage:
    HF_TOKEN=... python scripts/rollback.py [--reason "explanation"]

Use this when a freshly deployed model misbehaves in production. The
performance agent's RMSE-gate prevents most bad deploys, but it can't catch
distribution-shift surprises that only appear under live load.

This script is one-shot and idempotent: running it twice in a row simply
makes the same model active again. It does not chain backwards through
older versions — for that, use HF model tags (``vYYYYMMDD``) and the
HuggingFace UI.
"""
import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download

HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
HF_REPO_ID = f"{HF_USERNAME}/car_price_prediction"
HF_SPACE_ID = f"{HF_USERNAME}/car_price_prediction_space"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", default="manual rollback",
                        help="Short note recorded in the HF commit message.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable is missing.", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    print(f"Downloading lgbm_tuned_prev.pkl from {HF_REPO_ID}...")
    try:
        prev_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename="lgbm_tuned_prev.pkl", token=token,
        )
    except Exception as e:
        print(f"Failed: {e}\nNo previous model to roll back to.", file=sys.stderr)
        return 2

    # Copy to temp so the upload doesn't pin the HF cache file.
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    tmp.close()
    with open(prev_path, "rb") as src, open(tmp.name, "wb") as dst:
        dst.write(src.read())

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("Uploading as lgbm_tuned.pkl (overwrites active model)...")
    api.upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="lgbm_tuned.pkl",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"rollback: {args.reason} ({stamp})",
    )

    print(f"Restarting Space {HF_SPACE_ID}...")
    try:
        api.restart_space(repo_id=HF_SPACE_ID)
    except Exception as e:
        print(f"Restart failed (may need manual restart): {e}", file=sys.stderr)
        return 3

    print("Rollback complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
