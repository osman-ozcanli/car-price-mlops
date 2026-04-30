"""Agent 3 — Deploy.

Conditionally publishes a candidate model to HuggingFace. Uploads only the
artifacts that actually changed (model + preprocessor + meta). Saves the
previous active model as ``lgbm_tuned_prev.pkl`` for A/B labelling and as a
rollback target.

``interactions.pkl`` is intentionally not uploaded — ``AddInteractions`` is
stateless and constructed fresh at inference time.
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import joblib
from huggingface_hub import HfApi, hf_hub_download

# AddInteractions is imported here so any joblib-pickled artifact passed in
# from performance_agent can resolve the class via this module path.
from common.transformers import AddInteractions  # noqa: F401

_HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
HF_REPO_ID = f"{_HF_USERNAME}/car_price_prediction"
HF_SPACE_ID = f"{_HF_USERNAME}/car_price_prediction_space"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [deploy_agent] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(new_model, new_preprocessor, new_interactions, new_pt,
        new_rmse: float, old_rmse: float,
        drift_results: dict | None = None,
        n_feedback: int | None = None) -> tuple[bool, str]:
    """Returns ``(deployed, message)``.

    ``drift_results`` is the dict returned by ``training.drift.detect_drift``
    and is recorded in ``deploy_meta.json`` for post-hoc analysis. ``None``
    is treated as "drift detection was skipped."
    """
    if new_rmse >= old_rmse:
        msg = (f"Deploy skipped: new RMSE (${new_rmse:,.0f}) >= "
               f"old RMSE (${old_rmse:,.0f}). Keeping the existing model.")
        logger.warning(msg)
        return False, msg

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN missing — cannot deploy."
        logger.error(msg)
        return False, msg

    version_tag = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    tmp_dir = tempfile.mkdtemp(prefix="car_price_deploy_")

    artifacts = {
        "lgbm_tuned.pkl": new_model,
        "preprocessor.pkl": new_preprocessor,
    }
    for filename, obj in artifacts.items():
        joblib.dump(obj, f"{tmp_dir}/{filename}")

    drift_summary = None
    if drift_results:
        drift_summary = {
            "drifted_columns": [c for c, r in drift_results.items() if r.get("drifted")],
            "per_column": {c: {"p_value": r["p_value"], "drifted": r["drifted"]}
                           for c, r in drift_results.items()},
        }

    meta = {
        "version": version_tag,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "new_rmse": round(new_rmse, 2),
        "old_rmse": round(old_rmse, 2) if old_rmse != float("inf") else None,
        "improvement": (round(old_rmse - new_rmse, 2)
                        if old_rmse != float("inf") else None),
        "n_feedback": n_feedback,
        "drift": drift_summary,
    }
    with open(f"{tmp_dir}/deploy_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    api = HfApi(token=hf_token)

    # Snapshot the currently active model as the previous version for A/B
    # labelling and rollback. Skipped on the very first deploy.
    try:
        current_path = hf_hub_download(repo_id=HF_REPO_ID, filename="lgbm_tuned.pkl")
        api.upload_file(
            path_or_fileobj=current_path,
            path_in_repo="lgbm_tuned_prev.pkl",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"snapshot prev model (before {version_tag})",
        )
        logger.info("Previous model snapshotted as lgbm_tuned_prev.pkl.")
    except Exception as e:
        logger.warning(f"Could not snapshot previous model (likely first deploy): {e}")

    for filename in list(artifacts.keys()) + ["deploy_meta.json"]:
        api.upload_file(
            path_or_fileobj=f"{tmp_dir}/{filename}",
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"deploy {version_tag}: RMSE ${new_rmse:,.0f}",
        )

    try:
        api.create_tag(
            repo_id=HF_REPO_ID,
            repo_type="model",
            tag=version_tag,
            message=f"RMSE: ${new_rmse:,.0f}",
        )
    except Exception as e:
        logger.warning(f"Tag creation failed (may already exist): {e}")

    try:
        api.restart_space(repo_id=HF_SPACE_ID)
        logger.info(f"HF Space restart triggered: {HF_SPACE_ID}")
    except Exception as e:
        logger.warning(f"Space restart failed (manual restart may be needed): {e}")

    msg = (f"Deploy succeeded [{version_tag}]: RMSE ${new_rmse:,.0f}")
    logger.info(msg)
    return True, msg
