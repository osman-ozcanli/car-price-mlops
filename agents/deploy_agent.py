import os
import pickle
import json
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import HfApi


# AddInteractions her .py dosyasında tanımlı olmalı (joblib deserialize için)
class AddInteractions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_x_odo"] = X["age"] * X["odometer"]
        return X


HF_REPO_ID = "Osman-Ozcanli/car_price_prediction"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [deploy_agent] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def run(new_model, new_preprocessor, new_interactions, new_pt, new_rmse: float, old_rmse: float) -> tuple[bool, str]:
    """
    Döndürür: (deployed: bool, mesaj: str)
    """
    if new_rmse >= old_rmse:
        msg = (
            f"Deploy iptal: Yeni RMSE (${new_rmse:,.0f}) >= Eski RMSE (${old_rmse:,.0f}). "
            f"Eski model korunuyor."
        )
        logger.warning(msg)
        return False, msg

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN bulunamadı — deploy yapılamıyor."
        logger.error(msg)
        return False, msg

    version_tag = f"v{datetime.now().strftime('%Y%m%d')}"
    tmp_dir = "/tmp/car_price_deploy"
    os.makedirs(tmp_dir, exist_ok=True)

    files = {
        "lgbm_tuned.pkl": new_model,
        "preprocessor.pkl": new_preprocessor,
        "interactions.pkl": new_interactions,
        "power_transformer.pkl": new_pt,
    }
    for filename, obj in files.items():
        _save_pkl(obj, f"{tmp_dir}/{filename}")

    meta = {
        "version": version_tag,
        "deployed_at": datetime.utcnow().isoformat(),
        "new_rmse": round(new_rmse, 2),
        "old_rmse": round(old_rmse, 2),
        "improvement": round(old_rmse - new_rmse, 2),
    }
    with open(f"{tmp_dir}/deploy_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    api = HfApi(token=hf_token)

    for filename in list(files.keys()) + ["deploy_meta.json"]:
        api.upload_file(
            path_or_fileobj=f"{tmp_dir}/{filename}",
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"deploy {version_tag}: RMSE ${new_rmse:,.0f} (eski: ${old_rmse:,.0f})",
        )

    try:
        api.create_tag(
            repo_id=HF_REPO_ID,
            repo_type="model",
            tag=version_tag,
            message=f"RMSE: ${new_rmse:,.0f}",
        )
    except Exception as e:
        logger.warning(f"Tag oluşturulamadı (mevcut olabilir): {e}")

    msg = (
        f"Deploy başarılı [{version_tag}]: "
        f"RMSE ${new_rmse:,.0f} (iyileşme: ${old_rmse - new_rmse:,.0f})"
    )
    logger.info(msg)
    return True, msg
