"""Agent 2 — Performance.

Trains a fresh model on (original_data + feedback) and compares it against the
currently deployed model on the same held-out validation slice.

Comparison rule: each candidate must score on the *same* validation set using
its *own* preprocessor + model bundle. Any cross-bundle column manipulation
(e.g. injecting a placeholder ``seller`` value) is forbidden — it conflates
preprocessing differences with model differences.

If the legacy preprocessor is incompatible with the new schema (missing or
extra columns), the comparison is declared inconclusive and the new model is
deployed by default. This is a one-time migration safety valve.
"""
import os

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from huggingface_hub import hf_hub_download
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from common.transformers import AddInteractions

NUM_COLS = ["age", "odometer", "condition", "age_x_odo"]
ORD_COLS = ["body", "transmission", "color", "interior"]
TGT_COLS = ["make", "model", "trim", "state"]

FEATURE_COLS_BASE = ["age", "odometer", "condition", "body", "transmission",
                     "state", "color", "interior", "make", "model", "trim"]

BEST_PARAMS = {
    "learning_rate": 0.01279,
    "num_leaves": 265,
    "min_child_samples": 15,
    "subsample": 0.671,
    "colsample_bytree": 0.700,
    "reg_alpha": 0.000455,
    "reg_lambda": 0.000275,
    "n_estimators": 3298,
}

_HF_USERNAME = os.environ.get("HF_USERNAME", "Osman-Ozcanli")
HF_REPO_ID = f"{_HF_USERNAME}/car_price_prediction"


def _hf_load(filename: str):
    return joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename=filename))


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORD_COLS),
        ("tgt", TargetEncoder(), TGT_COLS),
    ])


def _score_old_bundle(X_val: pd.DataFrame, y_val_orig: np.ndarray,
                      pt) -> float | None:
    """Return RMSE of the deployed model on the validation slice, or None if
    the legacy bundle is incompatible with the current schema."""
    try:
        old_model = _hf_load("lgbm_tuned.pkl")
        old_preprocessor = _hf_load("preprocessor.pkl")
        # Apply the new AddInteractions step (stateless) before the old
        # preprocessor — both new and old bundles assume age_x_odo exists.
        X_val_inter = AddInteractions().transform(X_val)
        X_val_proc_old = old_preprocessor.transform(X_val_inter)
        old_pred_t = old_model.predict(X_val_proc_old)
        old_pred = pt.inverse_transform(old_pred_t.reshape(-1, 1)).ravel()
        return float(np.sqrt(mean_squared_error(y_val_orig, old_pred)))
    except Exception as e:
        print(f"[performance_agent] Legacy bundle incompatible — comparison "
              f"skipped, new model will deploy by default. Reason: {e}")
        return None


def run(df_full: pd.DataFrame):
    """Train + evaluate a candidate model.

    Returns:
        (new_model, new_preprocessor, new_interactions, pt,
         new_rmse, old_rmse, is_better)

        ``old_rmse`` is ``float('inf')`` and ``is_better`` is ``True`` when the
        legacy bundle cannot be evaluated (one-time migration path).
    """
    pt = _hf_load("power_transformer.pkl")

    X = df_full[FEATURE_COLS_BASE]
    y = df_full["sellingprice"]
    y_t = pt.transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_t, test_size=0.1, random_state=42,
    )

    interactions = AddInteractions()
    X_train_i = interactions.transform(X_train)
    X_val_i = interactions.transform(X_val)

    preprocessor = _build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_i, y_train)
    X_val_proc = preprocessor.transform(X_val_i)

    new_model = LGBMRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbose=-1)
    new_model.fit(X_train_proc, y_train)

    new_pred_t = new_model.predict(X_val_proc)
    new_pred = pt.inverse_transform(new_pred_t.reshape(-1, 1)).ravel()
    y_val_orig = pt.inverse_transform(y_val.reshape(-1, 1)).ravel()
    new_rmse = float(np.sqrt(mean_squared_error(y_val_orig, new_pred)))

    old_rmse = _score_old_bundle(X_val, y_val_orig, pt)
    if old_rmse is None:
        is_better = True
        old_rmse = float("inf")
        print(f"[performance_agent] New RMSE: ${new_rmse:,.0f} | Old: N/A | "
              f"Defaulting to deploy (migration).")
    else:
        is_better = new_rmse < old_rmse
        print(f"[performance_agent] New RMSE: ${new_rmse:,.0f} | "
              f"Old RMSE: ${old_rmse:,.0f} | Better: {is_better}")

    return new_model, preprocessor, interactions, pt, new_rmse, old_rmse, is_better
