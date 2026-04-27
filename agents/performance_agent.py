import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from huggingface_hub import hf_hub_download


# AddInteractions her .py dosyasında tanımlı olmalı (joblib deserialize için)
class AddInteractions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_x_odo"] = X["age"] * X["odometer"]
        return X

# app.py __main__ olarak çalışır; pickle uyumluluğu için modül yolunu sabitle
AddInteractions.__module__ = "__main__"


NUM_COLS = ["age", "odometer", "condition", "age_x_odo"]
ORD_COLS = ["body", "transmission", "color", "interior"]   # state -> TGT_COLS'a taşındı (Madde #5)
TGT_COLS = ["make", "model", "trim", "state"]  # state TargetEncoder'a alındı: bilinmeyen eyalet için global mean fallback

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

HF_REPO_ID = "Osman-Ozcanli/car_price_prediction"


def _load_current_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename="lgbm_tuned.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_current_preprocessor():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename="preprocessor.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_power_transformer():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename="power_transformer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_preprocessor():
    num_transformer = StandardScaler()
    ord_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tgt_transformer = TargetEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_COLS),
            ("ord", ord_transformer, ORD_COLS),
            ("tgt", tgt_transformer, TGT_COLS),
        ]
    )
    return preprocessor


def run(df_full: pd.DataFrame) -> tuple:
    """
    df_full: original_data + feedback birleşimi
    Döndürür: (new_model, new_preprocessor, new_pt, new_rmse, old_rmse, is_better)
    """
    pt_current = _load_power_transformer()
    df_full = df_full.copy()

    feature_cols = NUM_COLS[:-1] + ORD_COLS + TGT_COLS  # age_x_odo AddInteractions'da eklenir
    feature_cols_base = ["age", "odometer", "condition", "body", "transmission",
                         "state", "color", "interior", "make", "model", "trim"]

    X = df_full[feature_cols_base]
    y = df_full["sellingprice"]

    y_transformed = pt_current.transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_transformed, test_size=0.1, random_state=42
    )

    add_interactions = AddInteractions()
    X_train = add_interactions.transform(X_train)
    X_val = add_interactions.transform(X_val)

    preprocessor = _build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_val_proc = preprocessor.transform(X_val)

    new_model = LGBMRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbose=-1)
    new_model.fit(X_train_proc, y_train)

    y_pred_transformed = new_model.predict(X_val_proc)
    y_pred = pt_current.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
    y_val_orig = pt_current.inverse_transform(y_val.reshape(-1, 1)).ravel()

    new_rmse = float(np.sqrt(mean_squared_error(y_val_orig, y_pred)))

    # Eski model karşılaştırması: eski preprocessor (seller'lı) ile transform et
    old_model = _load_current_model()
    old_preprocessor = _load_current_preprocessor()
    X_val_old = X_val.copy()
    if "seller" not in X_val_old.columns:
        X_val_old["seller"] = "unknown"  # eski preprocessor seller bekliyor
    X_val_proc_old = old_preprocessor.transform(X_val_old)
    old_pred_transformed = old_model.predict(X_val_proc_old)
    old_pred = pt_current.inverse_transform(old_pred_transformed.reshape(-1, 1)).ravel()
    old_rmse = float(np.sqrt(mean_squared_error(y_val_orig, old_pred)))

    is_better = new_rmse < old_rmse

    print(f"[performance_agent] Yeni RMSE: ${new_rmse:,.0f} | Eski RMSE: ${old_rmse:,.0f} | İyi mi: {is_better}")

    return new_model, preprocessor, add_interactions, pt_current, new_rmse, old_rmse, is_better
