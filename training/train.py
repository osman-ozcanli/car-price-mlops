import os
import sys
import pandas as pd
from datasets import load_dataset
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_quality_agent import run as data_quality_run
from agents.performance_agent import run as performance_run
from agents.deploy_agent import run as deploy_run
from training.drift import detect_drift


# AddInteractions her .py dosyasında tanımlı olmalı (joblib deserialize için)
class AddInteractions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_x_odo"] = X["age"] * X["odometer"]
        return X


ORIGINAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "original_data.parquet")
FEEDBACK_REPO_ID = "Osman-Ozcanli/car_price_prediction_feedback"

REQUIRED_COLS = [
    "make", "model", "trim", "body", "transmission", "state",
    "condition", "odometer", "color", "interior",
    "sellingprice", "age"
]  # seller kaldırıldı (Madde #3): app.py her zaman "unknown" gönderiyor, bilgisiz feature


def load_feedback() -> pd.DataFrame:
    print("[train] HuggingFace'den feedback verisi çekiliyor...")
    try:
        dataset = load_dataset(FEEDBACK_REPO_ID, split="train")
        df = dataset.to_pandas()
        print(f"[train] {len(df)} satır feedback yüklendi.")
        return df
    except Exception as e:
        if "EmptyDatasetError" in type(e).__name__ or "doesn't contain any data files" in str(e):
            print("[train] Feedback dataset henüz boş. Pipeline atlanıyor.")
            return pd.DataFrame()
        raise


def main():
    print("=" * 60)
    print("[train] Pipeline başladı.")
    print("=" * 60)

    # 1. Feedback çek
    df_feedback = load_feedback()
    if len(df_feedback) == 0:
        print("[train] Feedback verisi boş, pipeline durduruluyor.")
        return

    # 2. Agent 1 — Veri Kalite Kontrolü
    print("\n[train] Agent 1: Veri kalite kontrolü...")
    ok, msg = data_quality_run(df_feedback)
    if not ok:
        print(f"[train] Veri kalite kontrolü BAŞARISIZ: {msg}")
        print("[train] Pipeline durduruluyor.")
        return
    print(f"[train] Agent 1 geçti: {msg}")

    # 3. Drift Detection
    print("\n[train] Drift detection...")
    df_original = pd.read_parquet(ORIGINAL_DATA_PATH)
    drift_results = detect_drift(df_original, df_feedback)
    # Drift varsa sadece logla, durdurmaz

    # 4. Orijinal + feedback birleştir
    print("\n[train] Orijinal veri + feedback birleştiriliyor...")
    df_full = pd.concat(
        [df_original[REQUIRED_COLS], df_feedback[REQUIRED_COLS]],
        ignore_index=True
    )
    print(f"[train] Toplam eğitim verisi: {len(df_full):,} satır")

    # 5. Agent 2 — Eğit + Karşılaştır
    print("\n[train] Agent 2: Model eğitimi ve karşılaştırma...")
    new_model, new_preprocessor, new_interactions, new_pt, new_rmse, old_rmse, is_better = performance_run(df_full)

    if not is_better:
        print(f"\n[train] Yeni model daha kötü (${new_rmse:,.0f} >= ${old_rmse:,.0f}). Deploy yapılmayacak.")
        print("[train] Pipeline tamamlandı — eski model korunuyor.")
        return

    # 6. Agent 3 — Deploy
    print("\n[train] Agent 3: Deploy...")
    deployed, msg = deploy_run(new_model, new_preprocessor, new_interactions, new_pt, new_rmse, old_rmse)
    print(f"[train] Deploy sonucu: {msg}")

    print("\n" + "=" * 60)
    print("[train] Pipeline tamamlandı.")
    print("=" * 60)


if __name__ == "__main__":
    main()
