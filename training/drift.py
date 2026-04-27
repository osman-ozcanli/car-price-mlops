import pandas as pd
from scipy.stats import ks_2samp


DRIFT_FEATURES = ["age", "odometer", "condition", "sellingprice"]


def detect_drift(df_original: pd.DataFrame, df_new: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    KS testi ile drift detection.
    Drift varsa sadece logla, pipeline'ı durdurma.
    """
    results = {}
    for col in DRIFT_FEATURES:
        stat, p_value = ks_2samp(df_original[col], df_new[col])
        drifted = p_value < threshold
        results[col] = {
            "drifted": drifted,
            "p_value": round(p_value, 4),
            "statistic": round(stat, 4),
        }
        status = "DRIFT VAR" if drifted else "normal"
        print(f"[drift] {col}: {status} (p={p_value:.4f})")

    drifted_cols = [col for col, r in results.items() if r["drifted"]]
    if drifted_cols:
        print(f"[drift] Uyarı: {drifted_cols} kolonlarında drift tespit edildi. Pipeline devam ediyor.")
    else:
        print("[drift] Drift yok, tüm kolonlar normal.")

    return results
