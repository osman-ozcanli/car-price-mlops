import pandas as pd
import numpy as np


def run(df_new: pd.DataFrame) -> tuple[bool, str]:
    if df_new.isnull().sum().sum() > 0:
        missing = df_new.isnull().sum()
        missing = missing[missing > 0].to_dict()
        return False, f"Eksik değer var: {missing}"

    if len(df_new) == 0:
        return False, "Veri seti boş"

    z = (df_new["sellingprice"] - df_new["sellingprice"].mean()) / df_new["sellingprice"].std()
    outlier_ratio = (z.abs() > 3).mean()
    if outlier_ratio > 0.05:
        return False, f"Aykırı değer oranı yüksek: %{outlier_ratio*100:.1f}"

    price_invalid = ((df_new["sellingprice"] < 500) | (df_new["sellingprice"] > 78_000)).sum()
    if price_invalid > 0:
        return False, f"Fiyat aralığı dışı satır var: {price_invalid} adet"

    odo_invalid = (df_new["odometer"] > 300_000).sum()
    if odo_invalid > 0:
        return False, f"Odometer sınırı aşılmış: {odo_invalid} adet"

    return True, "Veri temiz"
