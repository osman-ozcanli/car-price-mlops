# CLAUDE.md — Araba Fiyat Tahmin MLOps Pipeline

## Proje Özeti
Kullanıcıdan araba özelliği alıp fiyat tahmini yapan, kullanıcı geri bildirimiyle büyüyen,
otomatik yeniden eğitilen ve kendini güncelleyen end-to-end MLOps pipeline.

---

## HuggingFace Kaynakları

| Kaynak | Repo |
|--------|------|
| Model | `Osman-Ozcanli/car_price_prediction` |
| Dataset (feedback) | `Osman-Ozcanli/car_price_prediction_feedback` |
| Space | `Osman-Ozcanli/car_price_prediction_space` |

---

## Proje Yapısı

```
car-price-mlops/
│
├── app/
│   └── app.py                  # Streamlit uygulaması (UI + feedback + A/B)
│
├── training/
│   ├── train.py                # Agent zincirini orchestrate eder
│   ├── drift.py                # KS testi ile drift detection
│   └── original_data.parquet  # Orijinal eğitim verisi (ASLA değiştirilmez)
│
├── agents/
│   ├── data_quality_agent.py   # Agent 1: Veri kalite kontrolü
│   ├── performance_agent.py    # Agent 2: Model performans karşılaştırma
│   └── deploy_agent.py         # Agent 3: Deploy kararı
│
├── .github/
│   └── workflows/
│       └── retrain.yml         # GitHub Actions scheduler
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Model Detayları

### Feature Listesi (kesin, değişmez)
```python
# Numeric — StandardScaler
num_cols = ["age", "odometer", "condition", "age_x_odo"]

# Ordinal — OrdinalEncoder
ord_cols = ["body", "transmission", "state", "color", "interior"]

# Target — TargetEncoder
tgt_cols = ["make", "model", "trim"]   # seller kaldırıldı (Madde #3 düzeltmesi)

# Türetilen
# age_x_odo = age * odometer  →  AddInteractions class ile pipeline içinde eklenir

# seller → feature listesinden çıkarıldı. app.py mevcut pkl uyumluluğu için input_dict'te
#          "unknown" göndermeye devam eder; yeni preprocessor bu kolonu sessizce drop eder.
```

### Pipeline Sırası (kesin, değişmez)
```
input_dict → AddInteractions → preprocessor.pkl → lgbm_tuned.pkl
           → power_transformer.pkl (inverse) → clip(500, None)
           → x1.38 enflasyon katsayısı → kullanıcıya göster
```

### pkl Dosyaları (HF Model repo'da)
```
lgbm_tuned.pkl          # LightGBM tuned model
preprocessor.pkl        # ColumnTransformer (num+ord+tgt)
interactions.pkl        # AddInteractions transformer
power_transformer.pkl   # Yeo-Johnson PowerTransformer
car_hierarchy.json      # make→model→trim hiyerarşisi (UI için)
```

### Model Performansı
- Test RMSE: ~$1,814 (2015 veri bazlı)
- Enflasyon düzeltmesi: x1.38 (2015→2025)
- Eğitim verisi: 552,941 satır, ABD ikinci el araç müzayede kayıtları

### Best Params (best_params.json)
```json
{
  "learning_rate": 0.01279,
  "num_leaves": 265,
  "min_child_samples": 15,
  "subsample": 0.671,
  "colsample_bytree": 0.700,
  "reg_alpha": 0.000455,
  "reg_lambda": 0.000275,
  "n_estimators": 3298
}
```

---

## Mimari — Tam Döngü

```
Kullanıcı → Tahmin → Beğeni → Validasyon → HF Dataset
                                                ↓
                                     GitHub Actions (Pazar 02:00 UTC)
                                     veya threshold=10 yeni satır
                                                ↓
                                       Agent 1: Veri Kalite
                                                ↓
                                       drift.py: KS testi
                                                ↓
                                       Agent 2: Eğit + Karşılaştır
                                                ↓
                                       Agent 3: Deploy / İptal
                                                ↓
                                     HF Space yeni modeli çeker
```

---

## FAZA 1 — Temel Kurulum ✅ TAMAMLANDI
- Model eğitildi (01_cleaning → 05_evaluation)
- pkl dosyaları HF Model repo'ya yüklendi
- HF Space açıldı ve app.py deploy edildi
- HF Dataset repo açıldı (feedback için)
- GitHub repo açıldı

---

## FAZA 2 — Kullanıcı Verisi Toplama ✅ TAMAMLANDI

### Validasyon Kuralları (app.py'de aktif)
```python
# odometer > 300,000 mil → reddet
# price < 500 veya price > 78,000 → reddet
```

### A/B Testing (app.py'de aktif)
```python
st.session_state.model_version = random.choice(["v_current", "v_previous"])
# Her feedback satırına model_version eklenir
```

### Threshold Tetikleme
```python
THRESHOLD = 10  # 10 yeni onaylı satır → GitHub Actions workflow_dispatch
```

---

## FAZA 3 — Agent Zinciri (YAPILACAK)

### Agent 1 — data_quality_agent.py
```python
def run(df_new):
    # 1. Eksik değer kontrolü
    if df_new.isnull().sum().sum() > 0:
        return False, "Eksik değer var"
    # 2. Aykırı değer: sellingprice z-score > 3 olan satır oranı > %5
    z = (df_new["sellingprice"] - df_new["sellingprice"].mean()) / df_new["sellingprice"].std()
    if (z.abs() > 3).mean() > 0.05:
        return False, "Aykırı değer oranı yüksek"
    return True, "Veri temiz"
```

### Agent 2 — performance_agent.py
```python
def run(df_full):
    # df_full = original_data + yeni feedback
    # Eğit, val RMSE hesapla, eski model ile karşılaştır
    # Döndür: new_model, new_rmse, old_rmse, is_better (bool)
```

### Agent 3 — deploy_agent.py
```python
def run(new_model, new_rmse, old_rmse):
    # Eğer new_rmse >= old_rmse → deploy iptal, log yaz
    # Eğer new_rmse < old_rmse → HF'e push et, version tag ekle
    version_tag = f"v{datetime.now().strftime('%Y%m%d')}"
```

---

## FAZA 4 — Drift Detection (YAPILACAK)

### drift.py
```python
from scipy.stats import ks_2samp

DRIFT_FEATURES = ["age", "odometer", "condition", "sellingprice"]

def detect_drift(df_original, df_new, threshold=0.05):
    results = {}
    for col in DRIFT_FEATURES:
        stat, p_value = ks_2samp(df_original[col], df_new[col])
        results[col] = {"drifted": p_value < threshold, "p_value": round(p_value, 4)}
    return results
    # Drift varsa log yaz, pipeline devam eder (durdurmaz)
```

---

## FAZA 5 — GitHub Actions (YAPILACAK)

### retrain.yml
```yaml
name: Retrain Pipeline
on:
  schedule:
    - cron: "0 2 * * 0"   # Her Pazar 02:00 UTC
  workflow_dispatch:        # Manuel + threshold tetikleme
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - name: Agent Zincirini Çalıştır
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
        run: python training/train.py
```

---

## Sabitler

```python
# Enflasyon
INFLATION_MULTIPLIER = 1.38   # 2015 veri bazı → 2025 fiyatları

# Feedback eşiği
THRESHOLD = 10                 # kaç yeni satırda retraining tetiklensin

# Fiyat aralığı (model eğitim aralığı)
PRICE_MIN = 500
PRICE_MAX = 78_000

# Odometer sınırı
ODOMETER_MAX = 300_000
```

---

## Kesin Kurallar

1. `HF_TOKEN` hiçbir zaman kod içinde olmaz, sadece Secrets'ta.
2. `original_data.parquet` hiçbir zaman değiştirilmez — her retraining'de baz olarak kullanılır.
3. Yeni model eskisinden iyi değilse deploy yapılmaz, eski model kalır.
4. Kullanıcı feedback'i validasyondan geçmeden dataset'e girmez.
5. Her deploy versiyonlanır (HF tag) ve loglanır.
6. `seller` feature listesinden çıkarıldı (Madde #3). app.py geriye dönük uyumluluk için input_dict'te `"unknown"` göndermeye devam eder; yeni preprocessor bu kolonu drop eder.
7. `AddInteractions` class'ı her .py dosyasında tanımlı olmalı (joblib deserialize için).
8. Enflasyon katsayısı (x1.38) sadece kullanıcıya gösterilen son fiyata uygulanır — eğitimde kullanılmaz.

---

## GitHub Secrets

```
HF_TOKEN      → HuggingFace write token
HF_USERNAME   → Osman-Ozcanli
```
