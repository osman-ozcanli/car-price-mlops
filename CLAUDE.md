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

## Proje Yapısı (GitHub'daki güncel hal)

```
car-price-mlops/
│
├── app/
│   ├── app.py                  # Streamlit uygulaması (UI + feedback + A/B)
│   └── car_hierarchy.json      # make→model→trim hiyerarşisi (UI için)
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
├── scripts/
│   └── push_app_to_space.py    # app.py HF Space'e push eder (deploy_app.yml tarafından çağrılır)
│
├── .github/
│   └── workflows/
│       ├── retrain.yml         # Her Pazar 02:00 UTC + workflow_dispatch tetikleme
│       └── deploy_app.yml      # app/app.py değişince HF Space'e otomatik push
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

**Sadece lokalde olan dosyalar (gitignore'da):**
- `MEMORY.md`, `PROGRESS.md` — lokal takip dosyaları
- `car_prices_clean.csv` — 52 MB ham veri, repoya girmez

---

## Model Detayları

### Feature Listesi (kesin, değişmez)
```python
# Numeric — StandardScaler
num_cols = ["age", "odometer", "condition", "age_x_odo"]

# Ordinal — OrdinalEncoder
ord_cols = ["body", "transmission", "color", "interior"]   # state buradan KALDIRILDI

# Target — TargetEncoder
tgt_cols = ["make", "model", "trim", "state"]   # state buraya TAŞINDI, seller KALDIRILDI

# Türetilen
# age_x_odo = age * odometer → AddInteractions class ile pipeline içinde eklenir

# seller → kaldırıldı. app.py input_dict'te "unknown" göndermeye devam eder;
#          preprocessor bu kolonu sessizce drop eder (remainder='drop').
```

### Pipeline Sırası (kesin, değişmez)
```
input_dict → AddInteractions() [direkt oluşturulur] → preprocessor.pkl → lgbm_tuned.pkl
           → power_transformer.pkl (inverse) → clip(500, None)
           → x1.38 enflasyon katsayısı → kullanıcıya göster
```

**ÖNEMLİ:** `interactions.pkl` artık kullanılmıyor. `AddInteractions` stateless olduğu için
app.py'de `AddInteractions()` direkt oluşturulur, HF'ten yüklenmez.

### pkl Dosyaları (HF Model repo'da)
```
lgbm_tuned.pkl          # LightGBM tuned model (aktif)
lgbm_tuned_prev.pkl     # Bir önceki model (A/B test için, ilk deploydan sonra oluşur)
preprocessor.pkl        # ColumnTransformer (num+ord+tgt)
power_transformer.pkl   # Yeo-Johnson PowerTransformer
car_hierarchy.json      # make→model→trim hiyerarşisi (UI için)
deploy_meta.json        # Son deploy metadata (versiyon, RMSE, tarih)
```

### Model Performansı
- Test RMSE: ~$1,814 (2015 veri bazlı)
- Enflasyon düzeltmesi: x1.38 (2015→2025)
- Eğitim verisi: 552,941 satır, ABD ikinci el araç müzayede kayıtları

### Best Params (lgbm_tuned.pkl için)
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
                                     HF Space yeni modeli çeker (restart_space)

app/app.py değişince:
GitHub push → deploy_app.yml → scripts/push_app_to_space.py → HF Space güncellenir
```

---

## Tüm Fazlar — TAMAMLANDI ✅

### FAZA 1 — Temel Kurulum ✅
- Model eğitildi, pkl dosyaları HF'e yüklendi
- HF Space, HF Dataset, GitHub repo açıldı

### FAZA 2 — Kullanıcı Verisi Toplama ✅
- Feedback + validasyon (price $500–$78K, odometer ≤ 300K mil)
- A/B testing (v_current / v_previous)
- Threshold=10 → `total_new >= THRESHOLD and total_new % THRESHOLD == 0` → workflow_dispatch

### FAZA 3 — Agent Zinciri ✅
- `data_quality_agent.py`: eksik değer, aykırı değer kontrolü (≥30 satırda aktive)
- `performance_agent.py`: original_data + feedback birleştir, yeniden eğit, RMSE karşılaştır
- `deploy_agent.py`: RMSE iyiyse HF push + version tag + Space restart; kötüyse iptal

### FAZA 4 — Drift Detection ✅
- `drift.py`: KS testi, 4 kolon (age, odometer, condition, sellingprice), threshold=0.05
- Drift varsa sadece loglar, pipeline'ı durdurmaz

### FAZA 5 — GitHub Actions ✅
- `retrain.yml`: Her Pazar 02:00 UTC + workflow_dispatch
- `deploy_app.yml`: app/app.py değişince otomatik HF Space push
- GitHub Secrets: `HF_TOKEN`, `HF_USERNAME`

---

## Sabitler

```python
INFLATION_MULTIPLIER = 1.38   # 2015 veri bazı → 2025 fiyatları
THRESHOLD = 10                 # kaç yeni satırda retraining tetiklensin
PRICE_MIN = 500
PRICE_MAX = 78_000
ODOMETER_MAX = 300_000
```

---

## Kesin Kurallar

1. `HF_TOKEN` hiçbir zaman kod içinde olmaz, sadece Secrets'ta.
2. `original_data.parquet` hiçbir zaman değiştirilmez.
3. Yeni model eskisinden iyi değilse deploy yapılmaz, eski model kalır.
4. Kullanıcı feedback'i validasyondan geçmeden dataset'e girmez.
5. Her deploy versiyonlanır (HF tag `vYYYYMMDD`) ve loglanır.
6. `seller` feature listesinden çıkarıldı. app.py input_dict'te `"unknown"` gönderir; preprocessor drop eder.
7. `AddInteractions` class'ı app.py ve agents/*.py dosyalarında tanımlı olmalı (joblib deserialize için). `interactions.pkl` artık yüklenmez — direkt `AddInteractions()` oluşturulur.
8. Enflasyon katsayısı (x1.38) sadece kullanıcıya gösterilen son fiyata uygulanır — eğitimde kullanılmaz.
9. `state` kolonu `ord_cols`'tan `tgt_cols`'a taşındı (bilinmeyen eyaletler için global mean fallback).

---

## GitHub Secrets

```
HF_TOKEN      → HuggingFace write token (model + dataset + space push)
HF_USERNAME   → Osman-Ozcanli
```

## HF Space Secrets

```
HF_TOKEN      → feedback dataset'e yazmak için
GITHUB_TOKEN  → Harici PAT (workflow:write scope) — GitHub Actions'ın otomatik token'ı değil
```
