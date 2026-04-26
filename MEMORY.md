# MEMORY.md — Proje Durumu

## Tamamlananlar

### Eğitim (lokal)
- 01_cleaning.py → car_prices_clean.csv (552,941 satır)
- 02_eda.py → eda_0*.png
- 03_modeling.py → lgbm_baseline.pkl
- 04_tuning.py → lgbm_tuned.pkl, preprocessor.pkl, interactions.pkl, power_transformer.pkl, best_params.json
- 05_evaluation.py → eval_0*.png, SHAP analizi
- Test RMSE: ~$1,814 | MAE: ~$1,054 | R²: 0.966

### HuggingFace
- Model repo: `Osman-Ozcanli/car_price_prediction`
  - lgbm_tuned.pkl ✅
  - preprocessor.pkl ✅
  - interactions.pkl ✅ (suspicious flag — normal, görmezden gel)
  - power_transformer.pkl ✅
  - car_hierarchy.json ✅
- Dataset repo: `Osman-Ozcanli/car_price_prediction_feedback` ✅ (boş, feedback bekliyor)
- Space: `Osman-Ozcanli/car_price_prediction_space` ✅ (çalışıyor)

### GitHub
- Repo: `Osman-Ozcanli/car-price-mlops` ✅
- app.py kök dizine koyuldu ✅
- README.md yazıldı ✅
- Klasör yapısı oluşturuldu: app/, agents/, training/, .github/workflows/ ✅

### app.py (FAZA 2) ✅
- Hiyerarşik marka/model/trim seçimi (selectbox, listede olmayanlar seçilemiyor)
- x1.38 enflasyon katsayısı uygulandı
- Feedback + validasyon (price $500-$78K, odometer ≤ 300K mil)
- A/B testing (session_state ile v_current / v_previous)
- Threshold=10 → GitHub Actions workflow_dispatch tetikleme

---

## Bir Sonraki Adım — Claude Code'da yapılacaklar (FAZA 3-5)

### Sıra
1. `training/original_data.parquet` oluştur
   - car_prices_clean.csv'den üret, training/ klasörüne koy
   - HF Dataset'e de yükle (retraining sırasında çekilecek)

2. `agents/data_quality_agent.py` yaz

3. `agents/performance_agent.py` yaz
   - best_params.json'daki parametreleri kullan
   - original_data + feedback birleştir, eğit, karşılaştır

4. `agents/deploy_agent.py` yaz
   - HF'e push + version tag

5. `training/drift.py` yaz
   - KS testi, DRIFT_FEATURES = ["age", "odometer", "condition", "sellingprice"]

6. `training/train.py` yaz
   - Tüm zinciri orchestrate et

7. `.github/workflows/retrain.yml` yaz

8. GitHub Secrets ekle: HF_TOKEN, HF_USERNAME

9. Push et, manuel workflow_dispatch ile test et

---

## Kritik Hatırlatmalar

- `AddInteractions` class'ı her .py dosyasında tanımlı olmalı
- `seller` her zaman "unknown"
- `original_data.parquet` asla değiştirilmez
- Enflasyon katsayısı (x1.38) sadece app.py'de, eğitimde kullanılmaz
- Yeni model eskiden kötüyse deploy yapılmaz
