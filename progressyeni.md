# Progress — P0 Refactor (2026-04-30)

Bu dosya, `CLAUDECODE_ELESTIRILERI.md` ve `TAMAMENKENDICIKARDIGIM NOTLAR.txt`
dosyalarındaki bulgular doğrultusunda yapılan P0 değişikliklerin kaydıdır.

Strateji: **Plan A — kademeli refactor** (mevcut çalışan sistem korundu, üzerine
İngilizceleştirme + temizlik). LLM-tabanlı agent değil, **deterministik
istatistik kuralları**.

---

## P0 — Tamamlandı

| # | Değişiklik | Dosya |
|---|---|---|
| 1 | `AddInteractions` tek noktada (4 dosyadaki kopyalar silindi) | **yeni** `common/transformers.py` |
| 2 | Sabitler tek noktada (`THRESHOLD`, `INFLATION`, `PRICE_*` vb.) | **yeni** `common/constants.py` |
| 3 | 3 katmanlı deterministik validasyon (madde 1) | **yeni** `common/validation.py` |
| 4 | `interactions.pkl` deploy'dan tamamen çıkarıldı (A3) | `agents/deploy_agent.py` |
| 5 | `seller="unknown"` hack'i silindi, ilk migrate için "default deploy" yolu (B2) | `agents/performance_agent.py` |
| 6 | `app.py` → tamamen İngilizce, `common/`'dan import, `model_version` UI'dan gizlendi | `app/app.py` |
| 7 | "Yeterli veri birikti" mesajı silindi (madde 14) | `app/app.py` |
| 8 | Tahmin band'ı genişletildi: ±5% veya ±$500 (madde 4) | `app/app.py` |
| 9 | `is_valid` 5x→2x daraltıldı + $15K abs deviation cap (madde 13) | `common/validation.py` |
| 10 | HF Space keep-alive (her gün 06:00 UTC ping) (madde 5) | **yeni** `.github/workflows/keepalive.yml` |
| 11 | `push_app_to_space.py` artık `common/`'u da Space'e push ediyor | `scripts/push_app_to_space.py` |
| 12 | `deploy_app.yml` `common/**` değiştiğinde de tetikleniyor | `.github/workflows/deploy_app.yml` |
| 13 | Tüm pipeline çıktıları İngilizce (drift, train, agents) | `training/*.py`, `agents/*.py` |
| 14 | Workflow YAML'larındaki Türkçe step isimleri İngilizceye çevrildi | `retrain.yml`, `deploy_app.yml` |
| 15 | `.gitignore`'a kişisel not pattern'leri eklendi (D4) | `.gitignore` |

---

## 3 Katmanlı Agent Validasyonu — Mimari (madde 1)

`common/validation.py`:

- **Layer 1 — `layer1_schema`:** Sıkı sınır kontrolü ($500–$78K, ≤300K mil).
  UI bypass'a karşı dataset-boundary fence.
- **Layer 2 — `layer2_prediction_sanity`:** Tahmine göre sanity:
  oran 0.5x–2.0x **VE** mutlak fark ≤ $15K. Örnek senaryo:
  $19K tahmin → $35K kullanıcı girişi (ratio 1.84) **artık reddedilir**.
- **Layer 3 — `validate_feedback` meta-auditor:** Tüm katmanları sırayla
  çalıştırır, ilk reddi yakalar, structured reason code (`L1`/`L2` + kod) döner.
  UI'da kullanıcıya neden gösterilir.

LLM yok, ekstra maliyet yok. Tamamen deterministik.

---

## Soru-Cevap Raporları

- **Madde 5 (HF Space uyku):** Premium gerekmez. `keepalive.yml` günlük cron +
  curl ile çözüldü. Manuel müdahale gerekmiyor.
- **Madde 6 (otomatik commit/push):** GitHub Actions yeni model üretince
  GitHub repo'ya commit atmıyor — direkt HF Model repo'ya push ediyor. Manuel
  bir şey yapmana gerek yok.
- **Madde 7 (token/üyelik):** GitHub Actions public repo'da limitsiz dakika.
  HF token'ı süresiz. Premium gerekmez.
- **Madde 11 (model_version hep v_previous):** Bug değil — `random.choice`
  session başına sticky. Artık UI'dan tamamen gizli, sadece feedback
  parquet'te A/B analizi için kayıtlı.
- **Madde 12 (dataset private):** Yapılabilir, model sağlığına etkisi yok.
  Tavsiye edilir — kullanıcı verisi private olsun. HF Settings → Make private.
- **Madde 8 (HF model sayfası 500-78K):** Sıkıntı yok, doğru bilgi.
- **Madde 9 (HF README'leri boş):** P1'de yazılacak.
- **Madde 10 (HF dataset gecikmesi):** HF Parquet cache'i, platform davranışı.
- **Madde 15 (eski feedback'ler çöp mü?):** Çöp olmuyor. `train.py` her
  seferinde `original_data + tüm birikmiş feedback` ile çalışıyor (cumulative).

---

## Threshold Politikası (madde 15)

`common/constants.py`:

```python
THRESHOLD_DEMO = 10    # şu an aktif — gözlemlenebilirlik için
THRESHOLD_PROD = 100   # gerçek kullanımda istatistik anlamlı batch
THRESHOLD = THRESHOLD_DEMO
```

**Karar:** Test bittiğinde `THRESHOLD = THRESHOLD_PROD`'a çevir.
Justification: 552K satıra 10 satır eklemek modeli ölçülebilir biçimde
değiştirmiyor; her seferinde 3298-estimator LightGBM eğitmek boşa compute.

---

## Test Sırası

1. **Lokal test:** `streamlit run app/app.py` — yükleniyor mu, tahmin yapıyor mu?
2. **Push:** `git push` — `deploy_app.yml` Space'e otomatik push eder.
3. **Keep-alive workflow'u manuel tetikle** ilk seferde: GitHub → Actions →
   "Keep HF Space Awake" → Run workflow.
4. **Madde 13 senaryosu:** $19K tahmin → $35K feedback gir → "Feedback rejected
   (L2/RATIO_MISMATCH)" gelmeli.

---

## Sırada — P1 (yarım gün)

- `tests/` + `ci.yml` (pytest + ruff)
- HF Model + Dataset + Space README'leri (İngilizce, portföy kalitesinde)
- `requirements.txt` tam pin (lightgbm, numpy, joblib pickle uyumluluğu için)
- `CLAUDE.md` İngilizceye çevir + yeni yapıyı yansıt
- `README.md` upgrade (mimari diagram + reproduce + workflow trigger)

## P2 (fırsatta)

- `scripts/ab_report.py` (model_version × liked analizi)
- `runs/{timestamp}.json` retrain log + `deploy_meta.json` schema
- `scripts/rollback.py`
- Pre-commit hook (ruff + büyük dosya guard)
