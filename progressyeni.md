# progressyeni.md — Canlı Yol Haritası ve Durum

> Bu dosya **canlı kayıt**: her oturumda buradan okuyup devam et, her değişiklikten
> sonra güncelle. Önceki `PROGRESS.md` arşiv (oturum 1-4 tarihi).
>
> Strateji: **Plan A — kademeli refactor.** LLM yok, deterministik istatistik agent'ları.
> Hedef: çalışan MLOps demo'sunu junior → mid-level portföy seviyesine taşımak.
>
> Son güncelleme: 2026-04-30 — P0 + P1.1 + P1.2 + P1.3 + P1.4 tamamlandı

---

## Sabit Referanslar

- **GitHub:** `https://github.com/Osman-Ozcanli/car-price-mlops`
- **HF Model:** `Osman-Ozcanli/car_price_prediction`
- **HF Dataset:** `Osman-Ozcanli/car_price_prediction_feedback`
- **HF Space:** `Osman-Ozcanli/car_price_prediction_space`
- **Detaylı eleştiri kaynağı:** `CLAUDECODE_ELESTIRILERI.md` (lokal, gitignore)
- **Kullanıcı notları:** `TAMAMENKENDICIKARDIGIM NOTLAR.txt` (lokal, gitignore)

---

## P0 — Tamamlandı ✅ (commit `2e3fb26`, 2026-04-30)

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
| 16 | Kök dizindeki gereksiz `push_app_to_space.py` silindi | (silindi) |

### 3 Katmanlı Agent Validasyonu — Mimari (madde 1)

`common/validation.py`:

- **Layer 1 — `layer1_schema`:** Sıkı sınır kontrolü ($500–$78K, ≤300K mil).
  UI bypass'a karşı dataset-boundary fence.
- **Layer 2 — `layer2_prediction_sanity`:** Tahmine göre sanity:
  oran 0.5x–2.0x **VE** mutlak fark ≤ $15K. $19K → $35K (ratio 1.84) artık reddedilir.
- **Layer 3 — `validate_feedback` meta-auditor:** Tüm katmanları sırayla
  çalıştırır, ilk reddi yakalar, structured reason code (`L1`/`L2` + kod) döner.
  UI'da kullanıcıya neden gösterilir.

### Soru-Cevap Raporları (kalıcı kayıt)

- **Madde 5 (HF Space uyku):** Premium gerekmez, keepalive.yml çözdü.
- **Madde 6 (otomatik commit/push):** GitHub repo'ya commit atılmıyor, direkt HF'e push.
- **Madde 7 (token/üyelik):** GitHub Actions public repo'da limitsiz, HF token süresiz, ücretsiz tier yeterli.
- **Madde 11 (model_version hep v_previous):** Bug değil, session sticky. UI'dan gizlendi.
- **Madde 12 (dataset private):** Yapılabilir, model sağlığına etkisi yok. Tavsiye edilir.
- **Madde 8 (HF model sayfası 500-78K):** Doğru bilgi.
- **Madde 9 (HF README'leri boş):** P1'de yazılacak.
- **Madde 10 (HF dataset gecikmesi):** Platform davranışı, kodda çözüm yok.
- **Madde 15 (eski feedback'ler çöp mü?):** Cumulative kullanılıyor, çöp olmuyor.

### Threshold Politikası

`common/constants.py`:
```python
THRESHOLD_DEMO = 10    # şu an aktif
THRESHOLD_PROD = 100   # production için önerilen
THRESHOLD = THRESHOLD_DEMO
```
Test bittiğinde `THRESHOLD = THRESHOLD_PROD`'a çevir.

---

## P1 — Devam ediyor 🟡

### P1.1 — Test altyapısı + CI (CLAUDECODE C4 + C5) ✅
- [x] `tests/__init__.py` oluşturuldu
- [x] `tests/test_validation.py` — 3 katman accept/reject (canonical $19K→$35K dahil), 14 test
- [x] `tests/test_transformers.py` — `AddInteractions` `age × odometer` + immutability + statelessness
- [x] `tests/test_constants.py` — sabit drift'ine karşı sanity guard
- [x] `.github/workflows/ci.yml` — push + PR'da ruff + pytest
- [x] `pyproject.toml` — ruff + pytest config (line-length=100, isort)
- [x] **20/20 test geçti, ruff temiz** (lokal doğrulama)
- [Not] `test_pipeline.py` (HF artifact ile predict smoke test) atlandı — CI'da HF_TOKEN olmadan çalışmaz; P2'de mock'lu bir alternatif düşünülebilir.

### P1.2 — Tam İngilizceleştirme (madde 2-3, kalanlar) ✅
- [x] `CLAUDE.md` tamamen İngilizce + yeni `common/` yapısı + 3-katman validator + tüm sabitler dokümante
- [x] `README.md` tamamen İngilizce + **mermaid mimari diagram** + reproduce adımları + workflow trigger + CI rozet
- [Not] `progressyeni.md` Türkçe kaldı (kullanıcının çalışma dosyası).

### P1.3 — HF README'leri (madde 8-9) ✅
- [x] `hf_assets/MODEL_README.md` — model card (training data, metrics, intended use, limitations, best params)
- [x] `hf_assets/DATASET_README.md` — schema, 3-katman validation rules, retrain mekanizması
- [x] `hf_assets/SPACE_README.md` — Streamlit Space metadata (frontmatter) + kullanım + privacy notu
- [x] `scripts/push_hf_readmes.py` — manuel upload helper (`python scripts/push_hf_readmes.py`)
- [Aksiyon] Kullanıcı: `HF_TOKEN` env ile bir kere `python scripts/push_hf_readmes.py` çalıştırması gerekli.

### P1.4 — Dependency pinning (CLAUDECODE D5 + E1) ✅
- [x] `requirements.txt` tam pin: pandas, numpy, lightgbm, joblib, category_encoders, scipy hepsi `==`
- [x] sklearn zaten pin'liydi
- [Not] `train_env.txt` ayrıca HF Model'e P2'de eklenecek (deploy_meta.json schema güncellemesinin parçası olarak).

### P1.5 — Dataset privacy (madde 12) ⏳
- [ ] HF Dataset repo'sunu private'a çevir (manuel, HF UI → Settings → "Make this dataset private")
- [ ] `train.py` private repo erişimini doğrula (HF_TOKEN zaten kullanıyor, beklenen sorun yok)
- [Aksiyon] Bu adım **kullanıcı kararı** — kullanıcı istediğinde tek tıkla yapacak.

---

## P2 — Fırsatta (1 gün) 🟢

### P2.1 — A/B raporu (CLAUDECODE B3)
- [ ] `scripts/ab_report.py` — feedback dataset'i çek, `model_version × |feedback − prediction|` özet tablosu üret
- [ ] Workflow'a günlük cron eklensin → çıktıyı `reports/ab_YYYYMMDD.md` olarak HF Dataset'e push
- [ ] README'ye markdown çıktısının link'i

### P2.2 — Retrain logging + deploy_meta schema (CLAUDECODE C3 + E2)
- [ ] Her retrain run'ı sonunda `runs/{timestamp}.json` yaz: `{rmse_old, rmse_new, drift_flags, deployed: bool, n_feedback, sklearn_version, lightgbm_version}`
- [ ] HF Dataset olarak push (örn. `Osman-Ozcanli/car_price_prediction_runs`)
- [ ] `deploy_meta.json` schema'sını sabitle ve dokümante et

### P2.3 — Rollback yolu (CLAUDECODE E3)
- [ ] `scripts/rollback.py` — HF'te `lgbm_tuned_prev.pkl` → `lgbm_tuned.pkl` rename + Space restart
- [ ] README'ye "Acil rollback" bölümü

### P2.4 — Streamlit error handling (CLAUDECODE E4)
- [ ] `load_models()` etrafında try/except + `st.error("Model temporarily unavailable, please retry in a few minutes")`

### P2.5 — Pre-commit hook (CLAUDECODE E6)
- [ ] `.pre-commit-config.yaml`: ruff + check-added-large-files + nbstripout
- [ ] `pre-commit install` ve `pre-commit run --all-files` doğrulaması

### P2.6 — Drift bayrağı deploy_meta'ya (CLAUDECODE B4)
- [ ] `train.py` `detect_drift` çıktısını `deploy_meta.json`'a yansıtsın

---

## P3 — Nice-to-have 🔵

- [ ] CPI/enflasyon dinamik (CLAUDECODE C2) — yıllık manuel güncelleme yeterli
- [ ] Feedback flag mekanizması (market median ± 3σ, CLAUDECODE C6) — Layer 2.5 olarak `validation.py`'ye
- [ ] `THRESHOLD = THRESHOLD_PROD`'a geçiş (test bittikten sonra)
- [ ] PROGRESS.md → PROGRESS_archive.md yeniden adlandırma (kullanıcı kararı)

---

## Kalıcı Kurallar (CLAUDE.md ile uyumlu)

1. `HF_TOKEN` kod içinde olmaz — sadece Secrets.
2. `original_data.parquet` asla değiştirilmez.
3. Yeni model eskisinden kötüyse deploy yapılmaz; ilk migration'da default deploy.
4. Feedback 3 katmanlı validasyondan geçmeden dataset'e girmez.
5. Her deploy `vYYYYMMDD` tag ile versiyonlanır.
6. `seller` kaldırıldı; preprocessor sessizce drop eder.
7. `AddInteractions` **sadece** `common/transformers.py`'de yaşar; her dosya oradan import eder.
8. `interactions.pkl` artık deploy edilmiyor — stateless transformer pickle'lanmaz.
9. x1.38 enflasyon katsayısı sadece kullanıcıya gösterilen son fiyatta, eğitimde değil.
10. `state` `tgt_cols`'ta (TargetEncoder), `ord_cols`'ta değil.
11. Tüm yeni kod İngilizce. UI, log, README, comment — istisnasız.

---

## Test Sırası (her commit sonrası)

1. **Lokal:** `streamlit run app/app.py` — yükleniyor, tahmin yapıyor, feedback gönderiyor.
2. **Push:** `git push` — `deploy_app.yml` Space'e otomatik push eder.
3. **Space test:** Birkaç dakika bekle, HF Space'i aç, tahmin + feedback dene.
4. **Validation kanıt:** $19K tahmin → $35K feedback gir → `L2/RATIO_MISMATCH` reddi gelmeli.
5. **GitHub Actions sekmesi:** Hatalı workflow var mı kontrol et.
