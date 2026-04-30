# CLAUDE.md — Car Price Prediction MLOps Pipeline

## Overview
End-to-end MLOps pipeline that takes vehicle attributes from a user, returns a
price estimate, collects validated feedback, and retrains itself when enough
new data has accumulated.

---

## HuggingFace Resources

| Resource | Repo |
|---|---|
| Model | `Osman-Ozcanli/car_price_prediction` |
| Feedback dataset | `Osman-Ozcanli/car_price_prediction_feedback` |
| Space | `Osman-Ozcanli/car_price_prediction_space` |

---

## Repo Layout

```
car-price-mlops/
│
├── app/
│   ├── app.py                  # Streamlit UI (prediction + feedback + A/B)
│   └── car_hierarchy.json      # make → model → trim hierarchy (UI source)
│
├── common/                     # Shared package — single source of truth
│   ├── transformers.py         # AddInteractions (stateless)
│   ├── constants.py            # THRESHOLD, INFLATION, PRICE_*, repo IDs
│   └── validation.py           # 3-layer deterministic feedback validator
│
├── training/
│   ├── train.py                # Orchestrates the agent chain
│   ├── drift.py                # KS-test drift detection (logging only)
│   └── original_data.parquet   # Base training data (NEVER modified)
│
├── agents/
│   ├── data_quality_agent.py   # Agent 1: schema + outlier check
│   ├── performance_agent.py    # Agent 2: train + RMSE compare
│   └── deploy_agent.py         # Agent 3: conditional publish
│
├── tests/                      # pytest suite
│   ├── test_validation.py      # 3-layer validator coverage
│   ├── test_transformers.py    # AddInteractions correctness
│   └── test_constants.py       # Sanity guards on bounds
│
├── scripts/
│   └── push_app_to_space.py    # Mirrors app.py + common/ to HF Space
│
├── .github/workflows/
│   ├── retrain.yml             # Sunday 02:00 UTC + workflow_dispatch
│   ├── deploy_app.yml          # Pushes to HF Space when app/common changes
│   ├── keepalive.yml           # Daily 06:00 UTC ping (prevents 48h sleep)
│   └── ci.yml                  # ruff + pytest on push & PR
│
├── pyproject.toml              # ruff + pytest config
├── requirements.txt
├── README.md
├── CLAUDE.md
└── progressyeni.md             # Live roadmap (P0 done, P1/P2/P3 queues)
```

**Local-only files (gitignored):**
- `MEMORY.md`, `PROGRESS.md` — legacy session notes
- `progressyeni.md` is committed (live roadmap)
- `car_prices_clean.csv` — 52 MB raw data
- `*ELESTIRILERI*.md`, `TAMAMENKENDICIKARDIGIM*.txt` — personal critique drafts

---

## Model

### Feature schema (frozen)
```python
# Numeric — StandardScaler
NUM_COLS = ["age", "odometer", "condition", "age_x_odo"]

# Ordinal — OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
ORD_COLS = ["body", "transmission", "color", "interior"]

# Target — TargetEncoder (global-mean fallback for unseen categories)
TGT_COLS = ["make", "model", "trim", "state"]

# age_x_odo = age * odometer  — added by common.transformers.AddInteractions
# seller — DROPPED. app.py still sends "unknown"; preprocessor's
#          remainder='drop' silently discards it.
```

### Inference pipeline (frozen)
```
input_dict → AddInteractions() → preprocessor.pkl → lgbm_tuned.pkl
           → power_transformer.pkl (inverse) → clip(500, None)
           → × INFLATION_MULTIPLIER (1.38) → user
```

`interactions.pkl` is **not** loaded. `AddInteractions` is stateless and
constructed fresh at inference time — this avoids pickle module-path coupling.

### Artifacts on HF Model repo
```
lgbm_tuned.pkl          # Active model
lgbm_tuned_prev.pkl     # Previous model (A/B + rollback target)
preprocessor.pkl        # ColumnTransformer (num + ord + tgt)
power_transformer.pkl   # Yeo-Johnson PowerTransformer
car_hierarchy.json      # make→model→trim taxonomy for the UI
deploy_meta.json        # Last deploy metadata (version, RMSE, timestamp)
```

### Performance
- Test RMSE: ~$1,814 (validation slice on 2015-vintage data)
- Training data: 552,941 rows, US used-car auction records
- Inflation adjustment: ×1.38 applied at display time only (never during training)

### Best params (lgbm_tuned.pkl)
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

## Architecture — full loop

```
User → Predict → Like? → 3-layer validator → HF Dataset (feedback)
                                                    ↓
                                  GitHub Actions
                                  (Sunday 02:00 UTC OR threshold trigger)
                                                    ↓
                                       Agent 1: data quality
                                                    ↓
                                       drift.py: KS test (log only)
                                                    ↓
                                       Agent 2: train + RMSE compare
                                                    ↓
                                       Agent 3: deploy or skip
                                                    ↓
                                       HF Space restart_space()

When app/app.py or common/** changes:
  GitHub push → deploy_app.yml → scripts/push_app_to_space.py → HF Space
```

---

## Three-layer feedback validator

`common/validation.py` enforces deterministic, statistically-grounded checks
before any user feedback enters the dataset:

- **Layer 1 — schema/range:** price within [$500, $78K], odometer ≤ 300K mi.
  This is the dataset boundary; UI restricts these but a determined user can
  bypass.
- **Layer 2 — prediction sanity:** ratio in [0.5x, 2.0x] AND absolute deviation
  ≤ $15K vs the model's prediction. Catches "$19K predicted, $35K typed" cases
  that the original 5x band let through.
- **Layer 3 — meta-auditor:** runs Layers 1–2 in sequence, returns the first
  rejection with a structured reason code (`L1`/`L2` + code) suitable for
  audit logging.

No LLM calls. All checks deterministic.

---

## Constants

`common/constants.py`:
```python
INFLATION_MULTIPLIER = 1.38     # 2015 → 2025 calibration; review annually
PRICE_MIN, PRICE_MAX = 500, 78_000
ODOMETER_MAX = 300_000

FEEDBACK_RATIO_MIN, FEEDBACK_RATIO_MAX = 0.5, 2.0
FEEDBACK_ABS_DEVIATION_MAX = 15_000

THRESHOLD_DEMO = 10              # currently active, observable end-to-end
THRESHOLD_PROD = 100             # statistically meaningful batch
THRESHOLD = THRESHOLD_DEMO       # flip to PROD once demo phase ends
```

---

## Hard rules

1. `HF_TOKEN` lives only in GitHub/HF Secrets — never in code.
2. `original_data.parquet` is never modified.
3. A new model is deployed only if its RMSE beats the active model's. The
   first migration after a schema change is the only exception (default deploy).
4. Feedback enters the dataset only after passing all three validation layers.
5. Every deploy is tagged `vYYYYMMDD` on the HF Model repo.
6. `seller` is dropped. `app.py` still sends `"unknown"`; the preprocessor's
   `remainder='drop'` discards it.
7. `AddInteractions` lives **only** in `common/transformers.py`. All other
   modules import from there — no copies. `interactions.pkl` is not deployed.
8. The ×1.38 inflation multiplier is applied only at display time, never
   during training.
9. `state` is in `TGT_COLS` (TargetEncoder), not `ORD_COLS` — global-mean
   fallback handles unseen states.
10. All new code is in English — UI strings, log messages, comments, READMEs.

---

## Secrets

**GitHub Secrets** (used by `retrain.yml`, `deploy_app.yml`, `keepalive.yml`):
```
HF_TOKEN      → HF write scope (model + dataset + space push)
HF_USERNAME   → Osman-Ozcanli
```

**HF Space Secrets** (used by `app.py` at runtime):
```
HF_TOKEN      → write the feedback dataset
GITHUB_TOKEN  → external PAT with workflow:write scope (NOT the auto-issued
                Actions token) — used to dispatch retrain.yml from the Space
```
