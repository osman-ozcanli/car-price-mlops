# car-price-mlops

## 🚗 Car Price Prediction — MLOps Pipeline

End-to-end MLOps pipeline for used car price prediction in the US market.  
Trained on 550K+ auction records. Automatically retrains as new user feedback arrives.

---

## Live Demo

👉 [HuggingFace Space](https://huggingface.co/spaces/Osman-Ozcanli/car_price_prediction_space)

---

## Architecture

```
User → Predict → Feedback → Validation → HF Dataset
                                               ↓
                                     GitHub Actions (Sunday 02:00)
                                               ↓
                                    Agent 1: Data Quality Check
                                               ↓
                                    Agent 2: Train + Compare
                                               ↓
                                    Agent 3: Deploy or Abort
                                               ↓
                                    HF Space pulls new model
                                               ↓
                                          User → ...
```

---

## Model

- **Algorithm:** LightGBM (Optuna-tuned, 50 trials)
- **Target:** Used car selling price ($500–$78,000)
- **Test RMSE:** ~$1,814
- **Features:** make, model, trim, body, transmission, state, color, interior, age, odometer, condition
- **Encoding:** TargetEncoder (make/model/trim/seller) + OrdinalEncoder + StandardScaler
- **Target transform:** Yeo-Johnson → inverse on predict
- **Inflation adjustment:** x1.38 multiplier (2015 data → 2025 prices)

---

## Project Structure

```
car-price-mlops/
├── app/
│   └── app.py                  # Streamlit UI + feedback + A/B testing
├── training/
│   ├── train.py                # Orchestrates agent chain
│   ├── drift.py                # KS-test drift detection
│   └── original_data.parquet  # Base training data (never modified)
├── agents/
│   ├── data_quality_agent.py  # Agent 1: null + outlier check
│   ├── performance_agent.py   # Agent 2: train + compare RMSE
│   └── deploy_agent.py        # Agent 3: push to HF or abort
├── .github/
│   └── workflows/
│       └── retrain.yml        # Scheduled retraining (Sunday 02:00 UTC)
├── requirements.txt
└── README.md
```

---

## Retraining Pipeline

Triggered automatically when:
- **10+ new feedback rows** collected in HF Dataset (via `workflow_dispatch`)
- **Every Sunday at 02:00 UTC** (scheduled)

### Agent Chain
| Agent | Role | On Failure |
|-------|------|------------|
| Data Quality | Null + outlier check | Pipeline stops |
| Performance | Train new model, compare RMSE | Deploy cancelled |
| Deploy | Push to HF Model Hub with version tag | Logs error |

### Safety Rules
- New model only deployed if RMSE < current model RMSE
- Original training data never modified
- All feedback validated before entering dataset (price $500–$78K, odometer ≤ 300K mi)
- Drift detection (KS-test) runs before training

---

## HuggingFace Resources

| Resource | Link |
|----------|------|
| Model | `Osman-Ozcanli/car_price_prediction` |
| Dataset | `Osman-Ozcanli/car_price_prediction_feedback` |
| Space | `Osman-Ozcanli/car_price_prediction_space` |

---

## Tech Stack

| Need | Tool |
|------|------|
| Frontend | Streamlit |
| Model | LightGBM |
| Hyperparameter tuning | Optuna |
| Model deploy | HuggingFace Model Hub |
| Data storage | HuggingFace Dataset (Parquet) |
| Drift detection | scipy (KS-test) |
| Scheduler | GitHub Actions |
| Versioning | HF model tags |

---

## Secrets Required

**GitHub Secrets** (training pipeline için):
```
HF_TOKEN       → HuggingFace write token (model + dataset push)
HF_USERNAME    → Osman-Ozcanli
```

**HuggingFace Space Secrets** (app.py için):
```
HF_TOKEN       → feedback dataset'e yazmak için
GITHUB_TOKEN   → Personal Access Token (workflow:write scope), GitHub Actions tetiklemek için
                 NOT: GitHub Actions'ın otomatik sağladığı GITHUB_TOKEN değildir; harici PAT gerekir.
```
