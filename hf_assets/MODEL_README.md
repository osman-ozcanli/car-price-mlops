---
license: mit
language:
  - en
tags:
  - tabular-regression
  - lightgbm
  - mlops
  - automotive
  - feedback-loop
library_name: scikit-learn
pipeline_tag: tabular-regression
---

# Car Price Prediction (US Used-Car Auction)

LightGBM regression model that estimates the sale price of a used vehicle
from a small set of attributes. Continuously retrained from validated user
feedback as part of the
[`car-price-mlops`](https://github.com/Osman-Ozcanli/car-price-mlops) pipeline.

## Intended use

- Rough estimation of fair market value for a US used vehicle.
- Demonstration of an end-to-end MLOps loop (feedback → validation →
  retraining → conditional deploy) — the model is the artifact, the system
  around it is the point.

**Not intended for:** legally binding valuations, insurance settlements, or
markets outside the United States.

## Inference pipeline (frozen)

```
input_dict → AddInteractions (age * odometer)
           → preprocessor (StandardScaler + OrdinalEncoder + TargetEncoder)
           → LightGBM model
           → PowerTransformer.inverse_transform (Yeo-Johnson)
           → clip(500, None)
           → × 1.38 inflation multiplier (2015 → 2025)
           → estimated price
```

## Features

| Group | Columns | Encoding |
|---|---|---|
| Numeric | `age`, `odometer`, `condition`, `age_x_odo` | StandardScaler |
| Ordinal | `body`, `transmission`, `color`, `interior` | OrdinalEncoder (unknown → -1) |
| Target-encoded | `make`, `model`, `trim`, `state` | TargetEncoder (global-mean fallback) |

`age_x_odo = age × odometer` is added at runtime by the `AddInteractions`
transformer. `seller` is intentionally absent — it was dropped after evidence
that the production app could not reliably populate it.

## Artifacts in this repo

| File | Purpose |
|---|---|
| `lgbm_tuned.pkl` | Active LightGBM model |
| `lgbm_tuned_prev.pkl` | Previous model (A/B labelling + rollback target) |
| `preprocessor.pkl` | ColumnTransformer (num + ord + tgt) |
| `power_transformer.pkl` | Yeo-Johnson PowerTransformer for the target |
| `car_hierarchy.json` | make → model → trim taxonomy used by the UI |
| `deploy_meta.json` | Last deploy metadata (version tag, RMSEs, timestamp) |

`AddInteractions` is **stateless** and is constructed at inference time —
it is not pickled or downloaded.

## Training data

- 552,941 US used-car auction records (2015 vintage, public Kaggle dataset).
- 90/10 train/validation split; the same split is used to compare a candidate
  model against the active one before deploy.

## Performance

- Test RMSE: **~$1,814** on the 2015 validation slice (raw, pre-inflation).
- Inflation adjustment of ×1.38 is applied at display time only — never
  during training.

## Best hyperparameters

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

Tuned with Optuna over 50 trials.

## Limitations

- Trained on 2015-era data; market structure (EV mix, post-pandemic supply
  shocks) is partially captured by the inflation multiplier but not learned.
- Sparse states (e.g. AK, ND, WY) fall back to the global mean via
  TargetEncoder — predictions there are coarse.
- Body type `Pickup` is not represented in training and is excluded from the
  UI selector.
- The model is recalibrated only when feedback accumulates — there is a lag
  between market shifts and updated parameters.

## Versioning and rollback

Each successful deploy is tagged `vYYYYMMDD` on this repo. The previous
active model is snapshotted to `lgbm_tuned_prev.pkl` before each deploy,
enabling A/B labelling at inference time and a rollback target.

## License

MIT. The training data is from a public Kaggle dataset; check the original
source for any attribution requirements before commercial use.
