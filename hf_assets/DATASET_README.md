---
license: mit
language:
  - en
tags:
  - tabular-regression
  - automotive
  - mlops
  - feedback-loop
size_categories:
  - n<1K
---

# Car Price Prediction — User Feedback

Validated user-submitted sale prices for the
[`Osman-Ozcanli/car_price_prediction`](https://huggingface.co/Osman-Ozcanli/car_price_prediction)
model. This dataset is the input side of an MLOps feedback loop: every row
here was accepted by a three-layer deterministic validator before being
written.

The full pipeline lives at
[`Osman-Ozcanli/car-price-mlops`](https://github.com/Osman-Ozcanli/car-price-mlops).

## How rows get here

1. User loads the [HF Space](https://huggingface.co/spaces/Osman-Ozcanli/car_price_prediction_space)
   and requests a price estimate.
2. User clicks "Yes" on the "was this useful?" prompt and enters the actual
   sale price.
3. The submission passes through three deterministic validation layers
   (`common/validation.py` in the source repo):
   - **Layer 1 — schema/range:** `$500 ≤ price ≤ $78,000`, `odometer ≤ 300,000`.
   - **Layer 2 — prediction sanity:** ratio `price / predicted` must lie in
     `[0.5, 2.0]` AND absolute deviation must be `≤ $15,000`.
   - **Layer 3 — meta-auditor:** if any layer rejects, the row is dropped
     with a structured reason code; only accepted rows reach this dataset.
4. Accepted row is appended to `feedback.parquet` here.

## Schema

| Column | Type | Source |
|---|---|---|
| `make` | str | UI selectbox |
| `model` | str | UI selectbox (filtered by make) |
| `trim` | str | UI selectbox (filtered by make + model) |
| `body` | str | UI selectbox |
| `transmission` | str | UI selectbox |
| `state` | str | UI selectbox (US state, lowercase) |
| `color`, `interior` | str | UI selectbox |
| `age` | int | UI numeric input (years) |
| `odometer` | int | UI numeric input (miles) |
| `condition` | float | UI slider (1.0–5.0, step 0.1) |
| `seller` | str | Always `"unknown"` — kept for backwards compatibility, dropped during preprocessing |
| `sellingprice` | int | User-entered actual sale price ($) |
| `model_version` | str | `v_current` or `v_previous` — A/B label assigned per session |
| `timestamp` | str | ISO-8601 UTC submission time |

## How retraining uses this data

Every training run (Sunday cron OR threshold-triggered) reads this dataset
in full, concatenates it with the immutable `original_data.parquet` (552K
auction records), and trains a candidate model. Past feedback is **never
discarded** — the dataset grows monotonically.

A candidate is deployed only if its RMSE beats the currently active model on
a shared validation slice (`agents/performance_agent.py`).

## Validation rules in code

```python
PRICE_MIN = 500
PRICE_MAX = 78_000
ODOMETER_MAX = 300_000
FEEDBACK_RATIO_MIN = 0.5
FEEDBACK_RATIO_MAX = 2.0
FEEDBACK_ABS_DEVIATION_MAX = 15_000
```

These constants live in
[`common/constants.py`](https://github.com/Osman-Ozcanli/car-price-mlops/blob/main/common/constants.py).

## Limitations

- Self-selected sample: only users who chose to submit a "real" price.
- A/B labels are assigned per session, so a single user's submissions are
  consistently labelled — useful for variance reduction, biases per-user
  preferences into the analysis.
- No identity, location precision, or VIN — dataset is intentionally minimal.

## License

MIT.
