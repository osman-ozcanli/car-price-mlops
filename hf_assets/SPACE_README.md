---
title: Car Price Predictor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
---

# 🚗 Car Price Predictor

Estimate the price of a US used vehicle from a few attributes. The estimate
comes from a LightGBM model trained on 552K+ auction records and continuously
retrained from validated user feedback.

This Space is the user-facing front-end of a full MLOps loop — the source
code, agents, retraining pipeline, and CI live at
**[Osman-Ozcanli/car-price-mlops](https://github.com/Osman-Ozcanli/car-price-mlops)**.

## How it works

1. Pick a vehicle's make / model / trim and enter its mileage, age, condition.
2. Click **Estimate price** — the model returns a point estimate plus an
   indicative range.
3. If the estimate was useful, optionally submit the **actual** sale price.
   Three deterministic validation layers check the submission before it
   enters the feedback dataset.
4. Once enough validated feedback accumulates, GitHub Actions retrains the
   model and pushes a new version — but only if it beats the current one
   on a shared validation slice.

## What this Space is not

- Not a legal valuation tool. The training data is 2015-vintage with a
  ×1.38 inflation adjustment, and predictions for sparse market segments
  (rare trims, low-volume states) revert toward the global mean.

## Reading the result

- **Estimated price** is the model's point prediction.
- **Indicative range** is `±5%` or `±$500`, whichever is larger — meant as
  a rough confidence hint, not a statistical interval.

## Privacy

Feedback you submit is appended to a public HF dataset and used to retrain
the model. No identity or location precision is stored — only the vehicle
attributes you entered, the actual sale price, and a UTC timestamp.

## Links

- 📦 Model: [`Osman-Ozcanli/car_price_prediction`](https://huggingface.co/Osman-Ozcanli/car_price_prediction)
- 📊 Feedback dataset: [`Osman-Ozcanli/car_price_prediction_feedback`](https://huggingface.co/datasets/Osman-Ozcanli/car_price_prediction_feedback)
- 🔧 Source code: [github.com/Osman-Ozcanli/car-price-mlops](https://github.com/Osman-Ozcanli/car-price-mlops)
