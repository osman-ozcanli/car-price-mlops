"""Project-wide constants. Single source of truth."""

# Inflation adjustment: training data is from 2015, app shows 2025+ prices.
# Update annually. Last calibrated: 2026-04.
INFLATION_MULTIPLIER = 1.38

# Validation guardrails for user feedback.
PRICE_MIN = 500
PRICE_MAX = 78_000
ODOMETER_MAX = 300_000

# Feedback validation: tightened from 5x/0.2x. Layer 1 deterministic guard.
# A prediction of $19K with feedback of $35K (ratio 1.84) is suspicious;
# real-world auction sale variance rarely exceeds 2x of a well-calibrated model.
FEEDBACK_RATIO_MAX = 2.0
FEEDBACK_RATIO_MIN = 0.5
# Layer 2: absolute deviation cap regardless of ratio. Catches edge cases on
# both ends (e.g. $500 prediction vs $1500 feedback passes 0.5x but is suspect).
FEEDBACK_ABS_DEVIATION_MAX = 15_000

# Layer 2.5 market-range flag (soft flag, does not reject).
# Computed from training/original_data.parquet (552K rows, 2015 auction data).
# Recompute if training distribution changes significantly.
MARKET_PRICE_MEDIAN = 12_200
MARKET_PRICE_STD = 9_320
MARKET_SIGMA_MULTIPLIER = 3  # flags prices outside median ± 3σ (~$40K upper)

# Retraining trigger thresholds.
# Demo mode: small threshold so the loop is observable end-to-end during testing.
# Production mode: statistically meaningful batch.
THRESHOLD_DEMO = 10
THRESHOLD_PROD = 100
THRESHOLD = THRESHOLD_DEMO  # active threshold; flip to THRESHOLD_PROD for production

# HuggingFace repos.
HF_USERNAME = "Osman-Ozcanli"
MODEL_REPO = f"{HF_USERNAME}/car_price_prediction"
DATASET_REPO = f"{HF_USERNAME}/car_price_prediction_feedback"
SPACE_REPO = f"{HF_USERNAME}/car_price_prediction_space"

# GitHub Actions trigger.
GITHUB_OWNER = "Osman-Ozcanli"
GITHUB_REPO = "car-price-mlops"
GITHUB_WORKFLOW = "retrain.yml"
