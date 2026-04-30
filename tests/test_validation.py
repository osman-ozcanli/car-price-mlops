"""Tests for the three-layer feedback validator."""
import pytest

from common.validation import validate_feedback


# ── Layer 1: schema / range guard ────────────────────────────────────────────
class TestLayer1Schema:
    def test_price_below_minimum_rejected(self):
        result = validate_feedback(price=100, odometer=50_000, predicted=10_000)
        assert not result.accepted
        assert result.layer == "L1"
        assert result.reason_code == "PRICE_OUT_OF_RANGE"

    def test_price_above_maximum_rejected(self):
        result = validate_feedback(price=100_000, odometer=50_000, predicted=10_000)
        assert not result.accepted
        assert result.reason_code == "PRICE_OUT_OF_RANGE"

    def test_odometer_above_limit_rejected(self):
        result = validate_feedback(price=10_000, odometer=400_000, predicted=10_000)
        assert not result.accepted
        assert result.reason_code == "ODOMETER_OUT_OF_RANGE"


# ── Layer 2: prediction-relative sanity ──────────────────────────────────────
class TestLayer2PredictionSanity:
    def test_canonical_user_scenario_rejected(self):
        # Live evidence from issue note #13: $19K prediction, $35K feedback.
        # Old 5x guard let this through; the tightened 2x band must catch it.
        result = validate_feedback(price=35_000, odometer=60_000, predicted=19_000)
        assert not result.accepted
        assert result.layer == "L2"
        assert result.reason_code in {"RATIO_MISMATCH", "ABS_DEVIATION_TOO_LARGE"}

    def test_ratio_above_2x_rejected(self):
        result = validate_feedback(price=21_000, odometer=60_000, predicted=10_000)
        assert not result.accepted
        assert result.layer == "L2"

    def test_ratio_below_half_rejected(self):
        result = validate_feedback(price=4_000, odometer=60_000, predicted=10_000)
        assert not result.accepted
        assert result.layer == "L2"

    def test_absolute_deviation_cap_rejected(self):
        # Within ratio band (12000/8000=1.5) but absolute jump too large
        # is still reasonable here — pick a case that explicitly trips abs cap.
        # 30000/20000=1.5 ratio (passes), but |30000-20000|=10000 (passes).
        # Use 35000/20000=1.75 ratio (passes), |35000-20000|=15000 (boundary).
        # 36000/20000=1.8 ratio (passes), |36000-20000|=16000 → trips abs cap.
        result = validate_feedback(price=36_000, odometer=60_000, predicted=20_000)
        assert not result.accepted
        assert result.reason_code == "ABS_DEVIATION_TOO_LARGE"

    def test_no_predicted_price_skips_layer2(self):
        # If we have no prediction context, layer 2 cannot evaluate; should
        # accept (assuming layer 1 passed).
        result = validate_feedback(price=10_000, odometer=50_000, predicted=None)
        assert result.accepted


# ── Meta: accept path ────────────────────────────────────────────────────────
class TestAccept:
    def test_clean_feedback_accepted(self):
        result = validate_feedback(price=10_500, odometer=60_000, predicted=10_000)
        assert result.accepted
        assert result.layer == "META"
        assert result.reason_code == "ACCEPTED"

    @pytest.mark.parametrize("price,predicted", [
        (10_000, 10_000),   # exact match
        (15_000, 10_000),   # 1.5x — within band
        (8_000, 10_000),    # 0.8x — within band
        (5_000, 10_000),    # 0.5x boundary
    ])
    def test_within_ratio_band_accepted(self, price, predicted):
        result = validate_feedback(price=price, odometer=50_000, predicted=predicted)
        assert result.accepted
