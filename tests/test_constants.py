"""Sanity checks on project-wide constants — guards against accidental edits."""
from common import constants as c


def test_validation_band_consistency():
    assert 0 < c.FEEDBACK_RATIO_MIN < 1 < c.FEEDBACK_RATIO_MAX
    assert c.FEEDBACK_RATIO_MAX <= 2.0, "Band must stay tight; widen only with intent."


def test_price_range_sane():
    assert c.PRICE_MIN < c.PRICE_MAX
    assert c.PRICE_MIN >= 500
    assert c.PRICE_MAX <= 100_000


def test_threshold_demo_smaller_than_prod():
    assert c.THRESHOLD_DEMO < c.THRESHOLD_PROD
    assert c.THRESHOLD in (c.THRESHOLD_DEMO, c.THRESHOLD_PROD)


def test_inflation_multiplier_in_plausible_range():
    # 2015 → 2026 cumulative US CPI is roughly 30–40%. Anything outside
    # 1.2–1.6 means the constant drifted unintentionally.
    assert 1.2 <= c.INFLATION_MULTIPLIER <= 1.6
