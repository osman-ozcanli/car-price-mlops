"""Multi-layer deterministic feedback validation.

Checks run in sequence; the meta-layer aggregates results.
Each layer can reject independently — defense in depth.

    Layer 1 — Schema / range guard
        Hard limits on price and odometer. UI already restricts these but a
        determined user can bypass; this is the dataset-boundary fence.

    Layer 2 — Prediction-relative sanity
        Ratio (0.5x .. 2.0x) AND absolute deviation (<= $15K) vs. the model's
        own prediction. Catches "model said $19K, user typed $35K" cases.

    Layer 2.5 — Market-range flag (soft, non-blocking)
        Flags prices outside training-distribution median ± 3σ (~$500–$40K).
        Does NOT reject; sets result.flagged=True so the dataset row can be
        filtered or down-weighted at retraining time.

    Layer 3 — Meta-auditor
        Aggregates results, returns a structured decision with reason codes
        suitable for logging to a rejected-feedback audit dataset.

All checks are deterministic. No LLM, no external calls.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

from common.constants import (
    FEEDBACK_ABS_DEVIATION_MAX,
    FEEDBACK_RATIO_MAX,
    FEEDBACK_RATIO_MIN,
    MARKET_PRICE_MEDIAN,
    MARKET_PRICE_STD,
    MARKET_SIGMA_MULTIPLIER,
    ODOMETER_MAX,
    PRICE_MAX,
    PRICE_MIN,
)


@dataclass
class ValidationResult:
    accepted: bool
    layer: str            # which layer made the final decision
    reason_code: str      # machine-readable
    reason_message: str   # human-readable, English
    flagged: bool = field(default=False)   # Layer 2.5 soft flag (price outside median±3σ)
    flag_reason: str = field(default="")


def layer1_schema(price: float, odometer: float) -> Optional[ValidationResult]:
    if odometer > ODOMETER_MAX:
        return ValidationResult(False, "L1", "ODOMETER_OUT_OF_RANGE",
                                f"Odometer {odometer:,.0f} exceeds the {ODOMETER_MAX:,} mile limit.")
    if price < PRICE_MIN or price > PRICE_MAX:
        return ValidationResult(False, "L1", "PRICE_OUT_OF_RANGE",
                                f"Price ${price:,.0f} is outside the accepted range "
                                f"(${PRICE_MIN:,}–${PRICE_MAX:,}).")
    return None


def layer2_prediction_sanity(price: float, predicted: Optional[float]) -> Optional[ValidationResult]:
    if not predicted or predicted <= 0:
        return None  # cannot evaluate; defer
    ratio = price / predicted
    if ratio > FEEDBACK_RATIO_MAX or ratio < FEEDBACK_RATIO_MIN:
        return ValidationResult(False, "L2", "RATIO_MISMATCH",
                                f"Reported price (${price:,.0f}) deviates too far from "
                                f"the prediction (${predicted:,.0f}) — ratio {ratio:.2f}x "
                                f"is outside the accepted band "
                                f"({FEEDBACK_RATIO_MIN}x–{FEEDBACK_RATIO_MAX}x).")
    abs_dev = abs(price - predicted)
    if abs_dev > FEEDBACK_ABS_DEVIATION_MAX:
        return ValidationResult(False, "L2", "ABS_DEVIATION_TOO_LARGE",
                                f"Reported price (${price:,.0f}) differs from the "
                                f"prediction (${predicted:,.0f}) by ${abs_dev:,.0f}, "
                                f"exceeding the ${FEEDBACK_ABS_DEVIATION_MAX:,} cap.")
    return None


def layer25_market_flag(price: float) -> tuple:
    """Soft flag: True if price falls outside training-distribution median ± 3σ.

    Returns (flagged: bool, reason: str). Never rejects — caller attaches to result.
    """
    upper = MARKET_PRICE_MEDIAN + MARKET_SIGMA_MULTIPLIER * MARKET_PRICE_STD
    lower = max(PRICE_MIN, MARKET_PRICE_MEDIAN - MARKET_SIGMA_MULTIPLIER * MARKET_PRICE_STD)
    if price > upper:
        return True, (f"Price ${price:,.0f} exceeds market upper bound "
                      f"${upper:,.0f} (median+{MARKET_SIGMA_MULTIPLIER}σ).")
    if price < lower:
        return True, (f"Price ${price:,.0f} is below market lower bound "
                      f"${lower:,.0f} (median-{MARKET_SIGMA_MULTIPLIER}σ).")
    return False, ""


def validate_feedback(price: float, odometer: float,
                      predicted: Optional[float] = None) -> ValidationResult:
    """Meta-auditor: runs all layers, returns the first rejection or an accept."""
    rejection = layer1_schema(price, odometer)
    if rejection:
        return rejection
    rejection = layer2_prediction_sanity(price, predicted)
    if rejection:
        return rejection
    flagged, flag_reason = layer25_market_flag(price)
    return ValidationResult(True, "META", "ACCEPTED", "Feedback accepted.",
                            flagged=flagged, flag_reason=flag_reason)


def to_dict(result: ValidationResult) -> dict:
    return asdict(result)
