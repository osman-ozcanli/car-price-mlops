"""Multi-layer deterministic feedback validation.

Three independent checks run in sequence; the meta-layer aggregates results.
Each layer can reject independently — defense in depth.

    Layer 1 — Schema / range guard
        Hard limits on price and odometer. UI already restricts these but a
        determined user can bypass; this is the dataset-boundary fence.

    Layer 2 — Prediction-relative sanity
        Ratio (0.5x .. 2.0x) AND absolute deviation (<= $15K) vs. the model's
        own prediction. Catches "model said $19K, user typed $35K" cases.

    Layer 3 — Meta-auditor
        Aggregates results, returns a structured decision with reason codes
        suitable for logging to a rejected-feedback audit dataset.

All checks are deterministic. No LLM, no external calls.
"""
from dataclasses import dataclass, asdict
from typing import Optional

from common.constants import (
    PRICE_MIN, PRICE_MAX, ODOMETER_MAX,
    FEEDBACK_RATIO_MIN, FEEDBACK_RATIO_MAX, FEEDBACK_ABS_DEVIATION_MAX,
)


@dataclass
class ValidationResult:
    accepted: bool
    layer: str            # which layer made the final decision
    reason_code: str      # machine-readable
    reason_message: str   # human-readable, English


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


def validate_feedback(price: float, odometer: float,
                      predicted: Optional[float] = None) -> ValidationResult:
    """Meta-auditor: runs all layers, returns the first rejection or an accept."""
    rejection = layer1_schema(price, odometer)
    if rejection:
        return rejection
    rejection = layer2_prediction_sanity(price, predicted)
    if rejection:
        return rejection
    return ValidationResult(True, "META", "ACCEPTED", "Feedback accepted.")


def to_dict(result: ValidationResult) -> dict:
    return asdict(result)
