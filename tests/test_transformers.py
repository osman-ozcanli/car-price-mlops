"""Tests for the AddInteractions transformer."""
import pandas as pd

from common.transformers import AddInteractions


def test_age_x_odometer_column_added():
    df = pd.DataFrame({"age": [5, 10], "odometer": [50_000, 100_000]})
    out = AddInteractions().transform(df)
    assert "age_x_odo" in out.columns
    assert list(out["age_x_odo"]) == [250_000, 1_000_000]


def test_input_dataframe_not_mutated():
    df = pd.DataFrame({"age": [5], "odometer": [50_000]})
    AddInteractions().transform(df)
    assert "age_x_odo" not in df.columns


def test_fit_is_stateless():
    transformer = AddInteractions()
    df = pd.DataFrame({"age": [1], "odometer": [1000]})
    # Fit returns self and stores no state.
    assert transformer.fit(df) is transformer
    assert not any(attr.endswith("_") and not attr.startswith("__")
                   for attr in vars(transformer))
