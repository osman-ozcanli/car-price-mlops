"""Shared transformers. Single source of truth for joblib pickle compatibility."""
from sklearn.base import BaseEstimator, TransformerMixin


class AddInteractions(BaseEstimator, TransformerMixin):
    """Stateless transformer that adds the age * odometer interaction column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_x_odo"] = X["age"] * X["odometer"]
        return X
