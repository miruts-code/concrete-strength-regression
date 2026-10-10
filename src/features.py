"""The shared preprocessing + model pipeline."""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_pipeline(degree: int, estimator) -> Pipeline:
    """scale -> polynomial expansion -> scale -> estimator.

    * The first scaler keeps the polynomial expansion numerically tame
      (raw terms like age**4 are huge).
    * The second scaler gives every polynomial term unit variance, so the
      Ridge / Lasso penalty treats all terms fairly.
    * Everything lives inside the pipeline, so cross-validation refits the
      scalers on each training fold only (no leakage).
    """
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale_poly", StandardScaler()),
            ("model", estimator),
        ]
    )


def n_poly_features(degree: int, n_inputs: int = 8) -> int:
    """Number of polynomial terms (excluding the intercept)."""
    from math import comb

    return comb(n_inputs + degree, degree) - 1
