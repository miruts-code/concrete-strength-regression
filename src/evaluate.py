"""Metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def coef_summary(model) -> dict:
    """Total number of coefficients and how many are exactly non-zero."""
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    coef = np.ravel(est.coef_)
    return {"n_features": int(coef.size), "n_nonzero": int(np.count_nonzero(coef))}
