"""Residual analysis, error breakdowns and the log(age) experiment (used by notebook 02).

All diagnostics use out-of-fold predictions on the TRAINING set, so the test set
is never used to make modelling decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from models.train import MODELS
from src import config as C
from src.results import available_models, load_summary
from src.tuning import cv_grid, select_best


def best_model():
    """(model_key, best_params) of the model with the lowest CV RMSE."""
    summaries = {m: load_summary(m) for m in available_models()}
    key = min(summaries, key=lambda m: summaries[m]["cv_rmse"])
    return key, summaries[key]["best_params"]


def oof_predictions(model_key: str, params: dict, X, y) -> np.ndarray:
    """Out-of-fold predictions: each row is predicted by a model that never saw it."""
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    return cross_val_predict(MODELS[model_key].build(**params), X, y, cv=kf)


def error_frame(X, y, pred) -> pd.DataFrame:
    df = X.copy()
    df["actual"] = np.asarray(y)
    df["predicted"] = np.asarray(pred)
    df["residual"] = df["actual"] - df["predicted"]  # actual - predicted
    df["abs_error"] = df["residual"].abs()
    return df


def error_by_group(df: pd.DataFrame, column: str, bins=None) -> pd.DataFrame:
    """Typical error (MAE), bias (mean residual) and row count per group."""
    key = pd.cut(df[column], bins=bins) if bins is not None else df[column]
    g = df.groupby(key, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "MAE": g["abs_error"].mean(),
        "bias (mean residual)": g["residual"].mean(),
    }).round(3)


def log_age_experiment(X, y, alphas=None) -> pd.DataFrame:
    """CV comparison of raw age vs log(age) for OLS and Ridge across degrees.

    Strength rises steeply with age and then flattens (log-like). If log(age)
    lets a LOWER degree match a higher one, the polynomial was only imitating a log.
    """
    alphas = [float(a) for a in (C.ALPHAS[::3] if alphas is None else alphas)]
    variants = {"raw age": X, "log(age)": X.assign(age=np.log(X["age"]))}
    rows = []
    for variant, Xv in variants.items():
        ols = cv_grid(MODELS["ols"].build, [{"degree": d} for d in C.DEGREES],
                      Xv, y, verbose=False)
        ridge = cv_grid(MODELS["ridge"].build,
                        [{"degree": d, "alpha": a} for d in C.DEGREES for a in alphas],
                        Xv, y, verbose=False)
        for name, grid in (("OLS", ols), ("Ridge", ridge)):
            idx = grid.groupby("degree")["cv_rmse"].idxmin()
            for _, r in grid.loc[idx].iterrows():
                rows.append({"features": variant, "model": name, "degree": int(r["degree"]),
                             "cv_rmse": r["cv_rmse"], "cv_rmse_std": r["cv_rmse_std"]})
    return pd.DataFrame(rows)
