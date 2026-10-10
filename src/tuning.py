"""Hyperparameter search: grid + 5-fold CV on the TRAINING set only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from src import config as C

PARAM_ORDER = ["degree", "alpha", "l1_ratio"]


def cv_grid(build_fn, grid: list[dict], X, y, n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """Cross-validate every hyperparameter combination.

    Returns one row per combination with mean/std validation RMSE and the
    mean training RMSE (to expose over-fitting).
    """
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    rows = []
    for i, params in enumerate(grid, 1):
        res = cross_validate(
            build_fn(**params), X, y, cv=kf,
            scoring="neg_root_mean_squared_error",
            return_train_score=True, n_jobs=n_jobs,
        )
        val, tr = -res["test_score"], -res["train_score"]
        rows.append({
            **params,
            "cv_rmse": val.mean(),
            "cv_rmse_std": val.std(ddof=1),
            "train_rmse": tr.mean(),
        })
        if verbose and (i % 20 == 0 or i == len(grid)):
            print(f"    {i}/{len(grid)} combinations done")
    return pd.DataFrame(rows)


def param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in PARAM_ORDER if c in df.columns]


def select_best(df: pd.DataFrame, rule: str = "min") -> dict:
    """Pick hyperparameters.

    rule="min"     -> lowest mean CV RMSE.
    rule="one_se"  -> simplest model within one standard error of the best
                      (lowest degree, then strongest penalty).
    """
    best = df.loc[df["cv_rmse"].idxmin()]
    if rule == "one_se":
        se = best["cv_rmse_std"] / np.sqrt(C.CV_FOLDS)
        ok = df[df["cv_rmse"] <= best["cv_rmse"] + se]
        sort_cols = [c for c in ["degree", "alpha", "l1_ratio"] if c in ok.columns]
        ascending = [c == "degree" for c in sort_cols]
        best = ok.sort_values(sort_cols, ascending=ascending).iloc[0]
    out = {}
    for c in param_cols(df):
        v = best[c]
        out[c] = int(v) if c == "degree" else float(v)
    return out


def best_per_degree(df: pd.DataFrame) -> pd.DataFrame:
    """For every degree, the row with the lowest CV RMSE."""
    idx = df.groupby("degree")["cv_rmse"].idxmin()
    return df.loc[idx].reset_index(drop=True)
