"""Coefficient paths: how weights change with the penalty strength alpha."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, Ridge, lasso_path

from src import config as C
from src.features import make_pipeline


def _design(degree, X_train):
    pre = make_pipeline(degree, Ridge())[:-1].fit(X_train)
    names = pre.named_steps["poly"].get_feature_names_out(C.FEATURES)
    return pre.transform(X_train), names


def coef_path(kind: str, degree: int, X_train, y_train, alphas=None):
    """Return (alphas ascending, coefs[n_alphas, n_features], feature_names)."""
    alphas = np.sort(np.asarray(C.ALPHAS if alphas is None else alphas))
    A, names = _design(degree, X_train)
    if kind == "ridge":
        coefs = np.array([Ridge(alpha=a).fit(A, y_train).coef_ for a in alphas])
    elif kind == "lasso":
        a_desc, c, _ = lasso_path(A, np.asarray(y_train), alphas=alphas[::-1],
                                  max_iter=C.LASSO_MAX_ITER, tol=C.LASSO_TOL)
        order = np.argsort(a_desc)
        alphas, coefs = a_desc[order], c.T[order]
    else:
        raise ValueError("kind must be 'ridge' or 'lasso'")
    return alphas, coefs, names
