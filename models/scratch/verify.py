"""From-scratch checks and the gradient-descent experiment (run: python -m models.scratch.verify).

1. Compare the NumPy normal equation with scikit-learn (OLS and Ridge). cond_XtX shows
   how ill-conditioned X'X is; "pinv" in solver_used means the plain solve was judged
   unreliable and the pseudo-inverse was used instead.
2. Run gradient descent over learning rate x iterations to show
   convergence, slow progress and divergence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from models.scratch.model import ScratchLinearRegression
from src import config as C
from src.data import load_split
from src.features import make_pipeline


def _design(degree, X_train, X_other):
    """Scaled polynomial design matrices (steps before the estimator)."""
    pre = make_pipeline(degree, LinearRegression())[:-1].fit(X_train)
    return pre.transform(X_train), pre.transform(X_other)


def verify_against_sklearn(X_tr, X_te, y_tr) -> pd.DataFrame:
    rows = []
    for degree in C.DEGREES:
        A_tr, A_te = _design(degree, X_tr, X_te)
        for label, ours, ref in [
            ("OLS", ScratchLinearRegression("normal"), LinearRegression()),
            ("Ridge(alpha=10)", ScratchLinearRegression("normal", 10.0), Ridge(alpha=10.0)),
        ]:
            ours.fit(A_tr, y_tr)
            ref.fit(A_tr, y_tr)
            rows.append({
                "degree": degree,
                "model": label,
                "solver_used": ours.solver_used_,
                "cond_XtX": ours.condition_number_,
                "max_abs_prediction_diff": float(np.max(np.abs(ours.predict(A_te) - ref.predict(A_te)))),
                "max_abs_coef_diff": float(np.max(np.abs(ours.coef_ - ref.coef_))),
            })
    return pd.DataFrame(rows)


def gd_experiment(X_tr, y_tr) -> pd.DataFrame:
    A_tr, _ = _design(C.GD_DEGREE, X_tr, X_tr)
    exact = ScratchLinearRegression("normal").fit(A_tr, y_tr)
    exact_mse = float(np.mean((exact.predict(A_tr) - y_tr) ** 2))
    rows = []
    for lr in C.GD_LEARNING_RATES:
        for n_iter in C.GD_ITERATIONS:
            gd = ScratchLinearRegression("gd", lr=lr, n_iter=n_iter).fit(A_tr, y_tr)
            if gd.diverged_:
                mse, gap = np.nan, np.nan
            else:
                mse = float(np.mean((gd.predict(A_tr) - y_tr) ** 2))
                gap = float(np.linalg.norm(gd.coef_ - exact.coef_))
            rows.append({
                "degree": C.GD_DEGREE, "lr": lr, "n_iter": n_iter,
                "diverged": gd.diverged_, "iterations_run": gd.n_iter_,
                "train_mse": mse, "exact_train_mse": exact_mse,
                "coef_distance_to_exact": gap,
            })
    return pd.DataFrame(rows)


def main():
    X_tr, X_te, y_tr, _ = load_split()
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ver = verify_against_sklearn(X_tr, X_te, y_tr)
    ver.to_csv(C.METRICS_DIR / "scratch_verification.csv", index=False)
    gd = gd_experiment(X_tr, y_tr)
    gd.to_csv(C.METRICS_DIR / "gd_experiment.csv", index=False)
    print("[scratch] NumPy normal equation vs scikit-learn:")
    print(ver.to_string(index=False))
    return ver, gd


if __name__ == "__main__":
    main()