"""Linear / Ridge regression implemented with NumPy only (no ML library solver).

Two ways to find the weights:

  method="normal"  the explicit normal equation   (X'X + alpha*I) w = X'y
  method="gd"      batch gradient descent on the same objective

Objective (the same minimiser as scikit-learn's Ridge; alpha=0 gives OLS):
    J(w) = 1/(2n) * ||y - Xw||^2  +  alpha/(2n) * ||w||^2
The intercept is never penalised: data are centred, then the intercept is
recovered as  b = mean(y) - mean(X) . w.

Conditioning
------------
Forming X'X squares the condition number of the problem (cond(X'X) = cond(X)^2).
When columns are (nearly) duplicated, as can happen with high-degree polynomial terms of
features that are often exactly 0, X'X becomes singular or badly conditioned and a plain
solve is unreliable. "normal" therefore records cond(X'X) and falls back to the
pseudo-inverse (minimum-norm solution) when it is too large or the solve fails.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

COND_LIMIT = 1e12  # above this, a plain solve of X'X loses most of its digits


class ScratchLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, method: str = "normal", alpha: float = 0.0,
                 lr: float = 0.05, n_iter: int = 2000, tol: float = 1e-10):
        self.method = method
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean

        self.loss_history_: list[float] = []
        self.diverged_ = False
        self.n_iter_ = 0
        self.solver_used_ = self.method
        self.condition_number_ = np.nan  # cond(X'X), set by method="normal"

        if self.method == "normal":
            w = self._normal_equation(Xc, yc)
        elif self.method == "gd":
            w = self._gradient_descent(Xc, yc)
        else:
            raise ValueError("method must be 'normal' or 'gd'")

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    # ------------------------------------------------------------ solvers
    def _normal_equation(self, Xc, yc):
        """Solve (X'X + alpha*I) w = X'y directly."""
        p = Xc.shape[1]
        gram = Xc.T @ Xc + self.alpha * np.eye(p)
        rhs = Xc.T @ yc
        with np.errstate(all="ignore"):
            self.condition_number_ = float(np.linalg.cond(gram))
        if np.isfinite(self.condition_number_) and self.condition_number_ < COND_LIMIT:
            try:
                self.solver_used_ = "solve"
                return np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                pass
        self.solver_used_ = "pinv"  # singular / ill-conditioned: minimum-norm solution
        return np.linalg.pinv(gram) @ rhs

    def _gradient_descent(self, Xc, yc):
        n, p = Xc.shape
        w = np.zeros(p)
        prev = np.inf
        with np.errstate(all="ignore"):
            for i in range(self.n_iter):
                resid = Xc @ w - yc
                loss = resid @ resid / (2 * n) + self.alpha * (w @ w) / (2 * n)
                if not np.isfinite(loss) or loss > 1e12:
                    self.diverged_ = True
                    break
                self.loss_history_.append(float(loss))
                self.n_iter_ = i + 1
                if abs(prev - loss) < self.tol:
                    break
                prev = loss
                w = w - self.lr * (Xc.T @ resid / n + self.alpha * w / n)
        return w

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_