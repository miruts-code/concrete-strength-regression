"""The from-scratch NumPy implementation must reproduce scikit-learn."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from models.scratch.model import ScratchLinearRegression


def _data(n=200, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 3.0 + rng.normal(scale=0.1, size=n)
    return X, y


def test_normal_equation_matches_sklearn_ols():
    X, y = _data()
    ours, ref = ScratchLinearRegression("normal").fit(X, y), LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-8)
    assert abs(ours.intercept_ - ref.intercept_) < 1e-8


def test_normal_equation_matches_sklearn_ridge():
    X, y = _data()
    ours = ScratchLinearRegression("normal", alpha=5.0).fit(X, y)
    ref = Ridge(alpha=5.0).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-8)
    assert abs(ours.intercept_ - ref.intercept_) < 1e-8


def test_weights_satisfy_the_normal_equation():
    """The solution must satisfy (X'X + alpha*I) w = X'y on centred data."""
    X, y = _data()
    alpha = 2.0
    m = ScratchLinearRegression("normal", alpha=alpha).fit(X, y)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    lhs = (Xc.T @ Xc + alpha * np.eye(X.shape[1])) @ m.coef_
    np.testing.assert_allclose(lhs, Xc.T @ yc, atol=1e-8)
    assert m.solver_used_ == "solve" and np.isfinite(m.condition_number_)


def test_normal_equation_falls_back_when_xtx_is_singular():
    X, y = _data()
    X_dup = np.hstack([X, X[:, :1]])           # an exactly duplicated column -> singular X'X
    ours = ScratchLinearRegression("normal").fit(X_dup, y)
    ref = LinearRegression().fit(X_dup, y)
    assert ours.solver_used_ == "pinv"
    np.testing.assert_allclose(ours.predict(X_dup), ref.predict(X_dup), atol=1e-6)


def test_gradient_descent_converges_to_ols():
    X, y = _data()
    gd = ScratchLinearRegression("gd", lr=0.1, n_iter=5000).fit(X, y)
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(gd.coef_, ref.coef_, atol=1e-3)
    assert not gd.diverged_
    assert gd.loss_history_[-1] < gd.loss_history_[0]


def test_gradient_descent_detects_divergence():
    X, y = _data()
    gd = ScratchLinearRegression("gd", lr=50.0, n_iter=500).fit(X, y)
    assert gd.diverged_


def test_unknown_method_raises():
    X, y = _data()
    with pytest.raises(ValueError):
        ScratchLinearRegression("lstsq").fit(X, y)