import numpy as np
import pandas as pd
import pytest

from models.train import MODELS
from src import config as C
from src.features import n_poly_features
from src.tuning import best_per_degree, select_best


@pytest.mark.parametrize("degree,expected", [(1, 8), (2, 44), (3, 164), (4, 494)])
def test_number_of_polynomial_terms(degree, expected):
    assert n_poly_features(degree) == expected


def _toy(n=120):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.uniform(1, 5, size=(n, 8)), columns=C.FEATURES)
    y = X["cement"] * 2 + X["age"] ** 2 / 5 + rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("key", list(MODELS))
def test_every_model_fits_and_predicts(key):
    X, y = _toy()
    p = {"degree": 2}
    if "alpha" in MODELS[key].params:
        p["alpha"] = 0.1
    if "l1_ratio" in MODELS[key].params:
        p["l1_ratio"] = 0.5
    pred = MODELS[key].build(**p).fit(X, y).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


@pytest.mark.parametrize("key,size", [("ols", 4), ("ridge", 120), ("lasso", 120), ("elastic_net", 144)])
def test_grid_sizes(key, size):
    grid = MODELS[key].grid()
    assert len(grid) == size
    assert all(list(g) == MODELS[key].params for g in grid)       # same keys, same order
    assert all(isinstance(g["degree"], int) for g in grid)         # plain Python types


def test_spec_is_single_source_of_truth():
    assert MODELS["ols"].params == ["degree"]
    assert MODELS["ridge"].params == ["degree", "alpha"]
    assert MODELS["elastic_net"].params == ["degree", "alpha", "l1_ratio"]
    assert all(spec.key == key for key, spec in MODELS.items())


def test_scaler_is_inside_pipeline_so_cv_has_no_leakage():
    steps = [name for name, _ in MODELS["ridge"].build(2, alpha=1.0).steps]
    assert steps == ["scale_in", "poly", "scale_poly", "model"]


def test_select_best_rules():
    df = pd.DataFrame({
        "degree": [1, 2, 3], "alpha": [1.0, 1.0, 1.0],
        "cv_rmse": [9.0, 5.0, 4.9], "cv_rmse_std": [0.5, 0.5, 0.5],
        "train_rmse": [9.0, 4.0, 3.0],
    })
    assert select_best(df, "min")["degree"] == 3
    # degree 2 is within one standard error of the best -> simpler model wins
    assert select_best(df, "one_se")["degree"] == 2
    assert list(best_per_degree(df)["degree"]) == [1, 2, 3]


def test_diagnostics_helpers():
    from src.diagnostics import error_by_group, error_frame

    X, y = _toy()
    err = error_frame(X, y, y + 1.0)  # predictions are 1 MPa too high
    assert np.allclose(err["residual"], -1.0) and np.allclose(err["abs_error"], 1.0)
    table = error_by_group(err, "age", bins=[0, 3, 6])
    assert (table["MAE"] == 1.0).all() and (table["bias (mean residual)"] == -1.0).all()
