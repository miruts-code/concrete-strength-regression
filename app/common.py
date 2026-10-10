"""Shared, cached helpers for the Streamlit pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import KFold, cross_validate

from models.train import MODELS
from src import config as C
from src.analysis import coef_path
from src.data import load_split
from src.evaluate import coef_summary, regression_metrics
from src.results import available_models


@st.cache_data(show_spinner=False)
def get_split():
    return load_split()


def require_split():
    """Return the split, or stop the page with a helpful message."""
    try:
        return get_split()
    except FileNotFoundError:
        st.error("No data found. Run `python download_data.py` and then `python run_all.py` "
                 "in the project folder, then reload this page.")
        st.stop()


def require_results():
    if not available_models():
        st.warning("No trained results found yet. Run `python run_all.py` first, then reload.")
        st.stop()


@st.cache_data(show_spinner=False)
def live_cv(model_key: str, params: dict) -> dict:
    """Train + 5-fold CV on the TRAINING set for one hyperparameter setting."""
    X_tr, _, y_tr, _ = get_split()
    build = MODELS[model_key].build
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    res = cross_validate(build(**params), X_tr, y_tr, cv=kf,
                         scoring="neg_root_mean_squared_error", return_train_score=True)
    model = build(**params).fit(X_tr, y_tr)
    est = model.named_steps["model"]
    names = model.named_steps["poly"].get_feature_names_out(C.FEATURES)
    return {
        "train_rmse": float(-res["train_score"].mean()),
        "cv_rmse": float(-res["test_score"].mean()),
        "cv_std": float(res["test_score"].std(ddof=1)),
        "coef": np.ravel(est.coef_).tolist(),
        "names": list(names),
        **coef_summary(model),
    }


@st.cache_data(show_spinner=False)
def live_test_score(model_key: str, params: dict) -> dict:
    """Test-set score. Only call this after the user has chosen their setting."""
    X_tr, X_te, y_tr, y_te = get_split()
    model = MODELS[model_key].build(**params).fit(X_tr, y_tr)
    return regression_metrics(y_te, model.predict(X_te))


@st.cache_data(show_spinner=False)
def cached_coef_path(kind: str, degree: int):
    X_tr, _, y_tr, _ = get_split()
    return coef_path(kind, degree, X_tr, y_tr)


def snap_to_grid(value: float, grid) -> float:
    grid = np.asarray(grid, dtype=float)
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(value)))])


def training_ranges() -> pd.DataFrame:
    X_tr, *_ = get_split()
    return X_tr.agg(["min", "median", "max"]).T
