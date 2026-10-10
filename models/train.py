"""Model definitions and training:

A model is one `ModelSpec`. To add a model, add one entry to `MODELS`; the training
loop, the app, the notebooks and the tests all read from here.

Usage:
    python -m models.train --model ridge
    python -m models.train --all
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src import config as C
from src.features import make_pipeline
from src.runner import run_model


def _plain(value):
    """numpy scalars -> plain Python numbers (clean JSON / CSV / printing)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """Everything that defines one model.

    key        short id used in file names (results/metrics/<key>_summary.json)
    label      display name for tables, plots and the app
    estimator  scikit-learn class that goes at the end of the pipeline
    tune       hyperparameters searched by cross-validation (always includes "degree")
    fixed      estimator arguments that are set once and never searched
    """

    key: str
    label: str
    estimator: type
    tune: dict[str, Any]
    fixed: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> list[str]:
        """Names of the tuned hyperparameters, e.g. ['degree', 'alpha']."""
        return list(self.tune)

    def build(self, degree: int, **hyperparams) -> Pipeline:
        """An unfitted pipeline for one hyperparameter combination."""
        return make_pipeline(degree, self.estimator(**self.fixed, **hyperparams))

    def grid(self) -> list[dict]:
        """Every hyperparameter combination (degree is the outermost loop)."""
        names = list(self.tune)
        return [
            {n: _plain(v) for n, v in zip(names, combo)}
            for combo in itertools.product(*self.tune.values())
        ]


_ITERATIVE = {"max_iter": C.LASSO_MAX_ITER, "tol": C.LASSO_TOL}  # coordinate-descent solvers

MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec("ols", "OLS", LinearRegression,
                  tune={"degree": C.DEGREES}),
        ModelSpec("ridge", "Ridge (L2)", Ridge,
                  tune={"degree": C.DEGREES, "alpha": C.ALPHAS}),
        ModelSpec("lasso", "Lasso (L1)", Lasso,
                  tune={"degree": C.DEGREES, "alpha": C.ALPHAS},
                  fixed=_ITERATIVE),
        ModelSpec("elastic_net", "Elastic Net", ElasticNet,
                  tune={"degree": C.DEGREES, "alpha": C.ALPHAS_COARSE, "l1_ratio": C.L1_RATIOS},
                  fixed=_ITERATIVE),
    ]
}


def train(key: str) -> dict:
    """Tune with 5-fold CV, refit the winner, score once on the test set, save results."""
    return run_model(MODELS[key])


def main():
    ap = argparse.ArgumentParser(description="Train one model or all of them.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODELS))
    group.add_argument("--all", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for key in (MODELS if args.all else [args.model]):
        train(key)


if __name__ == "__main__":
    main()
