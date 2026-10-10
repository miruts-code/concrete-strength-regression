"""The train/tune/evaluate routine shared by every model (called from models/train.py)."""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from src import config as C
from src.data import load_split
from src.evaluate import coef_summary, regression_metrics
from src.tuning import best_per_degree, cv_grid, param_cols, select_best


def _fit_eval(build_fn, params, X_tr, y_tr, X_te, y_te):
    model = build_fn(**params).fit(X_tr, y_tr)
    return (
        model,
        regression_metrics(y_tr, model.predict(X_tr)),
        regression_metrics(y_te, model.predict(X_te)),
    )


def run_model(spec) -> dict:
    """Grid-search a `ModelSpec` with 5-fold CV, refit the winner, score once on the test set."""
    name, build_fn, grid = spec.key, spec.build, spec.grid()
    t0 = time.time()
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    C.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, y_tr, y_te = load_split()
    print(f"[{name}] searching {len(grid)} combinations x {C.CV_FOLDS} folds")

    # 1) tuning on the training set only
    grid_df = cv_grid(build_fn, grid, X_tr, y_tr)
    grid_df.to_csv(C.METRICS_DIR / f"{name}_grid.csv", index=False)

    # 2) selection
    best = select_best(grid_df, "min")
    one_se = select_best(grid_df, "one_se")

    # 3) diagnostic per-degree table (selection uses CV only; test shown for plots)
    rows = []
    for _, r in best_per_degree(grid_df).iterrows():
        p = {c: (int(r[c]) if c == "degree" else float(r[c])) for c in param_cols(grid_df)}
        m, tr_m, te_m = _fit_eval(build_fn, p, X_tr, y_tr, X_te, y_te)
        rows.append({**p, "cv_rmse": r["cv_rmse"], "train_rmse": tr_m["rmse"],
                     "test_rmse": te_m["rmse"], **coef_summary(m)})
    pd.DataFrame(rows).to_csv(C.METRICS_DIR / f"{name}_per_degree.csv", index=False)

    # 4) final model: refit the CV winner on the full training set, test ONCE
    model, train_m, test_m = _fit_eval(build_fn, best, X_tr, y_tr, X_te, y_te)
    joblib.dump(model, C.SAVED_MODELS_DIR / f"{name}.joblib")

    summary = {
        "model": name,
        "best_params": best,
        "one_se_params": one_se,
        "cv_rmse": float(grid_df["cv_rmse"].min()),
        "train": train_m,
        "test": test_m,
        **coef_summary(model),
        "grid_size": len(grid),
        "cv_folds": C.CV_FOLDS,
        "seed": C.SEED,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(C.METRICS_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] best={best} cv_rmse={summary['cv_rmse']:.3f} "
          f"test_rmse={test_m['rmse']:.3f} ({summary['runtime_seconds']}s)")
    return summary
