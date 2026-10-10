"""Build the comparison table and static figures from saved results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from models.train import MODELS
from src import config as C
from src import plots
from src.results import available_models, load_grid, load_per_degree, load_summary


def build_report():
    models = available_models()
    if not models:
        print("[report] no model results found - run the training scripts first.")
        return
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in models:
        s = load_summary(m)
        rows.append({
            "model": MODELS[m].label,
            "best_params": ", ".join(f"{k}={v:.4g}" for k, v in s["best_params"].items()),
            "cv_rmse": s["cv_rmse"],
            "test_rmse": s["test"]["rmse"],
            "test_mae": s["test"]["mae"],
            "test_r2": s["test"]["r2"],
            "n_features": s["n_features"],
            "n_nonzero": s["n_nonzero"],
        })
    comp = pd.DataFrame(rows).sort_values("cv_rmse")
    comp.to_csv(C.METRICS_DIR / "comparison.csv", index=False)
    print("\n" + comp.round(3).to_string(index=False))

    per_degree = {m: load_per_degree(m) for m in models}
    plots.degree_curves(per_degree).savefig(C.FIGURES_DIR / "degree_curves.png", dpi=150)
    for m in models:
        if m == "ols":
            continue
        g = load_grid(m)
        plots.heatmap(g, f"{MODELS[m].label}: CV RMSE over degree x alpha").savefig(
            C.FIGURES_DIR / f"{m}_heatmap.png", dpi=150)
        plots.alpha_curves(g, MODELS[m].label).savefig(C.FIGURES_DIR / f"{m}_alpha_curves.png", dpi=150)
    plt.close("all")
    print(f"[report] figures saved to {C.FIGURES_DIR}")


if __name__ == "__main__":
    build_report()
