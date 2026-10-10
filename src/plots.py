"""Matplotlib figures shared by the report script and the Streamlit app."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from models.train import MODELS



def degree_curves(per_degree: dict):
    """Train / CV / test RMSE against polynomial degree, one panel per model."""
    n = len(per_degree)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True, squeeze=False)
    for ax, (name, df) in zip(axes[0], per_degree.items()):
        ax.plot(df["degree"], df["train_rmse"], "o-", label="train")
        ax.plot(df["degree"], df["cv_rmse"], "s-", label="5-fold CV")
        ax.plot(df["degree"], df["test_rmse"], "^--", label="test (diagnostic)")
        ax.set_yscale("log")
        ax.set_xticks(df["degree"])
        ax.set_title(MODELS[name].label)
        ax.set_xlabel("polynomial degree")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("RMSE (MPa, log scale)")
    axes[0][0].legend()
    fig.tight_layout()
    return fig


def heatmap(grid_df, title: str):
    """CV RMSE over degree x alpha (best l1_ratio per cell for Elastic Net)."""
    g = grid_df.groupby(["degree", "alpha"])["cv_rmse"].min().unstack("alpha")
    fig, ax = plt.subplots(figsize=(9, 2.8 + 0.3 * len(g)))
    vals = g.values
    im = ax.imshow(vals, aspect="auto", cmap="viridis_r",
                   norm=LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    ax.set_yticks(range(len(g)), [f"degree {d}" for d in g.index])
    cols = list(g.columns)
    step = max(1, len(cols) // 8)
    ax.set_xticks(range(0, len(cols), step), [f"{cols[i]:.0e}" for i in range(0, len(cols), step)])
    r, c = np.unravel_index(np.nanargmin(vals), vals.shape)
    ax.plot(c, r, "r*", markersize=16, label="best")
    ax.set_xlabel("alpha (penalty strength)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="CV RMSE (MPa, log colour scale)")
    fig.tight_layout()
    return fig


def alpha_curves(grid_df, title: str):
    """Validation curve over alpha, one line per degree (train dashed)."""
    g = grid_df.groupby(["degree", "alpha"])[["cv_rmse", "train_rmse"]].min().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, sub in g.groupby("degree"):
        line, = ax.plot(sub["alpha"], sub["cv_rmse"], label=f"degree {d} - CV")
        ax.plot(sub["alpha"], sub["train_rmse"], "--", color=line.get_color(), alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("RMSE (MPa)")
    ax.set_title(title + " (dashed = training error)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def coef_path_plot(alphas, coefs, title: str, highlight_alpha: float | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(alphas, coefs, lw=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("coefficient value")
    axes[0].set_title(title)
    axes[1].plot(alphas, np.count_nonzero(np.abs(coefs) > 1e-12, axis=1), "o-")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("non-zero coefficients")
    axes[1].set_title("Model size")
    for ax in axes:
        ax.grid(alpha=0.3)
        if highlight_alpha:
            ax.axvline(highlight_alpha, color="red", ls=":")
    fig.tight_layout()
    return fig


def coef_bar(names, values, title: str, top: int = 15):
    idx = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(idx) + 1.2))
    ax.barh([names[i] for i in idx], values[idx],
            color=["tab:red" if values[i] < 0 else "tab:blue" for i in idx])
    ax.set_title(title)
    ax.set_xlabel("coefficient (on standardised features)")
    fig.tight_layout()
    return fig


def gd_loss_plot(histories: dict, title: str = "Gradient descent loss"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, h in histories.items():
        if len(h):
            ax.plot(h, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss J(w)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def pred_vs_actual(y_true, y_pred, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.6)
    axes[0].plot([lo, hi], [lo, hi], "r--")
    axes[0].set_xlabel("actual strength (MPa)")
    axes[0].set_ylabel("predicted (MPa)")
    axes[0].set_title(title)
    resid = np.asarray(y_true) - np.asarray(y_pred)
    axes[1].scatter(y_pred, resid, s=10, alpha=0.6)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("predicted (MPa)")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def residual_diagnostics(y_pred, residual, age, title: str = "Residual diagnostics"):
    """Four standard residual checks (residual = actual - predicted)."""
    from scipy import stats

    y_pred, residual, age = map(np.asarray, (y_pred, residual, age))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.scatter(y_pred, residual, s=10, alpha=0.5)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("predicted strength (MPa)")
    ax.set_ylabel("residual (MPa)")
    ax.set_title("Residuals vs predicted")

    ax = axes[0, 1]
    ax.hist(residual, bins=30, density=True, alpha=0.7)
    xs = np.linspace(residual.min(), residual.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, residual.mean(), residual.std()), "r-", label="normal curve")
    ax.set_xlabel("residual (MPa)")
    ax.set_title("Distribution of residuals")
    ax.legend()

    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(residual, dist="norm")
    ax.scatter(osm, osr, s=10, alpha=0.6)
    ax.plot(osm, slope * np.asarray(osm) + intercept, "r--")
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("ordered residuals")
    ax.set_title("Q-Q plot")

    ax = axes[1, 1]
    ax.scatter(age, residual, s=10, alpha=0.5)
    ax.axhline(0, color="r", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("age (days, log scale)")
    ax.set_ylabel("residual (MPa)")
    ax.set_title("Residuals vs age")

    for a in axes.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def log_age_plot(df):
    """CV RMSE vs degree for raw age and log(age), one panel per model."""
    models = list(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for variant, sub in df[df["model"] == m].groupby("features"):
            sub = sub.sort_values("degree")
            ax.errorbar(sub["degree"], sub["cv_rmse"], yerr=sub["cv_rmse_std"], marker="o",
                        capsize=3, label=variant)
        ax.set_yscale("log")
        ax.set_xticks(sorted(df["degree"].unique()))
        ax.set_xlabel("polynomial degree")
        ax.set_title(m)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("5-fold CV RMSE (MPa, log scale)")
    axes[0][0].legend()
    fig.tight_layout()
    return fig
