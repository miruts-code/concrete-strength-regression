import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app.common import get_split, require_split
from models.scratch.model import ScratchLinearRegression
from src import config as C
from src.features import make_pipeline
from src.plots import gd_loss_plot


st.set_page_config(page_title="Gradient Descent", page_icon="⛰️", layout="wide")
st.title("⛰️ How the weights are found: gradient descent")
require_split()

st.markdown(
    """
Weights are not chosen by hand: they are the values that minimise a loss.
OLS/Ridge have a closed-form answer (the **normal equation**, implemented in NumPy and compared with
scikit-learn; the dotted line below is its closed-form solution). **Gradient descent** reaches the same answer step by step:
`w <- w - learning_rate * gradient`. Too small a learning rate is slow; too large **diverges**.
"""
)


@st.cache_data(show_spinner=False)
def design(degree: int):
    X_tr, _, y_tr, _ = get_split()
    pre: Pipeline = make_pipeline(degree, Ridge())[:-1].fit(X_tr)
    return pre.transform(X_tr), np.asarray(y_tr)


c1, c2 = st.columns(2)
degree = c1.slider("Polynomial degree", 1, C.MAX_DEGREE, C.GD_DEGREE)
n_iter = c2.slider("Iterations", 50, 5000, 1000, step=50)
lrs = st.multiselect("Learning rates to compare", C.GD_LEARNING_RATES,
                     default=[0.01, 0.1, 2.0])
alpha = st.select_slider("Optional Ridge penalty alpha (0 = plain OLS)",
                         options=[0.0] + [float(a) for a in C.ALPHAS[::3]], value=0.0,
                         format_func=lambda a: f"{a:.2g}")

A, y = design(degree)
n = len(y)
exact = ScratchLinearRegression("normal", alpha).fit(A, y)  # closed-form reference
resid = exact.predict(A) - y
exact_loss = float(resid @ resid / (2 * n) + alpha * (exact.coef_ @ exact.coef_) / (2 * n))

histories, rows = {}, []
for lr in lrs:
    gd = ScratchLinearRegression("gd", alpha=alpha, lr=lr, n_iter=n_iter).fit(A, y)
    histories[f"lr={lr}" + (" (diverged)" if gd.diverged_ else "")] = gd.loss_history_
    rows.append({
        "learning rate": lr, "status": "DIVERGED" if gd.diverged_ else "ok",
        "iterations run": gd.n_iter_,
        "final loss": None if not gd.loss_history_ else gd.loss_history_[-1],
        "distance to exact weights": None if gd.diverged_ else float(np.linalg.norm(gd.coef_ - exact.coef_)),
    })

fig = gd_loss_plot(histories)
fig.axes[0].axhline(exact_loss, color="k", ls=":", label="exact (normal equation)")
fig.axes[0].legend(fontsize=8)
st.pyplot(fig)
st.dataframe(pd.DataFrame(rows).round(6), hide_index=True)
st.caption("The dotted line is the minimum loss from the closed-form solution. "
           "Converged runs approach it; a learning rate that is too large blows up.")