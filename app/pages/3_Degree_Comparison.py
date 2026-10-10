import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.common import live_cv, require_results, require_split
from models.train import MODELS
from src import config as C
from src.features import n_poly_features
from src.plots import degree_curves
from src.results import available_models, load_per_degree

st.set_page_config(page_title="Degree Comparison", page_icon="📈", layout="wide")
st.title("📈 Why one polynomial degree beats another")
require_split()

tab_pre, tab_live = st.tabs(["Precomputed (all models)", "Live: train it yourself"])

with tab_pre:
    require_results()
    per_degree = {m: load_per_degree(m) for m in available_models()}
    st.pyplot(degree_curves(per_degree))
    st.markdown(
        """
* **Degree 1 underfits**: training *and* validation error are both high.
* **Higher degrees** drive training error down, but validation error stops improving or rises:
  the model memorises noise (**overfitting**).
* Compare **OLS** (no protection) with **Ridge / Lasso / Elastic Net**: the penalty keeps the
  high-degree models usable.
* The test curve is shown as a diagnostic. Model selection used cross-validation only.
"""
    )
    rows = []
    for m, df in per_degree.items():
        for _, r in df.iterrows():
            rows.append({"model": MODELS[m].label, "degree": int(r["degree"]),
                         "train RMSE": r["train_rmse"], "CV RMSE": r["cv_rmse"],
                         "test RMSE": r["test_rmse"], "non-zero coefs": int(r["n_nonzero"])})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)

with tab_live:
    st.write("Pick a model and settings; the app trains every degree live and reports "
             "**training error vs 5-fold CV error** (test set not used).")
    key = st.selectbox("Model", list(MODELS), format_func=lambda k: MODELS[k].label)
    params_needed = MODELS[key].params
    alpha = l1 = None
    if "alpha" in params_needed:
        alpha = float(st.select_slider("alpha", options=[float(a) for a in C.ALPHAS],
                                       value=float(C.ALPHAS[18]), format_func=lambda a: f"{a:.2g}"))
    if "l1_ratio" in params_needed:
        l1 = st.select_slider("l1_ratio", options=C.L1_RATIOS, value=0.5)

    if st.button("Train all degrees"):
        rows = []
        with st.spinner("Training with 5-fold cross-validation..."):
            for d in C.DEGREES:
                p = {"degree": d}
                if alpha is not None:
                    p["alpha"] = alpha
                if l1 is not None:
                    p["l1_ratio"] = l1
                r = live_cv(key, p)
                rows.append({"degree": d, "polynomial terms": n_poly_features(d),
                             "train RMSE": r["train_rmse"], "CV RMSE": r["cv_rmse"],
                             "non-zero coefs": r["n_nonzero"]})
        df = pd.DataFrame(rows)
        st.dataframe(df.round(3), hide_index=True)
        st.line_chart(df.set_index("degree")[["train RMSE", "CV RMSE"]])
        st.caption("A growing gap between the two lines is overfitting.")
