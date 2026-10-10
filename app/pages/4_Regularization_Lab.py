import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st

from app.common import (cached_coef_path, live_cv, live_test_score, require_results,
                        require_split, snap_to_grid)
from models.train import MODELS
from src import config as C
from src.plots import alpha_curves, coef_bar, coef_path_plot, heatmap
from src.results import available_models, load_grid, load_summary

st.set_page_config(page_title="Regularization Lab", page_icon="🧪", layout="wide")
st.title("🧪 Regularization lab")
require_split()

ALPHA_OPTIONS = [float(a) for a in C.ALPHAS]

# --- session defaults ------------------------------------------------------
st.session_state.setdefault("lab_model", "ridge")
st.session_state.setdefault("lab_degree", 3)
st.session_state.setdefault("lab_alpha", ALPHA_OPTIONS[18])
st.session_state.setdefault("lab_l1", 0.5)


def jump_to_best():
    s = load_summary(st.session_state["lab_model"])
    if not s:
        return
    bp = s["best_params"]
    st.session_state["lab_degree"] = int(bp["degree"])
    if "alpha" in bp:
        st.session_state["lab_alpha"] = snap_to_grid(bp["alpha"], ALPHA_OPTIONS)
    if "l1_ratio" in bp:
        st.session_state["lab_l1"] = float(bp["l1_ratio"])


tab_live, tab_grid = st.tabs(["Live training", "Precomputed grid search"])

with tab_live:
    left, right = st.columns([1, 2])
    with left:
        key = st.selectbox("Model", list(MODELS), key="lab_model",
                           format_func=lambda k: MODELS[k].label)
        degree = st.slider("Polynomial degree", 1, C.MAX_DEGREE, key="lab_degree")
        params = {"degree": degree}
        if "alpha" in MODELS[key].params:
            params["alpha"] = float(st.select_slider(
                "alpha (penalty strength)", options=ALPHA_OPTIONS, key="lab_alpha",
                format_func=lambda a: f"{a:.2g}"))
        if "l1_ratio" in MODELS[key].params:
            params["l1_ratio"] = st.select_slider("l1_ratio (1 = pure Lasso)",
                                                   options=C.L1_RATIOS, key="lab_l1")
        st.button("Jump to best setting from grid search", on_click=jump_to_best,
                  disabled=not available_models())
        if key in ("lasso", "elastic_net") and degree >= 4 and params.get("alpha", 1) < 1e-2:
            st.info("Lasso with a tiny alpha at degree 4 can take a little while.")

    with right:
        with st.spinner("Training + 5-fold CV..."):
            r = live_cv(key, params)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train RMSE", f"{r['train_rmse']:.2f}")
        m2.metric("5-fold CV RMSE", f"{r['cv_rmse']:.2f}", help=f"std {r['cv_std']:.2f}")
        m3.metric("Gap (CV - train)", f"{r['cv_rmse'] - r['train_rmse']:.2f}")
        m4.metric("Non-zero coefficients", f"{r['n_nonzero']} / {r['n_features']}")
        st.pyplot(coef_bar(r["names"], np.array(r["coef"]),
                           f"Largest coefficients: {MODELS[key].label}, degree {degree}"))

    if key in ("ridge", "lasso"):
        st.subheader("Coefficient path (all alphas, this degree)")
        a, coefs, _ = cached_coef_path(key, degree)
        st.pyplot(coef_path_plot(a, coefs, f"{MODELS[key].label} coefficients vs alpha",
                                 highlight_alpha=params["alpha"]))
        st.caption("Ridge shrinks every coefficient smoothly; Lasso sets many to exactly zero. "
                   "The red line marks your current alpha.")

    with st.expander("Final test-set score for this exact setting (only after you have chosen!)"):
        st.write("Tuning by watching the test score would leak information. "
                 "Use cross-validation above to choose, then check here once.")
        if st.button("Reveal test score"):
            t = live_test_score(key, params)
            st.write(f"Test RMSE **{t['rmse']:.2f}** MPa, MAE {t['mae']:.2f}, R² {t['r2']:.3f}")

with tab_grid:
    require_results()
    which = [m for m in available_models() if m != "ols"]
    if not which:
        st.info("No Ridge/Lasso results yet.")
    else:
        m = st.selectbox("Model", which, format_func=lambda k: MODELS[k].label, key="grid_model")
        g = load_grid(m)
        s = load_summary(m)
        st.write("Best by CV:", s["best_params"], "| simplest within 1 SE:", s["one_se_params"])
        st.pyplot(heatmap(g, f"{MODELS[m].label}: CV RMSE over degree x alpha"))
        st.pyplot(alpha_curves(g, MODELS[m].label))
        st.caption("Each degree has its own best alpha: flexible high-degree models need a "
                   "stronger penalty than low-degree ones.")
