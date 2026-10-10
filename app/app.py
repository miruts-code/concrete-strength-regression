"""Home page. Run with:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import config as C
from src.results import load_comparison

st.set_page_config(page_title="Concrete Strength Regression", page_icon="🧱", layout="wide")

st.title("🧱 Concrete Strength: Polynomial Regression Lab")
st.markdown(
    """
Predict the **compressive strength of concrete (MPa)** from its mix recipe and age, and explore
*why* a model's flexibility must be controlled.

* **Polynomial degree** controls flexibility: low degrees *underfit*, high degrees *overfit*.
* **OLS** has no protection against overfitting. **Ridge**, **Lasso** and **Elastic Net** add a
  penalty on the coefficients (strength set by **alpha**).
* All hyperparameters are chosen by **5-fold cross-validation on the training set**.
  The test set is only used for the final score.

Use the pages in the sidebar: *Explore Data, Predict, Degree Comparison, Regularization Lab,
Gradient Descent*.
"""
)

st.subheader("The 8 input features")
info = pd.DataFrame(
    [{"feature": v[0], "unit": v[1], "meaning": v[2]} for v in C.FEATURE_INFO.values()]
)
st.dataframe(info, hide_index=True)
st.caption(f"Target: **{C.TARGET_LABEL}**.")

st.subheader("Final model comparison")
comp = load_comparison()
if comp is None:
    st.info("Results appear here after running `python run_all.py`.")
else:
    st.dataframe(comp.round(3), hide_index=True)
    st.caption("Hyperparameters chosen by 5-fold CV on the training set; test metrics computed once.")

st.divider()
st.caption(
    "Data: I-Cheng Yeh (1998), *Concrete Compressive Strength*, UCI Machine Learning Repository. "
    "Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural "
    "networks. *Cement and Concrete Research*, 28(12), 1797-1808."
)
