import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import streamlit as st

from app.common import require_split
from src import config as C

st.set_page_config(page_title="Explore Data", page_icon="🔎", layout="wide")
st.title("🔎 Explore the training data")

X_tr, _, y_tr, _ = require_split()
df = X_tr.assign(**{C.TARGET: y_tr})

st.write(f"{len(df)} training rows (the test set is kept hidden).")
st.dataframe(df.describe().T.round(2))

col1, col2 = st.columns(2)
with col1:
    feat = st.selectbox("Feature", C.FEATURES, format_func=lambda f: C.FEATURE_INFO[f][0])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df[feat], bins=30)
    ax.set_xlabel(f"{C.FEATURE_INFO[feat][0]} ({C.FEATURE_INFO[feat][1]})")
    ax.set_ylabel("count")
    ax.set_title("Distribution")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df[feat], df[C.TARGET], s=8, alpha=0.5)
    ax.set_xlabel(C.FEATURE_INFO[feat][0])
    ax.set_ylabel(C.TARGET_LABEL)
    ax.set_title("Relationship with strength")
    st.pyplot(fig)

st.subheader("Why a straight line is not enough: strength vs. age")
by_age = df.groupby("age")[C.TARGET].agg(["mean", "count"]).reset_index()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(by_age["age"], by_age["mean"], "o-")
ax.set_xlabel("age (days)")
ax.set_ylabel("mean strength (MPa)")
ax.grid(alpha=0.3)
st.pyplot(fig)
st.caption("Strength rises steeply in early days and then flattens, a curve a degree-1 model cannot follow.")

st.subheader("Correlations")
corr = df.corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
fig.colorbar(im)
st.pyplot(fig)
