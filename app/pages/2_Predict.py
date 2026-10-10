import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from models.train import MODELS
from app.common import require_results, require_split, training_ranges
from src import config as C
from src.results import available_models, load_saved_model, load_summary

st.set_page_config(page_title="Predict", page_icon="🎯", layout="wide")
st.title("🎯 Predict concrete strength")
require_split()
require_results()

rng = training_ranges()
st.write("Set the mix recipe. Sliders are limited to the range seen in training, because "
         "polynomial models extrapolate unreliably outside it.")

cols = st.columns(2)
values = {}
for i, f in enumerate(C.FEATURES):
    label, unit, meaning = C.FEATURE_INFO[f]
    lo, med, hi = (float(rng.loc[f, k]) for k in ("min", "median", "max"))
    with cols[i % 2]:
        values[f] = st.slider(f"{label} ({unit})", lo, hi, med, help=meaning)

x = pd.DataFrame([values])[C.FEATURES]
st.subheader("Predicted compressive strength")
out = []
for name in available_models():
    model = load_saved_model(name)
    s = load_summary(name)
    out.append({
        "model": MODELS[name].label,
        "tuned settings": ", ".join(f"{k}={v:.4g}" for k, v in s["best_params"].items()),
        "prediction (MPa)": round(float(model.predict(x)[0]), 2),
        "test RMSE (MPa)": round(s["test"]["rmse"], 2),
    })
out = pd.DataFrame(out)
best = out.sort_values("test RMSE (MPa)").iloc[0]
st.metric(f"Best model by test RMSE: {best['model']}", f"{best['prediction (MPa)']} MPa",
          help="Typical error is about the model's test RMSE.")
st.dataframe(out, hide_index=True)
st.caption("A prediction should be read as roughly +/- the test RMSE. This is a statistical "
           "model of one dataset, not an engineering design tool.")
