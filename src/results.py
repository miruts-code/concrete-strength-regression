"""Helpers for reading saved results (used by the report and the app)."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from models.train import MODELS
from src import config as C


def _read_csv(name: str):
    p = C.METRICS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def load_grid(model: str):
    return _read_csv(f"{model}_grid.csv")


def load_per_degree(model: str):
    return _read_csv(f"{model}_per_degree.csv")


def load_summary(model: str):
    p = C.METRICS_DIR / f"{model}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_saved_model(model: str):
    p = C.SAVED_MODELS_DIR / f"{model}.joblib"
    return joblib.load(p) if p.exists() else None


def load_comparison():
    return _read_csv("comparison.csv")


def available_models() -> list[str]:
    return [m for m in MODELS if (C.METRICS_DIR / f"{m}_summary.json").exists()]
