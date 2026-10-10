"""Single source of truth for paths, seeds and hyperparameter grids."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"

# Override with the CONCRETE_DATA_PATH environment variable (used by tests).
RAW_DATA_PATH = Path(os.environ.get("CONCRETE_DATA_PATH", RAW_DIR / "concrete.csv"))

# ------------------------------------------------------------ experiment
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ------------------------------------------------------------------ data
FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]
TARGET = "strength"

# (display name, unit, plain-language meaning)
FEATURE_INFO = {
    "cement": ("Cement", "kg/m³", "Binder that reacts with water to harden. Strongly raises strength."),
    "slag": ("Blast furnace slag", "kg/m³", "Steel-industry by-product replacing part of the cement. Acts slowly."),
    "fly_ash": ("Fly ash", "kg/m³", "Coal-power by-product, another cement substitute. Helps mostly at later ages."),
    "water": ("Water", "kg/m³", "More water makes the mix weaker and more porous."),
    "superplasticizer": ("Superplasticizer", "kg/m³", "Additive allowing a workable mix with less water."),
    "coarse_aggregate": ("Coarse aggregate", "kg/m³", "Gravel / crushed stone. Weak direct effect."),
    "fine_aggregate": ("Fine aggregate", "kg/m³", "Sand. Weak direct effect."),
    "age": ("Age", "days", "Curing time (1-365). Strength rises fast, then flattens: a non-linear effect."),
}
TARGET_LABEL = "Compressive strength (MPa)"

# ----------------------------------------------------------- hyperparameters
DEGREES = [1, 2, 3, 4]
MAX_DEGREE = max(DEGREES)  # cap used by the app as well

ALPHAS = np.logspace(-4, 3, 30)          # Ridge / Lasso
ALPHAS_COARSE = np.logspace(-3, 2, 12)   # Elastic Net (3 hyperparameters -> coarser)
L1_RATIOS = [0.2, 0.5, 0.8]              # Elastic Net

LASSO_MAX_ITER = 20_000
LASSO_TOL = 1e-3

# Gradient-descent experiment (from-scratch implementation)
GD_DEGREE = 2
GD_LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.5, 2.0]  # 2.0 is expected to diverge
GD_ITERATIONS = [100, 500, 2000]
