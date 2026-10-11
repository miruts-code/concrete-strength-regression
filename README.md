# Concrete Strength: Polynomial Regression, Regularization & Streamlit

Predict the **compressive strength of concrete (MPa)** from its mix recipe and age, and show **why** model flexibility has to be controlled: polynomial degrees 1–4, OLS vs. Ridge vs. Lasso vs. Elastic Net, hyperparameters chosen by 5-fold cross-validation, plus a from-scratch NumPy implementation (normal equation and gradient descent) verified against scikit-learn.

**Live demo:** [https://concrete-strength-regression-demo.streamlit.app/](https://concrete-strength-regression-demo.streamlit.app/)

---

## Dataset

UCI **Concrete Compressive Strength** (1,030 rows, 8 inputs, 1 target).

I-C. Yeh (1998), *Modeling of strength of high-performance concrete using artificial neural networks*, Cement and Concrete Research 28(12), 1797–1808. UCI Machine Learning Repository.

| Feature | Unit | Meaning |
|:--|:--:|:--|
| cement | kg/m³ | binder; strongly raises strength |
| slag | kg/m³ | blast furnace slag, slow-acting cement substitute |
| fly_ash | kg/m³ | coal-power by-product, cement substitute |
| water | kg/m³ | more water -> weaker, more porous mix |
| superplasticizer | kg/m³ | additive that allows less water |
| coarse_aggregate | kg/m³ | gravel / crushed stone |
| fine_aggregate | kg/m³ | sand |
| age | days | curing time (1-365); non-linear effect |

**Target:** `strength`, compressive strength in MPa.

---

## Method

1. **One fixed split:** 80% train / 20% test (seed 42). Exact duplicate rows are dropped first so they cannot leak across the split.

2. **Pipeline** (`src/features.py`): `StandardScaler -> PolynomialFeatures(degree) -> StandardScaler -> estimator`.

   Scaling is inside the pipeline, so cross-validation refits it on each training fold (no leakage), and the second scaler gives every polynomial term unit variance so the penalty is fair.

3. **Polynomial degrees 1–4** (8 / 44 / 164 / 494 terms, excluding the intercept).

4. **Models** (one `ModelSpec` each, defined in `models/train.py`):
   - `ols`: no penalty. Tunes degree only.
   - `ridge`: squared error + alpha * sum(beta²). Shrinks all coefficients.
   - `lasso`: squared error + alpha * sum(|beta|). Sets many coefficients exactly to zero.
   - `elastic_net`: mix of both (`l1_ratio`).
   - `scratch` (NumPy only, OLS and Ridge): the explicit normal equation `(X'X + alpha*I) w = X'y` and gradient descent. `verify.py` compares the normal equation with scikit-learn and reports cond(X'X): forming X'X squares the condition number, so agreement weakens as the degree grows, and a pseudo-inverse fallback is used if X'X is singular. Lasso and Elastic Net are scikit-learn only (L1 has no closed form).

5. **Tuning:** grid search with **5-fold CV on the training set only**, selection metric = mean CV RMSE.

   Grids: degree 1–4; alpha = 30 log-spaced values in [1e-4, 1e3] (Elastic Net: 12 values × 3 `l1_ratio`).

   The simplest model within one standard error of the best is also recorded (`one_se_params`).

6. **Final evaluation:** the CV winner is refit on the full training set and scored **once** on the test set.

---

## Repository layout

```text
concrete-strength-regression/
├── app/                              # Streamlit app
│   ├── pages/
│   │   ├── 1_Explore_Data.py          # distributions, strength vs. each feature, correlations (training data only)
│   │   ├── 2_Predict.py               # enter a mix recipe, get a prediction from every saved model
│   │   ├── 3_Degree_Comparison.py     # train / CV / test error by degree, plus live training of every degree
│   │   ├── 4_Regularization_Lab.py    # choose model, degree and alpha live; coefficient paths and grid heatmaps
│   │   └── 5_Gradient_Descent.py      # compare learning rates against the exact normal-equation solution
│   ├── __init__.py                   # empty; makes app a package
│   ├── app.py                         # home page: feature glossary and final model comparison
│   └── common.py                      # cached data loading and live cross-validation helpers
│
├── data/
│   ├── processed/                     # *.csv: fixed 80/20 train/test split (committed so the app can deploy)
│   └── raw/                           # downloaded dataset (git-ignored)
│
├── models/
│   ├── scratch/                       # NumPy implementation of OLS and Ridge
│   │   ├── __init__.py                # empty
│   │   ├── model.py                   # normal equation (with pseudo-inverse fallback) and gradient descent
│   │   └── verify.py                  # compares the normal equation with scikit-learn; gradient-descent experiment
│   ├── __init__.py                    # empty
│   └── train.py                       # ModelSpec and MODELS (OLS, Ridge, Lasso, Elastic Net); training command line
│
├── notebooks/
│   ├── 01_eda.ipynb                   # data exploration (training split only)
│   └── 02_results_analysis.ipynb      # results, residual analysis, error breakdown, log(age) experiment
│
├── results/                           # written by run_all.py and notebook 02; committed
│   ├── figures/                       # *.png: error curves, heatmaps, residual and error-breakdown plots
│   ├── metrics/                       # *.csv, *.json: model comparison, grid-search scores, summaries, diagnostics
│   └── saved_models/                  # *.joblib: one fitted pipeline per model
│
├── src/                               # shared code
│   ├── __init__.py                    # empty
│   ├── analysis.py                    # coefficient paths for Ridge and Lasso
│   ├── config.py                      # paths, seed, CV folds, degree and alpha grids, feature descriptions
│   ├── data.py                        # load the data, drop duplicates, make the train/test split
│   ├── diagnostics.py                 # residual analysis, error breakdowns, log(age) experiment
│   ├── evaluate.py                    # RMSE, MAE, R² and coefficient counts
│   ├── features.py                    # scale -> polynomial expansion -> scale pipeline
│   ├── plots.py                       # all matplotlib figures
│   ├── report.py                      # builds the comparison table and static figures
│   ├── results.py                     # loaders for saved metrics and models
│   ├── runner.py                      # tune, refit the winner, score the test set once, save results
│   └── tuning.py                      # grid search with 5-fold CV and best-parameter selection
│
├── tests/
│   ├── conftest.py                    # puts the project root on the import path
│   ├── test_pipeline.py               # model specs, grids, pipeline, tuning rules, diagnostics helpers
│   └── test_scratch.py                # NumPy implementation matches scikit-learn; gradient descent behaviour
│
├── .gitignore                         # files Git should not track
├── download_data.py                   # download the UCI dataset to data/raw/concrete.csv
├── README.md
├── requirements.txt                   # Python dependencies
└── run_all.py                         # split -> train every model -> comparison table and figures
```

---

## Adding a model

Add one `ModelSpec` entry to `MODELS` in `models/train.py` (label, scikit-learn estimator, the hyperparameters to tune, any fixed arguments). Training, the app, the notebooks and the report read it from there; nothing else needs editing.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python download_data.py          # or: python download_data.py --from-file Concrete_Data.xls
python run_all.py                # add --skip-elastic-net for a faster run
streamlit run app/app.py
pytest                           # optional: unit tests
```

Run time: Lasso is the slowest part. Real data may differ somewhat.

The Streamlit app loads saved results and trains single configurations live in about a second (Lasso/Elastic Net at degree 4 with a tiny alpha can take longer).

One model only: `python -m models.train --model ridge` (or `--all`). The scratch checks: `python -m models.scratch.verify`.

---

## Notebooks

Run them in VS Code (Python + Jupyter extensions, kernel = `.venv`), after `download_data.py` and `run_all.py`.

- `01_eda.ipynb`: summary statistics, histograms, strength vs. age, correlation heat map (training data only).
- `02_results_analysis.ipynb`: saved results, **residual analysis**, where the model fails (largest errors, error by age and by strength), a **`log(age)` experiment**, and a report-only test-set check. All diagnostics use out-of-fold predictions on the training set, so the test set never influences decisions.

Outputs live inside the saved `.ipynb` (run, then Ctrl+S, then commit). The analysis cells also write:

- `results/figures/{residual_diagnostics,error_breakdown,log_age_experiment}.png`
- `results/metrics/{error_by_age,error_by_strength,log_age_experiment}.csv`

---

## Streamlit app

| Page | What it does |
|:--|:--|
| Home | Feature glossary and final model comparison |
| Explore Data | Distributions, strength vs. feature, strength vs. age, correlations |
| Predict | Enter a mix recipe, get predictions from all tuned models |
| Degree Comparison | Train vs. CV vs. test error by degree; train every degree live |
| Regularization Lab | Choose model, degree, alpha live; coefficient bars and paths; precomputed heatmaps |
| Gradient Descent | Compare learning rates, watch convergence or divergence vs. the exact solution |

The test score is deliberately hidden behind a button in the lab: tune with cross-validation, look at the test set once.

---

## Results

Raw scores are in `results/metrics/`.

| Model | Best settings (CV) | CV RMSE | Test RMSE | Test MAE | Test R² |
|:--|:--|--:|--:|--:|--:|
| OLS | degree=3 | 7.1718120426546745 | 5.948419935549056 | 4.502015923437978 | 0.8813933627982411 |
| Ridge | degree=3, alpha=0.4175 | 6.448129323389783 | 6.266019729596481 | 4.618434276716982 | 0.8683898852379088 |
| Lasso | degree=4, alpha=0.02593 | 6.002197959133221 | 6.505348763467428 | 4.641076159102343 | 0.8581442574292415 |
| Elastic Net | degree=3, alpha=0.008111, l1_ratio=0.8 | 6.336879564125409 | 6.7205991015254005 | 4.800976972106227 | 0.8486014447159813 |

Figures are in `results/figures/` (`degree_curves.png`, `*_heatmap.png`, `*_alpha_curves.png`).

---

## Deploying on Streamlit Community Cloud

1. Run `python run_all.py` locally, then commit `results/` and `data/processed/`.
2. Push to GitHub. On share.streamlit.io choose the repo and set the main file to `app/app.py`.
3. then deploy .

---

## Limitations

- Polynomial models extrapolate poorly outside the training range (sliders are limited to it).
- The dataset comes from lab experiments, so this is a statistical model, not an engineering design tool.
- Strength vs. age is log-like; a `log(age)` feature might beat polynomials in age (a natural extension).