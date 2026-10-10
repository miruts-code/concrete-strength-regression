"""Run the whole pipeline: split data -> train every model -> build report.

    python run_all.py                       # everything
    python run_all.py --skip-elastic-net    # faster
    python run_all.py --models ridge lasso  # a subset
"""
from __future__ import annotations

import argparse
import time
import warnings

from sklearn.exceptions import ConvergenceWarning

from models.train import MODELS, train
from src.data import make_splits

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--skip-elastic-net", action="store_true")
    ap.add_argument("--skip-scratch", action="store_true")
    args = ap.parse_args()
    models = [m for m in args.models if not (args.skip_elastic_net and m == "elastic_net")]

    t0 = time.time()
    make_splits(save=True)  # one fixed split shared by every model

    for key in models:
        train(key)
    if not args.skip_scratch:
        from models.scratch.verify import main as verify_scratch

        verify_scratch()

    from src.report import build_report

    build_report()
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Launch the app with:\n"
          "    streamlit run app/app.py")


if __name__ == "__main__":
    main()
