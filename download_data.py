"""Download the UCI Concrete Compressive Strength dataset to data/raw/concrete.csv.

Source: I-Cheng Yeh (1998), UCI Machine Learning Repository (dataset id 165).

Usage:
    python download_data.py                       # download automatically
    python download_data.py --from-file path.xls  # use a file you already have
"""
from __future__ import annotations

import argparse
import sys
import urllib.request

import pandas as pd

from src import config as C

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
COLUMNS = C.FEATURES + [C.TARGET]  # the file lists the 9 columns in this order


def _from_excel(path) -> pd.DataFrame:
    return pd.read_excel(path)


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=165)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", help="path to an existing Concrete_Data.xls/.csv")
    args = ap.parse_args()

    df = None
    if args.from_file:
        df = pd.read_csv(args.from_file) if args.from_file.endswith(".csv") else _from_excel(args.from_file)
    else:
        try:
            tmp = C.RAW_DIR / "Concrete_Data.xls"
            C.RAW_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, tmp)
            df = _from_excel(tmp)
            tmp.unlink()
        except Exception as e:  # network / format problems -> fall back
            print(f"Direct download failed ({e}); trying ucimlrepo ...")
            try:
                df = _from_ucimlrepo()
            except Exception as e2:
                sys.exit(f"Could not download the data ({e2}).\n"
                         "Download Concrete_Data.xls manually from the UCI repository and run:\n"
                         "    python download_data.py --from-file Concrete_Data.xls")

    if df.shape[1] != len(COLUMNS):
        sys.exit(f"Expected {len(COLUMNS)} columns, found {df.shape[1]}.")
    df.columns = COLUMNS  # original names are long; map by position
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RAW_DIR / "concrete.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
