"""Loading, cleaning and splitting the Concrete Compressive Strength data."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


def load_raw() -> pd.DataFrame:
    """Load the raw dataset (CSV or Excel) and validate its columns."""
    path = C.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first "
            "(or `python download_data.py --from-file <Concrete_Data.xls>`)."
        )
    df = pd.read_excel(path) if path.suffix in {".xls", ".xlsx"} else pd.read_csv(path)
    missing = set(C.FEATURES + [C.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df[C.FEATURES + [C.TARGET]].astype(float)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows (they could leak across train/test)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def make_splits(save: bool = True):
    """Create the one fixed train/test split used by every model."""
    df, n_dropped = clean(load_raw())
    X, y = df[C.FEATURES], df[C.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.SEED
    )
    if save:
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{C.TARGET: y_train}).to_csv(C.PROCESSED_DIR / "train.csv", index=False)
        X_test.assign(**{C.TARGET: y_test}).to_csv(C.PROCESSED_DIR / "test.csv", index=False)
    print(f"[data] {len(df)} rows after dropping {n_dropped} duplicate(s): "
          f"{len(X_train)} train / {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def load_split():
    """Load the saved split (used by models and the app); build it if absent."""
    tr, te = C.PROCESSED_DIR / "train.csv", C.PROCESSED_DIR / "test.csv"
    if not (tr.exists() and te.exists()):
        return make_splits(save=True)
    train, test = pd.read_csv(tr), pd.read_csv(te)
    return (train[C.FEATURES], test[C.FEATURES], train[C.TARGET], test[C.TARGET])
