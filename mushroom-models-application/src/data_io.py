from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .config import DEFAULT_DATA_PATH, TARGET_COLUMN

def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {path}.")
    return df

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    X = X.replace("?", np.nan)
    y = df[TARGET_COLUMN].map({"e": 0, "p": 1})
    if y.isna().any():
        raise ValueError("Target column contains unexpected labels.")
    return X, y.astype(int)
