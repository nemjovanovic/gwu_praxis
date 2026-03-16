# ================================================================
# Data Loader for Superlearners
# ================================================================
#
# Loads augmented train/test CSVs from traintest_transforms/.
# Thin wrapper mirroring basemodels/data_loader.py, but reading
# from the forecast-augmented directory.
# ================================================================

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from superlearners.config import (
    TRANSFORMS_DIR,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    T_MIN,
    get_t_max,
)


def load_split(
    dataset: str, horizon: int, group: str, year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load augmented train and test DataFrames for a specific year."""
    train_path = os.path.join(
        TRANSFORMS_DIR, "train", f"train_{dataset}_h{horizon}_{group}_t{year}.csv",
    )
    test_path = os.path.join(
        TRANSFORMS_DIR, "test", f"test_{dataset}_h{horizon}_{group}_t{year}.csv",
    )
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Augmented train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Augmented test file not found: {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def get_available_years(dataset: str, horizon: int, group: str) -> List[int]:
    """Discover years available in traintest_transforms/."""
    train_dir = os.path.join(TRANSFORMS_DIR, "train")
    test_dir = os.path.join(TRANSFORMS_DIR, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        return []
    t_max = get_t_max(horizon)
    years = []
    for t in range(T_MIN, t_max + 1):
        tp = os.path.join(train_dir, f"train_{dataset}_h{horizon}_{group}_t{t}.csv")
        ep = os.path.join(test_dir, f"test_{dataset}_h{horizon}_{group}_t{t}.csv")
        if os.path.exists(tp) and os.path.exists(ep):
            years.append(t)
    return sorted(years)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """All columns except non-feature columns (includes forecast features)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def extract_Xy(
    df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (X, y, countries) from a DataFrame.

    Args:
        df: Augmented DataFrame.
        feature_cols: Which columns to use as features.

    Returns:
        (X, y, countries)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    countries = df[COUNTRY_COL].values
    # fill NaN with column mean
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        nans = np.where(np.isnan(X))
        X[nans] = np.take(col_means, nans[1])
    return X, y, countries
