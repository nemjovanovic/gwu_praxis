# ================================================================
# Data Loader for Base Models
# ================================================================
#
# Adapted from superlearner/v1/data_loader.py
# Adds support for CH (crisis history) column needed by Rule of Thumb
#
# ================================================================

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from .config import (
    SPLITS_BASE,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    CH_COL,
    T_MIN,
    get_t_max,
)


def get_split_dir(dataset: str, horizon: int, group: str) -> str:
    """
    Get the base directory containing train/test splits (flat layout).
    
    Args:
        dataset: 'baseline' or 'expanded'
        horizon: 2, 5, or 10
        group: 'ALL', 'EME', or 'LIC'
    
    Returns:
        Path to the splits base directory (data/final)
    """
    return SPLITS_BASE


def load_split(dataset: str, horizon: int, group: str, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test DataFrames for a specific year.
    
    Args:
        dataset: 'baseline' or 'expanded'
        horizon: 2, 5, or 10
        group: 'ALL', 'EME', or 'LIC'
        year: Test year (e.g., 2010)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = os.path.join(SPLITS_BASE, "train", f"train_{dataset}_h{horizon}_{group}_t{year}.csv")
    test_path = os.path.join(SPLITS_BASE, "test", f"test_{dataset}_h{horizon}_{group}_t{year}.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get feature column names (all columns except non-feature columns).
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        List of feature column names
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def extract_Xy(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature matrix X, target y, and country codes from DataFrame.
    
    Args:
        df: DataFrame with features and target
        feature_cols: Optional list of feature columns (auto-detected if None)
    
    Returns:
        Tuple of (X, y, countries) where:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
            countries: numpy array of country codes for fold splitting
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    countries = df[COUNTRY_COL].values
    
    # Handle any NaN values in features (fill with column mean)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
    
    return X, y, countries


def extract_Xy_with_ch(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature matrix X, target y, country codes, and CH (crisis history).
    
    Used by Rule of Thumb model which needs CH column.
    
    Args:
        df: DataFrame with features, target, and CH column
        feature_cols: Optional list of feature columns (auto-detected if None)
    
    Returns:
        Tuple of (X, y, countries, ch) where:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
            countries: numpy array of country codes
            ch: numpy array of crisis history values
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    countries = df[COUNTRY_COL].values
    
    # Extract CH column if present
    if CH_COL in df.columns:
        ch = df[CH_COL].values.astype(np.float32)
    else:
        ch = np.zeros(len(df), dtype=np.float32)
    
    # Handle any NaN values in features
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
    
    return X, y, countries, ch


def get_available_years(dataset: str, horizon: int, group: str) -> List[int]:
    """
    Get list of available test years for a given configuration.
    
    Args:
        dataset: 'baseline' or 'expanded'
        horizon: 2, 5, or 10
        group: 'ALL', 'EME', or 'LIC'
    
    Returns:
        List of available test years
    """
    train_dir = os.path.join(SPLITS_BASE, "train")
    test_dir = os.path.join(SPLITS_BASE, "test")
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Splits directory not found: {SPLITS_BASE} (expected train/ and test/ subdirs)")
    
    years = []
    t_max = get_t_max(horizon)
    
    for t in range(T_MIN, t_max + 1):
        train_path = os.path.join(train_dir, f"train_{dataset}_h{horizon}_{group}_t{t}.csv")
        test_path = os.path.join(test_dir, f"test_{dataset}_h{horizon}_{group}_t{t}.csv")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            years.append(t)
    
    return sorted(years)


def create_country_folds(
    countries: np.ndarray,
    n_folds: int,
    random_state: int = 1234
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create country-based cross-validation folds.
    
    Each fold holds out a subset of countries (not random samples).
    This prevents data leakage across countries.
    Uses country-based K-fold CV to prevent data leakage.
    
    Args:
        countries: Array of country codes for each sample
        n_folds: Number of folds
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    np.random.seed(random_state)
    
    unique_countries = np.unique(countries)
    np.random.shuffle(unique_countries)
    
    # Split countries into n_folds groups
    country_folds = np.array_split(unique_countries, n_folds)
    
    folds = []
    for i in range(n_folds):
        val_countries = set(country_folds[i])
        val_mask = np.array([c in val_countries for c in countries])
        
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        folds.append((train_idx, val_idx))
    
    return folds


class DataLoader:
    """
    High-level data loader for Base Models training.
    """
    
    def __init__(self, dataset: str, horizon: int, group: str):
        """
        Initialize data loader.
        
        Args:
            dataset: 'baseline' or 'expanded'
            horizon: 2, 5, or 10
            group: 'ALL', 'EME', or 'LIC'
        """
        self.dataset = dataset
        self.horizon = horizon
        self.group = group
        self.split_dir = get_split_dir(dataset, horizon, group)  # SPLITS_BASE
        self.available_years = get_available_years(dataset, horizon, group)
        self._feature_cols = None
    
    @property
    def feature_cols(self) -> List[str]:
        """Get feature columns (cached after first access)."""
        if self._feature_cols is None:
            # Load first available year to get columns
            if self.available_years:
                train_df, _ = self.load_year(self.available_years[0])
                self._feature_cols = get_feature_columns(train_df)
        return self._feature_cols
    
    def load_year(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test DataFrames for a specific year."""
        return load_split(self.dataset, self.horizon, self.group, year)
    
    def load_year_Xy(
        self, year: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train and test data as numpy arrays for a specific year.
        
        Returns:
            Tuple of (X_train, y_train, countries_train, X_test, y_test, countries_test)
        """
        train_df, test_df = self.load_year(year)
        
        X_train, y_train, countries_train = extract_Xy(train_df, self.feature_cols)
        X_test, y_test, countries_test = extract_Xy(test_df, self.feature_cols)
        
        return X_train, y_train, countries_train, X_test, y_test, countries_test
    
    def load_year_Xy_with_ch(
        self, year: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train and test data with CH column for Rule of Thumb model.
        
        Returns:
            Tuple of (X_train, y_train, countries_train, ch_train,
                      X_test, y_test, countries_test, ch_test)
        """
        train_df, test_df = self.load_year(year)
        
        X_train, y_train, countries_train, ch_train = extract_Xy_with_ch(train_df, self.feature_cols)
        X_test, y_test, countries_test, ch_test = extract_Xy_with_ch(test_df, self.feature_cols)
        
        return X_train, y_train, countries_train, ch_train, X_test, y_test, countries_test, ch_test
    
    def __repr__(self) -> str:
        return (
            f"DataLoader(dataset='{self.dataset}', horizon={self.horizon}, "
            f"group='{self.group}', years={len(self.available_years)})"
        )
