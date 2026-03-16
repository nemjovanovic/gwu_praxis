# ================================================================
# Base Forecaster (abstract)
# ================================================================
#
# All forecasters predict the next period's values for a set of
# macro variables given historical sequences.
# ================================================================

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import joblib


class BaseForecaster(ABC):
    """Abstract base class for macro-variable forecasters."""

    def __init__(
        self,
        name: str,
        forecast_vars: List[str],
        seq_len: int,
        random_state: int = 1234,
    ):
        self.name = name
        self.forecast_vars = forecast_vars
        self.n_vars = len(forecast_vars)
        self.seq_len = seq_len
        self.random_state = random_state

        self._is_fitted = False
        self.model = None
        self.mean_ = None
        self.std_ = None

    # ------ normalisation ------
    def _fit_normalization(self, X: np.ndarray) -> None:
        if X.shape[0] == 0:
            # No data: use neutral normalization (identity transform)
            n_vars = X.shape[-1] if X.ndim > 1 else 1
            self.mean_ = np.zeros(n_vars, dtype=np.float32)
            self.std_ = np.ones(n_vars, dtype=np.float32)
            return
        X_flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.mean(X_flat, axis=0)
        self.std_ = np.std(X_flat, axis=0)
        # Replace zero std with 1 to avoid division by zero
        self.std_[self.std_ < 1e-8] = 1.0

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / (self.std_ + 1e-8)

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        return X * (self.std_ + 1e-8) + self.mean_

    # ------ interface ------
    @abstractmethod
    def fit(self, X_seq, y_next, X_val=None, y_val=None, verbose=True):
        ...

    @abstractmethod
    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        ...

    # ------ forecast features ------
    def forecast_features(self, X_current: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        forecasts = self.predict(X_seq)
        changes = forecasts - X_current
        n = forecasts.shape[0]
        features = np.zeros((n, 2 * self.n_vars), dtype=np.float32)
        for i in range(self.n_vars):
            features[:, 2 * i] = forecasts[:, i]
            features[:, 2 * i + 1] = changes[:, i]
        return features

    def get_feature_names(self) -> List[str]:
        names = []
        for var in self.forecast_vars:
            names.append(f"{var}_forecast_{self.name}")
            names.append(f"{var}_change_{self.name}")
        return names

    # ------ persistence ------
    def save(self, path: str) -> None:
        metadata = {
            "name": self.name,
            "forecast_vars": self.forecast_vars,
            "seq_len": self.seq_len,
            "random_state": self.random_state,
            "mean_": self.mean_,
            "std_": self.std_,
        }
        joblib.dump(metadata, f"{path}_metadata.pkl")
        self._save_model(path)

    def load(self, path: str) -> "BaseForecaster":
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.name = metadata["name"]
        self.forecast_vars = metadata["forecast_vars"]
        self.n_vars = len(self.forecast_vars)
        self.seq_len = metadata["seq_len"]
        self.random_state = metadata["random_state"]
        self.mean_ = metadata["mean_"]
        self.std_ = metadata["std_"]
        self._load_model(path)
        self._is_fitted = True
        return self

    @abstractmethod
    def _save_model(self, path: str) -> None: ...

    @abstractmethod
    def _load_model(self, path: str) -> None: ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
