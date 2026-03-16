# ================================================================
# Base Forecaster (abstract) -- Multivariate, Multi-step
# ================================================================
#
# All forecasters predict the next FORECAST_STEPS periods' values
# for all macro variables jointly, given historical sequences.
#
# predict() returns (n_samples, forecast_steps, n_vars)
# forecast_features() returns (n_samples, 2 * n_vars * forecast_steps)
# ================================================================

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import joblib


class BaseForecaster(ABC):
    """Abstract base class for multivariate, multi-step macro forecasters."""

    def __init__(
        self,
        name: str,
        forecast_vars: List[str],
        seq_len: int,
        forecast_steps: int = 2,
        random_state: int = 1234,
    ):
        self.name = name
        self.forecast_vars = forecast_vars
        self.n_vars = len(forecast_vars)
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.random_state = random_state

        self._is_fitted = False
        self.model = None
        self.mean_ = None
        self.std_ = None

    # ------ normalisation ------
    def _fit_normalization(self, X: np.ndarray) -> None:
        if X.shape[0] == 0:
            n_vars = X.shape[-1] if X.ndim > 1 else 1
            self.mean_ = np.zeros(n_vars, dtype=np.float32)
            self.std_ = np.ones(n_vars, dtype=np.float32)
            return
        X_flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.mean(X_flat, axis=0)
        self.std_ = np.std(X_flat, axis=0)
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
        """
        Returns:
            predictions of shape (n_samples, forecast_steps, n_vars)
        """
        ...

    # ------ forecast features ------
    def forecast_features(self, X_current: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        """
        Generate forecast features for the RF classifier.

        Args:
            X_current: (n_samples, n_vars) -- current period values
            X_seq: (n_samples, seq_len, n_vars) -- historical sequences

        Returns:
            (n_samples, 2 * n_vars * forecast_steps) feature array.
            For each variable, for each step:
              - forecast value at step s
              - change = forecast at step s minus current value
        """
        forecasts = self.predict(X_seq)  # (n, forecast_steps, n_vars)
        n = forecasts.shape[0]
        n_features = 2 * self.n_vars * self.forecast_steps
        features = np.zeros((n, n_features), dtype=np.float32)

        idx = 0
        for i in range(self.n_vars):
            for s in range(self.forecast_steps):
                features[:, idx] = forecasts[:, s, i]          # forecast
                features[:, idx + 1] = forecasts[:, s, i] - X_current[:, i]  # change
                idx += 2

        return features

    def get_feature_names(self) -> List[str]:
        names = []
        for var in self.forecast_vars:
            for step in range(1, self.forecast_steps + 1):
                names.append(f"{var}_forecast_{step}step_{self.name}")
                names.append(f"{var}_change_{step}step_{self.name}")
        return names

    # ------ persistence ------
    def save(self, path: str) -> None:
        metadata = {
            "name": self.name,
            "forecast_vars": self.forecast_vars,
            "seq_len": self.seq_len,
            "forecast_steps": self.forecast_steps,
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
        self.forecast_steps = metadata.get("forecast_steps", 2)
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
