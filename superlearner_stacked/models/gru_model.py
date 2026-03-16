# ================================================================
# GRU Forecaster -- Multivariate, Multi-step
# ================================================================
#
# Single Keras GRU that takes all macro variables jointly
# and predicts FORECAST_STEPS periods ahead for all variables.
#
# Input:  (batch, seq_len, n_vars) = (batch, 5, 10)
# Output: (batch, forecast_steps * n_vars) reshaped to
#         (batch, forecast_steps, n_vars) = (batch, 2, 10)
#
# Based on the LSTM forecaster architecture with LSTM layers
# replaced by GRU layers.
# ================================================================

import os
import sys
import numpy as np
from typing import Optional, List
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from superlearner_stacked.models.base_forecaster import BaseForecaster
from superlearner_stacked.config import (
    GRU_CONFIG, FORECAST_VARS, SEQUENCE_LEN, FORECAST_STEPS,
)


class GRUForecaster(BaseForecaster):
    """Multivariate GRU forecaster -- one model for all variables jointly."""

    def __init__(
        self,
        forecast_vars: List[str] = None,
        seq_len: int = SEQUENCE_LEN,
        forecast_steps: int = FORECAST_STEPS,
        hidden_size: int = GRU_CONFIG["hidden_size"],
        num_layers: int = GRU_CONFIG["num_layers"],
        dropout: float = GRU_CONFIG["dropout"],
        learning_rate: float = GRU_CONFIG["learning_rate"],
        batch_size: int = GRU_CONFIG["batch_size"],
        epochs: int = GRU_CONFIG["epochs"],
        early_stopping_patience: int = GRU_CONFIG["early_stopping_patience"],
        random_state: int = GRU_CONFIG["random_state"],
    ):
        forecast_vars = forecast_vars or FORECAST_VARS
        super().__init__(
            name="GRU",
            forecast_vars=forecast_vars,
            seq_len=seq_len,
            forecast_steps=forecast_steps,
            random_state=random_state,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self._model = None

    def _build_model(self) -> keras.Model:
        tf.random.set_seed(self.random_state)
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.seq_len, self.n_vars)))
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(layers.GRU(
                self.hidden_size,
                return_sequences=return_sequences,
                dropout=self.dropout if i < self.num_layers - 1 else 0,
            ))
        # Output: forecast_steps * n_vars values, reshaped later
        model.add(layers.Dense(self.forecast_steps * self.n_vars))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def fit(
        self,
        X_seq: np.ndarray,
        y_next: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "GRUForecaster":
        """
        Args:
            X_seq: (n_samples, seq_len, n_vars)
            y_next: (n_samples, forecast_steps, n_vars)
        """
        if verbose:
            print(f"      Training multivariate GRU ({self.n_vars} vars, {self.forecast_steps} steps)...")

        self._fit_normalization(X_seq)
        X_seq_norm = self._normalize(X_seq)

        # Flatten targets for training: (n, forecast_steps, n_vars) -> (n, forecast_steps * n_vars)
        y_flat = self._normalize(y_next.reshape(-1, self.n_vars)).reshape(
            y_next.shape[0], self.forecast_steps * self.n_vars
        )

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize(X_val)
            y_val_flat = self._normalize(y_val.reshape(-1, self.n_vars)).reshape(
                y_val.shape[0], self.forecast_steps * self.n_vars
            )
            validation_data = (X_val_norm, y_val_flat)

        self._model = self._build_model()
        callbacks = []
        if validation_data is not None:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ))

        self._model.fit(
            X_seq_norm, y_flat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        self._is_fitted = True
        if verbose:
            print(f"      GRU training complete (1 joint model)")
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Returns:
            (n_samples, forecast_steps, n_vars)
        """
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before predicting.")
        X_seq_norm = self._normalize(X_seq)
        pred_flat = self._model.predict(X_seq_norm, verbose=0)  # (n, forecast_steps * n_vars)
        # Denormalize: reshape to (n * forecast_steps, n_vars) then back
        n = pred_flat.shape[0]
        pred_reshaped = pred_flat.reshape(n * self.forecast_steps, self.n_vars)
        pred_denorm = self._denormalize(pred_reshaped)
        return pred_denorm.reshape(n, self.forecast_steps, self.n_vars)

    def _save_model(self, path: str) -> None:
        if self._model is not None:
            self._model.save(f"{path}_gru_multivariate.keras")

    def _load_model(self, path: str) -> None:
        model_path = f"{path}_gru_multivariate.keras"
        if os.path.exists(model_path):
            self._model = keras.models.load_model(model_path)
