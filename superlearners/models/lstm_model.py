# ================================================================
# LSTM Forecaster
# ================================================================
#
# Channel-independent LSTM: one Keras LSTM per macro variable.
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

from superlearners.models.base_forecaster import BaseForecaster
from superlearners.config import LSTM_CONFIG, FORECAST_VARS, SEQUENCE_LEN


class LSTMForecaster(BaseForecaster):
    """Channel-independent LSTM forecaster (one model per variable)."""

    def __init__(
        self,
        forecast_vars: List[str] = None,
        seq_len: int = SEQUENCE_LEN,
        hidden_size: int = LSTM_CONFIG["hidden_size"],
        num_layers: int = LSTM_CONFIG["num_layers"],
        dropout: float = LSTM_CONFIG["dropout"],
        learning_rate: float = LSTM_CONFIG["learning_rate"],
        batch_size: int = LSTM_CONFIG["batch_size"],
        epochs: int = LSTM_CONFIG["epochs"],
        early_stopping_patience: int = LSTM_CONFIG["early_stopping_patience"],
        random_state: int = LSTM_CONFIG["random_state"],
    ):
        forecast_vars = forecast_vars or FORECAST_VARS
        super().__init__(
            name="LSTM",
            forecast_vars=forecast_vars,
            seq_len=seq_len,
            random_state=random_state,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.models = {}

    def _build_model(self, var_idx: int) -> keras.Model:
        tf.random.set_seed(self.random_state + var_idx)
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.seq_len, 1)))
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(layers.LSTM(
                self.hidden_size,
                return_sequences=return_sequences,
                dropout=self.dropout if i < self.num_layers - 1 else 0,
            ))
        model.add(layers.Dense(1))
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
    ) -> "LSTMForecaster":
        self._fit_normalization(X_seq)
        X_seq_norm = self._normalize(X_seq)
        y_next_norm = self._normalize(y_next)
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize(X_val)
            y_val_norm = self._normalize(y_val)

        for i, var in enumerate(self.forecast_vars):
            if verbose:
                print(f"      Training LSTM for {var}...")
            X_var = X_seq_norm[:, :, i:i + 1]
            y_var = y_next_norm[:, i:i + 1]
            model = self._build_model(i)
            callbacks = []
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_var = X_val_norm[:, :, i:i + 1]
                y_val_var = y_val_norm[:, i:i + 1]
                validation_data = (X_val_var, y_val_var)
                callbacks.append(keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                ))
            model.fit(
                X_var, y_var,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0,
            )
            self.models[var] = model

        self._is_fitted = True
        if verbose:
            print(f"      LSTM training complete ({len(self.forecast_vars)} models)")
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before predicting.")
        X_seq_norm = self._normalize(X_seq)
        n_samples = X_seq.shape[0]
        predictions = np.zeros((n_samples, self.n_vars), dtype=np.float32)
        for i, var in enumerate(self.forecast_vars):
            X_var = X_seq_norm[:, :, i:i + 1]
            pred_norm = self.models[var].predict(X_var, verbose=0)
            predictions[:, i] = pred_norm.flatten()
        predictions = self._denormalize(predictions)
        return predictions

    def _save_model(self, path: str) -> None:
        for var, model in self.models.items():
            model.save(f"{path}_lstm_{var}.keras")

    def _load_model(self, path: str) -> None:
        self.models = {}
        for var in self.forecast_vars:
            model_path = f"{path}_lstm_{var}.keras"
            if os.path.exists(model_path):
                self.models[var] = keras.models.load_model(model_path)
