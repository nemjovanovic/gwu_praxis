# ================================================================
# PatchTST Forecaster
# ================================================================
#
# Patch Time Series Transformer (channel-independent).
# One Keras model per macro variable, with patching + Transformer encoder.
# Reference: "A Time Series is Worth 64 Words" (Nie et al., ICLR 2023)
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
from superlearners.config import PATCHTST_CONFIG, FORECAST_VARS, SEQUENCE_LEN


# ----------------------------------------------------------------
# Custom Keras layers
# ----------------------------------------------------------------

class PatchTSTBlock(layers.Layer):
    """PatchTST encoder block (pre-norm, multi-head attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
        )
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_norm = self.layernorm1(x)
        attn = self.mha(x_norm, x_norm, training=training)
        attn = self.dropout1(attn, training=training)
        x = x + attn
        x_norm = self.layernorm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        return x + ffn_out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate,
        })
        return cfg


# ----------------------------------------------------------------
# PatchTST Forecaster
# ----------------------------------------------------------------

class PatchTSTForecaster(BaseForecaster):
    """
    PatchTST forecaster with channel independence.
    Trains a separate Transformer per variable, using patching.
    """

    def __init__(
        self,
        forecast_vars: List[str] = None,
        seq_len: int = SEQUENCE_LEN,
        patch_len: int = PATCHTST_CONFIG["patch_len"],
        stride: int = PATCHTST_CONFIG["stride"],
        d_model: int = PATCHTST_CONFIG["d_model"],
        n_heads: int = PATCHTST_CONFIG["n_heads"],
        n_layers: int = PATCHTST_CONFIG["n_layers"],
        d_ff: int = PATCHTST_CONFIG["d_ff"],
        dropout: float = PATCHTST_CONFIG["dropout"],
        learning_rate: float = PATCHTST_CONFIG["learning_rate"],
        batch_size: int = PATCHTST_CONFIG["batch_size"],
        epochs: int = PATCHTST_CONFIG["epochs"],
        early_stopping_patience: int = PATCHTST_CONFIG["early_stopping_patience"],
        random_state: int = PATCHTST_CONFIG["random_state"],
    ):
        forecast_vars = forecast_vars or FORECAST_VARS
        super().__init__(
            name="PatchTST",
            forecast_vars=forecast_vars,
            seq_len=seq_len,
            random_state=random_state,
        )
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.models = {}

    def _build_model(self, var_idx: int) -> keras.Model:
        tf.random.set_seed(self.random_state + var_idx)
        inputs = layers.Input(shape=(self.seq_len, 1))
        n_patches = (self.seq_len - self.patch_len) // self.stride + 1

        # Manual patching
        patch_list = []
        for i in range(n_patches):
            start = i * self.stride
            patch = inputs[:, start:start + self.patch_len, :]
            patch = layers.Flatten()(patch)
            patch_list.append(patch)

        if len(patch_list) > 1:
            x = layers.Lambda(lambda p: tf.stack(p, axis=1))(patch_list)
        else:
            x = layers.Lambda(lambda p: tf.expand_dims(p[0], axis=1))(patch_list)

        x = layers.Dense(self.d_model)(x)
        pos_embed = layers.Embedding(n_patches, self.d_model)(tf.range(n_patches))
        x = x + pos_embed
        x = layers.Dropout(self.dropout)(x)

        for _ in range(self.n_layers):
            x = PatchTSTBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                d_ff=self.d_ff, dropout=self.dropout,
            )(x)

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.d_ff, activation="gelu")(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
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
    ) -> "PatchTSTForecaster":
        if verbose:
            print("      Training PatchTST (channel-independent)...")
        self._fit_normalization(X_seq)
        X_seq_norm = self._normalize(X_seq)
        y_next_norm = self._normalize(y_next)
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize(X_val)
            y_val_norm = self._normalize(y_val)

        for i, var in enumerate(self.forecast_vars):
            if verbose:
                print(f"        Training PatchTST for {var}...")
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
            print(f"      PatchTST training complete ({len(self.forecast_vars)} models)")
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
            model.save(f"{path}_patchtst_{var}.keras")

    def _load_model(self, path: str) -> None:
        self.models = {}
        for var in self.forecast_vars:
            model_path = f"{path}_patchtst_{var}.keras"
            if os.path.exists(model_path):
                self.models[var] = keras.models.load_model(
                    model_path,
                    custom_objects={"PatchTSTBlock": PatchTSTBlock},
                )
