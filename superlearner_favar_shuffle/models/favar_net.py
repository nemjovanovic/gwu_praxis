# ================================================================
# FAVAR-Net v2: Factor-Augmented VAR with Regime-Switching
# ================================================================
#
# A neural network that recreates the economist's FAVAR structure:
#
#   x_{i,t+h} = sum_k pi_k(f_t) * (W_k * z_{i,t} + b_k)
#               + lambda * GRN(z_{i,t})
#
# Where:
#   z_{i,t} = [local_lags, global_factor_lags, theory_interactions]
#   f_t     = current global factor vector (for regime gating)
#   pi_k    = softmax regime weights conditioned on global state
#   W_k     = linear VAR coefficient matrix for regime k
#   GRN     = small Gated Residual Network (nonlinear correction)
#   lambda  = 0.1 scaling factor (keeps nonlinearity controlled)
#
# v2 changes:
#   - Supports "classify" mode: output_dim=1, sigmoid activation,
#     binary crossentropy loss.  Predicts P(crisis) directly.
#   - Supports "forecast" mode: output_dim=n_vars*steps, linear
#     activation, MSE loss.  (Legacy, kept for compatibility.)
#
# Total parameters: ~4k (appropriate for annual macro data)
# ================================================================

import os
import sys
import numpy as np
from typing import Optional, List, Dict
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from superlearner_favar_shuffle.config import (
    FAVAR_CONFIG, FORECAST_VARS,
)


# ----------------------------------------------------------------
# Custom Keras layer: Mixture of Experts
# ----------------------------------------------------------------

class MixtureOfExperts(layers.Layer):
    """
    K-expert Mixture-of-Experts layer.

    Given z (VAR state) and pi (regime weights from gate):
      output = sum_k pi_k * Expert_k(z)

    Each expert is a linear map: Dense(output_dim)(z).
    """

    def __init__(self, output_dim, n_experts=2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.experts = []

    def build(self, input_shape):
        for k in range(self.n_experts):
            self.experts.append(
                layers.Dense(self.output_dim, use_bias=True, name=f"expert_{k}")
            )
        super().build(input_shape)

    def call(self, inputs):
        z, pi = inputs  # z: (batch, input_dim), pi: (batch, n_experts)
        expert_outs = []
        for k in range(self.n_experts):
            expert_outs.append(self.experts[k](z))  # each: (batch, output_dim)

        # Stack: (batch, n_experts, output_dim)
        stacked = tf.stack(expert_outs, axis=1)

        # Expand pi: (batch, n_experts, 1)
        pi_expanded = tf.expand_dims(pi, axis=-1)

        # Weighted sum: (batch, output_dim)
        return tf.reduce_sum(stacked * pi_expanded, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "n_experts": self.n_experts,
        })
        return config


# ----------------------------------------------------------------
# Custom Keras layer: Scaled Residual GRN
# ----------------------------------------------------------------

class ScaledResidualGRN(layers.Layer):
    """
    Gated Residual Network with a fixed scaling factor.

    GRN(z) = scale * LayerNorm(z_proj + GLU(Dense(ELU(Dense(z)))))

    The scaling factor (default 0.1) ensures the nonlinear residual
    stays small relative to the linear MoE backbone.
    """

    def __init__(self, d_hidden, d_output, scale=0.1, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.scale = scale
        self._dropout_rate = dropout
        self.dense1 = layers.Dense(d_hidden, activation="elu", name="grn_hidden")
        self.dense2 = layers.Dense(d_output * 2, name="grn_glu")  # for GLU split
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(name="grn_ln")
        self.proj = None  # set in build if needed

    def build(self, input_shape):
        if input_shape[-1] != self.d_output:
            self.proj = layers.Dense(self.d_output, name="grn_proj")
        super().build(input_shape)

    def call(self, x, training=False):
        residual = self.proj(x) if self.proj is not None else x
        h = self.dense1(x)
        h = self.dropout(h, training=training)
        h = self.dense2(h)  # (batch, d_output * 2)
        # GLU: split into gate and value, apply sigmoid gate
        gate, value = tf.split(h, 2, axis=-1)  # each (batch, d_output)
        h = tf.sigmoid(gate) * value
        return self.scale * self.layer_norm(residual + h)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_hidden": self.d_hidden,
            "d_output": self.d_output,
            "scale": self.scale,
        })
        return config


# ----------------------------------------------------------------
# FAVAR-Net Keras model builder
# ----------------------------------------------------------------

def build_favar_model(
    input_dim: int,
    n_factors: int,
    output_dim: int,
    task: str = "classify",
    n_experts: int = 2,
    residual_hidden: int = 32,
    residual_scale: float = 0.1,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build the FAVAR-Net Keras Functional API model.

    Args:
        input_dim: dimension of z (local lags + global factor lags + interactions)
        n_factors: dimension of f_current (global factor vector)
        output_dim: 1 for classify, n_vars*steps for forecast
        task: "classify" or "forecast"
        n_experts: number of linear VAR heads (K=2)
        residual_hidden: hidden size for residual GRN
        residual_scale: lambda scaling for nonlinear residual
        dropout: dropout rate

    Returns:
        keras.Model with inputs [z_input, f_input] and output x_hat
    """
    # Inputs
    z_input = layers.Input(shape=(input_dim,), name="z_input")
    f_input = layers.Input(shape=(n_factors,), name="f_input")

    # --- Regime gate: softmax over K experts, conditioned on global state ---
    pi = layers.Dense(n_experts, activation="softmax", name="regime_gate")(f_input)

    # --- K linear VAR expert heads via MoE layer ---
    moe = MixtureOfExperts(output_dim=output_dim, n_experts=n_experts, name="moe")
    x_hat_moe = moe([z_input, pi])  # (batch, output_dim)

    # --- Scaled residual GRN ---
    grn = ScaledResidualGRN(
        d_hidden=residual_hidden,
        d_output=output_dim,
        scale=residual_scale,
        dropout=dropout,
        name="residual_grn",
    )
    x_hat_residual = grn(z_input)  # (batch, output_dim)

    # --- Combine MoE + scaled residual ---
    x_hat = layers.Add(name="combined")([x_hat_moe, x_hat_residual])

    # --- Final activation ---
    if task == "classify":
        # Sigmoid for binary classification
        x_hat = layers.Activation("sigmoid", name="output")(x_hat)
    else:
        # Linear for forecasting (identity -- just rename)
        x_hat = layers.Activation("linear", name="output")(x_hat)

    model = keras.Model(
        inputs=[z_input, f_input],
        outputs=x_hat,
        name="FAVAR_Net",
    )

    return model


# ----------------------------------------------------------------
# FAVARForecaster wrapper
# ----------------------------------------------------------------

class FAVARForecaster:
    """
    FAVAR-Net wrapper supporting both classification and forecasting.

    Handles normalization, training, prediction, persistence,
    and interpretability weight extraction.

    task="classify":
        - output_dim=1, sigmoid activation, BCE loss
        - predict() returns P(crisis) as (n, 1)
        - No target normalization (labels are 0/1)

    task="forecast" (legacy):
        - output_dim=n_vars*steps, linear activation, MSE loss
        - predict() returns denormalized forecasts
    """

    def __init__(
        self,
        task: str = FAVAR_CONFIG["task"],
        forecast_vars: List[str] = None,
        output_dim: int = FAVAR_CONFIG["output_dim"],
        p_lags: int = FAVAR_CONFIG["p_lags"],
        n_global_factors: int = FAVAR_CONFIG["n_global_factors"],
        n_experts: int = FAVAR_CONFIG["n_experts"],
        residual_hidden: int = FAVAR_CONFIG["residual_hidden"],
        residual_scale: float = FAVAR_CONFIG["residual_scale"],
        dropout: float = FAVAR_CONFIG["dropout"],
        learning_rate: float = FAVAR_CONFIG["learning_rate"],
        batch_size: int = FAVAR_CONFIG["batch_size"],
        epochs: int = FAVAR_CONFIG["epochs"],
        early_stopping_patience: int = FAVAR_CONFIG["early_stopping_patience"],
        weight_decay: float = FAVAR_CONFIG["weight_decay"],
        grad_clip: float = FAVAR_CONFIG["grad_clip"],
        random_state: int = FAVAR_CONFIG["random_state"],
    ):
        self.name = "FAVAR"
        self.task = task
        self.forecast_vars = forecast_vars or FORECAST_VARS
        self.n_vars = len(self.forecast_vars)
        self.output_dim = output_dim

        self.p_lags = p_lags
        self.n_global_factors = n_global_factors
        self.n_experts = n_experts
        self.residual_hidden = residual_hidden
        self.residual_scale = residual_scale
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.random_state = random_state

        self._model = None
        self._gate_model = None  # for extracting regime weights
        self._is_fitted = False

        # Normalization stats (input Z only for classify; Z+Y for forecast)
        self.z_mean_ = None
        self.z_std_ = None
        self.y_mean_ = None  # only used in forecast mode
        self.y_std_ = None   # only used in forecast mode

    # ------ normalization ------

    def _fit_normalization(self, Z: np.ndarray, Y: np.ndarray = None):
        """Fit normalization on training Z (inputs) and optionally Y (targets)."""
        self.z_mean_ = np.nanmean(Z, axis=0).astype(np.float32)
        self.z_std_ = np.nanstd(Z, axis=0).astype(np.float32)
        self.z_std_[self.z_std_ < 1e-8] = 1.0

        if self.task == "forecast" and Y is not None:
            self.y_mean_ = np.nanmean(Y, axis=0).astype(np.float32)
            self.y_std_ = np.nanstd(Y, axis=0).astype(np.float32)
            self.y_std_[self.y_std_ < 1e-8] = 1.0

    def _normalize_z(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.z_mean_) / (self.z_std_ + 1e-8)

    def _normalize_y(self, Y: np.ndarray) -> np.ndarray:
        """Only used in forecast mode."""
        return (Y - self.y_mean_) / (self.y_std_ + 1e-8)

    def _denormalize_y(self, Y: np.ndarray) -> np.ndarray:
        """Only used in forecast mode."""
        return Y * (self.y_std_ + 1e-8) + self.y_mean_

    # ------ training ------

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        f_train: np.ndarray,
        Z_val: np.ndarray = None,
        y_val: np.ndarray = None,
        f_val: np.ndarray = None,
        verbose: bool = True,
    ) -> "FAVARForecaster":
        """
        Train the FAVAR-Net.

        Args:
            Z_train: (n, input_dim) -- flattened VAR state vectors
            y_train: (n, output_dim) -- targets (crisis labels for classify,
                     macro forecasts for forecast)
            f_train: (n, n_factors) -- current global factors for gating
            Z_val, y_val, f_val: optional validation data
            verbose: print training info
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = Z_train.shape[1]
        loss_fn = "binary_crossentropy" if self.task == "classify" else "mse"

        if verbose:
            print(f"      FAVAR-Net ({self.task}): input_dim={input_dim}, "
                  f"n_factors={self.n_global_factors}, output_dim={self.output_dim}, "
                  f"n_experts={self.n_experts}, loss={loss_fn}")

        # Fit normalization
        self._fit_normalization(Z_train, y_train if self.task == "forecast" else None)
        Z_norm = self._normalize_z(Z_train).astype(np.float32)
        f_norm = f_train.astype(np.float32)

        # Prepare targets
        if self.task == "classify":
            # Labels are 0/1 -- no normalization needed
            y_prepared = y_train.astype(np.float32)
            if y_prepared.ndim == 1:
                y_prepared = y_prepared.reshape(-1, 1)
        else:
            y_prepared = self._normalize_y(y_train).astype(np.float32)

        # Replace NaNs
        Z_norm = np.nan_to_num(Z_norm, nan=0.0)
        y_prepared = np.nan_to_num(y_prepared, nan=0.0)
        f_norm = np.nan_to_num(f_norm, nan=0.0)

        # Build model
        self._model = build_favar_model(
            input_dim=input_dim,
            n_factors=self.n_global_factors,
            output_dim=self.output_dim,
            task=self.task,
            n_experts=self.n_experts,
            residual_hidden=self.residual_hidden,
            residual_scale=self.residual_scale,
            dropout=self.dropout,
        )

        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.grad_clip,
        )
        try:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.grad_clip,
                weight_decay=self.weight_decay,
            )
        except TypeError:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.grad_clip,
            )

        self._model.compile(optimizer=optimizer, loss=loss_fn)

        # Build gate extraction model
        self._gate_model = keras.Model(
            inputs=self._model.input,
            outputs=self._model.get_layer("regime_gate").output,
            name="gate_extractor",
        )

        # Callbacks
        callbacks = []
        validation_data = None
        if Z_val is not None and y_val is not None and f_val is not None:
            Z_val_norm = self._normalize_z(Z_val).astype(np.float32)
            f_val_norm = f_val.astype(np.float32)
            Z_val_norm = np.nan_to_num(Z_val_norm, nan=0.0)
            f_val_norm = np.nan_to_num(f_val_norm, nan=0.0)

            if self.task == "classify":
                y_val_prep = y_val.astype(np.float32)
                if y_val_prep.ndim == 1:
                    y_val_prep = y_val_prep.reshape(-1, 1)
            else:
                y_val_prep = self._normalize_y(y_val).astype(np.float32)
            y_val_prep = np.nan_to_num(y_val_prep, nan=0.0)

            validation_data = ([Z_val_norm, f_val_norm], y_val_prep)
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ))

        if verbose:
            print(f"      Training on {Z_train.shape[0]} samples "
                  f"(val={Z_val.shape[0] if Z_val is not None else 0})...")

        self._model.fit(
            [Z_norm, f_norm], y_prepared,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        self._is_fitted = True
        if verbose:
            n_params = self._model.count_params()
            print(f"      FAVAR-Net training complete ({n_params} params)")

        return self

    # ------ prediction ------

    def predict(self, Z: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Predict.

        For classify: returns (n, 1) crisis probabilities.
        For forecast: returns (n, forecast_steps, n_vars) denormalized.
        """
        if not self._is_fitted:
            raise RuntimeError("FAVARForecaster must be fitted before predicting.")

        Z_norm = self._normalize_z(Z).astype(np.float32)
        f_norm = f.astype(np.float32)
        Z_norm = np.nan_to_num(Z_norm, nan=0.0)
        f_norm = np.nan_to_num(f_norm, nan=0.0)

        pred = self._model.predict([Z_norm, f_norm], verbose=0)  # (n, output_dim)

        if self.task == "classify":
            # Clip probabilities to [0, 1] for safety
            return np.clip(pred, 0.0, 1.0)
        else:
            pred_denorm = self._denormalize_y(pred)
            n = pred_denorm.shape[0]
            return pred_denorm.reshape(n, -1, self.n_vars)

    def generate_neural_features(self, Z: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Generate neural_risk_score feature for the RF.

        Returns:
            (n,) array of crisis probabilities
        """
        preds = self.predict(Z, f)  # (n, 1)
        return preds.flatten()

    # ------ interpretability ------

    def extract_regime_weights(self, Z: np.ndarray, f: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract regime mixing weights for a batch of inputs.

        Returns:
            dict with:
                "pi": (n, n_experts) -- per-sample mixing weights
                "mean_pi": (n_experts,) -- average regime weights
                "f_values": (n, n_factors) -- the global factors used
        """
        if not self._is_fitted:
            raise RuntimeError("Must be fitted before extracting weights.")

        Z_norm = self._normalize_z(Z).astype(np.float32)
        f_norm = f.astype(np.float32)
        Z_norm = np.nan_to_num(Z_norm, nan=0.0)
        f_norm = np.nan_to_num(f_norm, nan=0.0)

        pi = self._gate_model.predict([Z_norm, f_norm], verbose=0)

        return {
            "pi": pi.astype(np.float32),
            "mean_pi": pi.mean(axis=0).astype(np.float32),
            "f_values": f.astype(np.float32),
        }

    def extract_expert_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Extract the linear VAR coefficient matrices from the expert heads.

        Returns:
            dict with W_k, b_k for each expert k.
        """
        if not self._is_fitted:
            raise RuntimeError("Must be fitted before extracting coefficients.")

        result = {}
        moe_layer = self._model.get_layer("moe")
        for k in range(self.n_experts):
            expert_layer = moe_layer.experts[k]
            weights = expert_layer.get_weights()
            result[f"W_{k}"] = weights[0].astype(np.float32)
            result[f"b_{k}"] = weights[1].astype(np.float32)

        return result

    # ------ persistence ------

    def save(self, path: str) -> None:
        """Save model and metadata."""
        metadata = {
            "name": self.name,
            "task": self.task,
            "forecast_vars": self.forecast_vars,
            "output_dim": self.output_dim,
            "p_lags": self.p_lags,
            "n_global_factors": self.n_global_factors,
            "n_experts": self.n_experts,
            "residual_hidden": self.residual_hidden,
            "residual_scale": self.residual_scale,
            "random_state": self.random_state,
            "z_mean_": self.z_mean_,
            "z_std_": self.z_std_,
            "y_mean_": self.y_mean_,
            "y_std_": self.y_std_,
        }
        joblib.dump(metadata, f"{path}_favar_metadata.pkl")
        if self._model is not None:
            self._model.save(f"{path}_favar.keras")

    def load(self, path: str) -> "FAVARForecaster":
        """Load model and metadata."""
        metadata = joblib.load(f"{path}_favar_metadata.pkl")
        self.name = metadata["name"]
        self.task = metadata.get("task", "forecast")
        self.forecast_vars = metadata["forecast_vars"]
        self.n_vars = len(self.forecast_vars)
        self.output_dim = metadata.get("output_dim", self.n_vars * 2)
        self.p_lags = metadata["p_lags"]
        self.n_global_factors = metadata["n_global_factors"]
        self.n_experts = metadata["n_experts"]
        self.residual_hidden = metadata["residual_hidden"]
        self.residual_scale = metadata["residual_scale"]
        self.random_state = metadata["random_state"]
        self.z_mean_ = metadata["z_mean_"]
        self.z_std_ = metadata["z_std_"]
        self.y_mean_ = metadata.get("y_mean_")
        self.y_std_ = metadata.get("y_std_")

        model_path = f"{path}_favar.keras"
        if os.path.exists(model_path):
            self._model = keras.models.load_model(
                model_path,
                custom_objects={
                    "ScaledResidualGRN": ScaledResidualGRN,
                    "MixtureOfExperts": MixtureOfExperts,
                },
            )
            self._gate_model = keras.Model(
                inputs=self._model.input,
                outputs=self._model.get_layer("regime_gate").output,
                name="gate_extractor",
            )
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
