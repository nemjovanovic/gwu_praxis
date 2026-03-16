# ================================================================
# Rule of Thumb Model for Base Models
# ================================================================
#
# Simple baseline that uses historical crisis frequency (CH)
# to predict future crisis probability.
# Formula: pi_rot = 1 - ((1 - CH/100)^h)
#
# ================================================================

import numpy as np
import joblib
from typing import Optional

from .base_model import BaseModel


class RuleOfThumbModel(BaseModel):
    """
    Rule of Thumb baseline model.
    
    Uses historical crisis history (CH column) to predict future crisis:
        P(crisis in next h years) = 1 - (1 - CH/100)^h
    
    This is not a machine learning model - it simply applies a formula.
    """
    
    def __init__(
        self,
        name: str = "RuleOfThumb",
        horizon: int = 2,
        random_state: int = 1234,
    ):
        """
        Initialize Rule of Thumb model.
        
        Args:
            name: Model name
            horizon: Forecast horizon (h) in years
            random_state: Random seed (unused, for API compatibility)
        """
        super().__init__(name, random_state)
        self.horizon = horizon
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "RuleOfThumbModel":
        """
        Fit the model (no-op for Rule of Thumb).
        
        The Rule of Thumb model doesn't learn from data - it just uses
        the CH column directly. This method exists for API compatibility.
        
        Args:
            X_train: Training features (unused)
            y_train: Training labels (unused)
            X_val: Validation features (unused)
            y_val: Validation labels (unused)
        
        Returns:
            self
        """
        # No actual fitting - this is a formula-based baseline
        self._is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray, ch: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict crisis probabilities using Rule of Thumb formula.
        
        Formula: P = 1 - (1 - CH/100)^h
        
        Args:
            X: Features (unused, for API compatibility)
            ch: Crisis history values (CH column), required
        
        Returns:
            Probability of crisis, shape (n_samples,)
        """
        if ch is None:
            raise ValueError("Rule of Thumb model requires 'ch' (crisis history) array")
        
        # Apply formula: 1 - (1 - CH/100)^h
        # CH is the historical crisis rate (percentage)
        ch_rate = np.clip(ch / 100.0, 0.0, 1.0)  # Convert to proportion, clip to [0, 1]
        proba = 1.0 - np.power(1.0 - ch_rate, self.horizon)
        
        # Clip to avoid extreme probability values
        proba = np.clip(proba, 0.0001, 0.9999)
        
        return proba.astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = f"{path}.pkl"
        joblib.dump({
            "horizon": self.horizon,
            "name": self.name,
        }, save_path)
    
    def load(self, path: str) -> "RuleOfThumbModel":
        """Load model from disk."""
        load_path = f"{path}.pkl"
        data = joblib.load(load_path)
        
        self.horizon = data["horizon"]
        self._is_fitted = True
        
        return self
