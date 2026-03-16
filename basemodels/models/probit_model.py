# ================================================================
# Probit Model for Base Models
# ================================================================
#
# GLM with probit link via statsmodels.
#
# ================================================================

import numpy as np
import joblib
import warnings
from typing import Optional

from .base_model import BaseModel
from ..config import PROBIT_CONFIG


class ProbitModel(BaseModel):
    """
    Probit regression classifier.
    
    GLM with probit link function.
    Uses statsmodels for the probit implementation.
    """
    
    def __init__(
        self,
        name: str = "Probit",
        max_iter: int = PROBIT_CONFIG["max_iter"],
        random_state: int = PROBIT_CONFIG["random_state"],
    ):
        """
        Initialize Probit model.
        
        Args:
            name: Model name
            max_iter: Maximum iterations for optimization
            random_state: Random seed
        """
        super().__init__(name, random_state)
        
        self.max_iter = max_iter
        self.model = None
        self._use_sklearn = False  # Flag for fallback
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "ProbitModel":
        """
        Fit Probit model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (unused)
            y_val: Validation labels (unused)
        
        Returns:
            self
        """
        try:
            # Try statsmodels first (exact R match)
            import statsmodels.api as sm
            
            # Add constant term (intercept)
            X_with_const = sm.add_constant(X_train, has_constant='add')
            
            # Fit probit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = sm.Probit(y_train, X_with_const)
                self._result = self.model.fit(disp=False, maxiter=self.max_iter)
            
            self._use_sklearn = False
            
        except Exception as e:
            # Fallback to sklearn LogisticRegression with approximate probit
            # Note: sklearn doesn't have true probit, but logistic is similar
            warnings.warn(f"Statsmodels Probit failed ({e}), falling back to sklearn LogisticRegression")
            
            from sklearn.linear_model import LogisticRegression
            
            self.model = LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.model.fit(X_train, y_train)
            self._use_sklearn = True
        
        self._is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict crisis probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probability of crisis (positive class)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba()")
        
        if self._use_sklearn:
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 1:
                return np.zeros(len(X))
            proba = proba[:, 1].astype(np.float32)
        else:
            # Statsmodels probit
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X, has_constant='add')
            proba = self._result.predict(X_with_const)
            # Clip to avoid extreme probability values
            proba = np.clip(proba.astype(np.float32), 0.0001, 0.9999)
        # Replace any NaN (e.g. from non-convergence or numerical overflow) so evaluators don't crash
        if np.isnan(proba).any():
            proba = np.where(np.isnan(proba), 0.5, proba)
        return proba
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_path = f"{path}.pkl"
        
        if self._use_sklearn:
            joblib.dump({
                "model": self.model,
                "use_sklearn": True,
                "config": {
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                }
            }, save_path)
        else:
            # Save statsmodels result params
            joblib.dump({
                "params": self._result.params,
                "use_sklearn": False,
                "config": {
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                }
            }, save_path)
    
    def load(self, path: str) -> "ProbitModel":
        """Load model from disk."""
        load_path = f"{path}.pkl"
        data = joblib.load(load_path)
        
        self._use_sklearn = data["use_sklearn"]
        
        if self._use_sklearn:
            self.model = data["model"]
        else:
            # For statsmodels, we need to store params and recreate prediction
            self._saved_params = data["params"]
            # Note: Full model reload not supported for statsmodels
            # This is a simplified approach
        
        self._is_fitted = True
        return self
