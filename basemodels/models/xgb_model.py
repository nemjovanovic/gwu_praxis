# ================================================================
# XGBoost Model for Base Models
# ================================================================
#
# XGBoost classifier with early stopping.
#
# ================================================================

import numpy as np
import joblib
from typing import Optional

from .base_model import BaseModel
from ..config import XGB_CONFIG


class XGBoostModel(BaseModel):
    """
    XGBoost classifier with early stopping.
    """
    
    def __init__(
        self,
        name: str = "XGBoost",
        n_estimators: int = XGB_CONFIG["n_estimators"],
        learning_rate: float = XGB_CONFIG["learning_rate"],
        max_depth: int = XGB_CONFIG["max_depth"],
        min_child_weight: int = XGB_CONFIG["min_child_weight"],
        subsample: float = XGB_CONFIG["subsample"],
        colsample_bytree: float = XGB_CONFIG["colsample_bytree"],
        gamma: float = XGB_CONFIG["gamma"],
        reg_alpha: float = XGB_CONFIG["reg_alpha"],
        reg_lambda: float = XGB_CONFIG["reg_lambda"],
        early_stopping_rounds: int = XGB_CONFIG["early_stopping_rounds"],
        random_state: int = XGB_CONFIG["random_state"],
        n_jobs: int = XGB_CONFIG["n_jobs"],
    ):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            n_estimators: Maximum number of boosting rounds
            learning_rate: Step size shrinkage
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction required to make a split
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            early_stopping_rounds: Rounds for early stopping
            random_state: Random seed
            n_jobs: Number of parallel threads
        """
        super().__init__(name, random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        
        self.model = None
        self.best_iteration = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "XGBoostModel":
        """
        Fit XGBoost model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for early stopping
            y_val: Validation labels
        
        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # Compute scale_pos_weight for class imbalance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='auc',
            use_label_encoder=False,
        )
        
        # Fit with or without early stopping
        if X_val is not None and y_val is not None and len(np.unique(y_val)) >= 2:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            # best_iteration only exists if early stopping triggered
            try:
                self.best_iteration = self.model.best_iteration
            except AttributeError:
                self.best_iteration = self.n_estimators
        else:
            self.model.fit(X_train, y_train, verbose=False)
            self.best_iteration = self.n_estimators
        
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
        
        proba = self.model.predict_proba(X)
        
        # Handle single class case
        if proba.shape[1] == 1:
            return np.zeros(len(X))
        
        # Clip to avoid extreme values
        result = np.clip(proba[:, 1], 0.0001, 0.9999)
        
        return result.astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_path = f"{path}.pkl"
        joblib.dump({
            "model": self.model,
            "best_iteration": self.best_iteration,
            "config": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "gamma": self.gamma,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_state": self.random_state,
            }
        }, save_path)
    
    def load(self, path: str) -> "XGBoostModel":
        """Load model from disk."""
        load_path = f"{path}.pkl"
        data = joblib.load(load_path)
        
        self.model = data["model"]
        self.best_iteration = data["best_iteration"]
        self._is_fitted = True
        
        return self
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances from fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_
