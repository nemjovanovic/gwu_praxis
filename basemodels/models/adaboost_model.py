# ================================================================
# AdaBoost Model for Base Models
# ================================================================
#
# Gradient Boosting with exponential loss (AdaBoost).
# Uses sklearn GradientBoostingClassifier with n_estimators tuning
# via country-based CV.
#
# ================================================================

import numpy as np
import joblib
from typing import Optional, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .base_model import BaseModel
from ..config import ADABOOST_CONFIG


class AdaBoostModel(BaseModel):
    """
    AdaBoost/Gradient Boosting classifier with country-based CV for n_estimators selection.
    
    Gradient boosting with exponential loss (AdaBoost):
    - shrinkage = 0.01
    - max_depth = 2
    - n_estimators tuning via K=10 fold country-based CV
    """
    
    def __init__(
        self,
        name: str = "AdaBoost",
        learning_rate: float = ADABOOST_CONFIG["learning_rate"],
        max_depth: int = ADABOOST_CONFIG["max_depth"],
        n_estimators_grid: list = ADABOOST_CONFIG["n_estimators_grid"],
        cv_folds: int = ADABOOST_CONFIG["cv_folds"],
        random_state: int = ADABOOST_CONFIG["random_state"],
    ):
        """
        Initialize AdaBoost model.
        
        Args:
            name: Model name
            learning_rate: Shrinkage rate
            max_depth: Maximum tree depth
            n_estimators_grid: List of n_trees values to try
            cv_folds: Number of CV folds
            random_state: Random seed
        """
        super().__init__(name, random_state)
        
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators_grid = n_estimators_grid
        self.cv_folds = cv_folds
        
        self.best_n_estimators = None
        self.model = None
    
    def _create_country_folds(
        self,
        countries: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create country-based cross-validation folds.
        
        Args:
            countries: Array of country codes
        
        Returns:
            List of (train_idx, val_idx) tuples
        """
        np.random.seed(self.random_state)
        
        unique_countries = np.unique(countries)
        np.random.shuffle(unique_countries)
        
        country_folds = np.array_split(unique_countries, self.cv_folds)
        
        folds = []
        for i in range(self.cv_folds):
            val_countries = set(country_folds[i])
            val_mask = np.array([c in val_countries for c in countries])
            
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            folds.append((train_idx, val_idx))
        
        return folds
    
    def _select_n_estimators(
        self,
        X: np.ndarray,
        y: np.ndarray,
        countries: np.ndarray
    ) -> int:
        """
        Select best n_estimators via country-based cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            countries: Country codes for fold splitting
        
        Returns:
            Best n_estimators value
        """
        # Create country-based folds
        folds = self._create_country_folds(countries)
        
        best_score = -np.inf
        best_n = self.n_estimators_grid[0]
        
        for n_est in self.n_estimators_grid:
            fold_scores = []
            
            for train_idx, val_idx in folds:
                if len(val_idx) == 0 or len(train_idx) == 0:
                    continue
                
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Skip if only one class in validation
                if len(np.unique(y_val)) < 2:
                    continue
                
                # Use exponential loss for AdaBoost-like behavior
                gb = GradientBoostingClassifier(
                    n_estimators=n_est,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    loss='exponential',  # AdaBoost loss
                    random_state=self.random_state,
                )
                
                gb.fit(X_tr, y_tr)
                proba = gb.predict_proba(X_val)
                
                if proba.shape[1] > 1:
                    auc = roc_auc_score(y_val, proba[:, 1])
                    fold_scores.append(auc)
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_n = n_est
        
        return best_n
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        countries: Optional[np.ndarray] = None,
        **kwargs
    ) -> "AdaBoostModel":
        """
        Fit AdaBoost with n_estimators selection via country-based CV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (unused)
            y_val: Validation labels (unused)
            countries: Country codes for CV fold splitting
        
        Returns:
            self
        """
        # Select best n_estimators via CV
        if countries is not None and len(np.unique(y_train)) >= 2:
            self.best_n_estimators = self._select_n_estimators(X_train, y_train, countries)
        else:
            # Default: median of grid
            self.best_n_estimators = self.n_estimators_grid[len(self.n_estimators_grid) // 2]
        
        # Train final model
        self.model = GradientBoostingClassifier(
            n_estimators=self.best_n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            loss='exponential',  # AdaBoost loss
            random_state=self.random_state,
        )
        
        self.model.fit(X_train, y_train)
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
        
        # Clip to avoid extreme probability values
        result = np.clip(proba[:, 1], 0.0001, 0.9999)
        
        return result.astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_path = f"{path}.pkl"
        joblib.dump({
            "model": self.model,
            "best_n_estimators": self.best_n_estimators,
            "config": {
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "n_estimators_grid": self.n_estimators_grid,
                "cv_folds": self.cv_folds,
                "random_state": self.random_state,
            }
        }, save_path)
    
    def load(self, path: str) -> "AdaBoostModel":
        """Load model from disk."""
        load_path = f"{path}.pkl"
        data = joblib.load(load_path)
        
        self.model = data["model"]
        self.best_n_estimators = data["best_n_estimators"]
        self._is_fitted = True
        
        return self
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances from fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_
