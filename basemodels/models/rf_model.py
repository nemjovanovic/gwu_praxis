# ================================================================
# Random Forest Model for Base Models
# ================================================================
#
# Random Forest with mtry hyperparameter tuning via 10-fold country-based CV.
#
# ================================================================

import numpy as np
import joblib
from typing import Optional, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .base_model import BaseModel
from ..config import RF_CONFIG


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with country-based cross-validation for mtry selection.
    
    Configuration:
    - n_estimators = 1000
    - mtry (max_features) tuning via K=10 fold country-based CV
    - class_weight='balanced' to handle class imbalance
    """
    
    def __init__(
        self,
        name: str = "RandomForest",
        n_estimators: int = RF_CONFIG["n_estimators"],
        mtry_grid: list = RF_CONFIG["mtry_grid"],
        cv_folds: int = RF_CONFIG["cv_folds"],
        class_weight: str = RF_CONFIG["class_weight"],
        random_state: int = RF_CONFIG["random_state"],
        n_jobs: int = RF_CONFIG["n_jobs"],
    ):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name
            n_estimators: Number of trees
            mtry_grid: List of max_features values to try
            cv_folds: Number of CV folds for mtry selection
            class_weight: Class weight strategy
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        super().__init__(name, random_state)
        
        self.n_estimators = n_estimators
        self.mtry_grid = mtry_grid
        self.cv_folds = cv_folds
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        
        self.best_mtry = None
        self.model = None
    
    def _create_country_folds(
        self,
        countries: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create country-based cross-validation folds.
        
        Splits by country to prevent data leakage.
        
        Args:
            countries: Array of country codes
        
        Returns:
            List of (train_idx, val_idx) tuples
        """
        np.random.seed(self.random_state)
        
        unique_countries = np.unique(countries)
        np.random.shuffle(unique_countries)
        
        # Split countries into folds
        country_folds = np.array_split(unique_countries, self.cv_folds)
        
        folds = []
        for i in range(self.cv_folds):
            val_countries = set(country_folds[i])
            val_mask = np.array([c in val_countries for c in countries])
            
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            folds.append((train_idx, val_idx))
        
        return folds
    
    def _select_mtry(
        self,
        X: np.ndarray,
        y: np.ndarray,
        countries: np.ndarray
    ) -> int:
        """
        Select best max_features via country-based cross-validation.
        
        Uses AUC as the selection criterion.
        
        Args:
            X: Training features
            y: Training labels
            countries: Country codes for fold splitting
        
        Returns:
            Best max_features value
        """
        n_features = X.shape[1]
        
        # Filter mtry values that are valid for this dataset
        valid_mtry = [m for m in self.mtry_grid if m <= n_features]
        if not valid_mtry:
            valid_mtry = [min(n_features, max(1, int(np.sqrt(n_features))))]
        
        # Create country-based folds
        folds = self._create_country_folds(countries)
        
        best_score = -np.inf
        best_mtry = valid_mtry[0]
        
        for mtry in valid_mtry:
            fold_scores = []
            
            for train_idx, val_idx in folds:
                if len(val_idx) == 0 or len(train_idx) == 0:
                    continue
                
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Skip if only one class in validation
                if len(np.unique(y_val)) < 2:
                    continue
                
                rf = RandomForestClassifier(
                    n_estimators=min(100, self.n_estimators),  # Fewer trees for CV speed
                    max_features=mtry,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                
                rf.fit(X_tr, y_tr)
                proba = rf.predict_proba(X_val)
                
                if proba.shape[1] > 1:
                    auc = roc_auc_score(y_val, proba[:, 1])
                    fold_scores.append(auc)
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_mtry = mtry
        
        return best_mtry
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        countries: Optional[np.ndarray] = None,
        **kwargs
    ) -> "RandomForestModel":
        """
        Fit Random Forest with mtry selection via country-based CV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (unused for RF)
            y_val: Validation labels (unused for RF)
            countries: Country codes for CV fold splitting
        
        Returns:
            self
        """
        # Select best mtry via CV (only if we have country info)
        if countries is not None and len(np.unique(y_train)) >= 2:
            self.best_mtry = self._select_mtry(X_train, y_train, countries)
        else:
            # Default: sqrt of features
            self.best_mtry = min(X_train.shape[1], max(1, int(np.sqrt(X_train.shape[1]))))
        
        # Train final model with all trees
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.best_mtry,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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
        
        # Handle case where only one class was seen
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
            "best_mtry": self.best_mtry,
            "config": {
                "n_estimators": self.n_estimators,
                "mtry_grid": self.mtry_grid,
                "cv_folds": self.cv_folds,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            }
        }, save_path)
    
    def load(self, path: str) -> "RandomForestModel":
        """Load model from disk."""
        load_path = f"{path}.pkl"
        data = joblib.load(load_path)
        
        self.model = data["model"]
        self.best_mtry = data["best_mtry"]
        self._is_fitted = True
        
        return self
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances from fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_
