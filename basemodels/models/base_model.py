# ================================================================
# Base Model Abstract Class for Base Models
# ================================================================

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseModel(ABC):
    """
    Abstract base class for all base models.
    
    All models must implement fit(), predict_proba(), save(), and load().
    """
    
    def __init__(self, name: str, random_state: int = 1234):
        """
        Initialize base model.
        
        Args:
            name: Model name for logging and file naming
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Fit the model on training data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels
            **kwargs: Additional arguments (e.g., countries for CV)
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
            **kwargs: Additional arguments (e.g., ch for Rule of Thumb)
        
        Returns:
            Probabilities of positive class (crisis), shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: File path (without extension, model will add appropriate extension)
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """
        Load the model from disk.
        
        Args:
            path: File path (without extension)
        
        Returns:
            self
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
