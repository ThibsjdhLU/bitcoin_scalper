"""
Base model interface for all ML models in the Bitcoin Scalper system.

This module defines the abstract base class that all models must implement,
ensuring a consistent interface across classical ML and deep learning models.
This enables seamless model swapping in production pipelines.

Key Features:
- Standard training interface with sample_weights support
- Early stopping via eval_set
- Model persistence (save/load)
- Prediction methods (predict, predict_proba)
- Integration with Triple Barrier labeling

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 6: Ensemble Methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    This class defines the standard interface that all models (XGBoost, PyTorch,
    etc.) must implement. It ensures consistency across the system and allows
    models to be swapped seamlessly.
    
    Attributes:
        model: The underlying model object (XGBoost, PyTorch, etc.)
        is_fitted: Whether the model has been trained
        feature_names: Names of features the model was trained on
        n_features: Number of features
        classes_: Class labels for classification tasks
    """
    
    def __init__(self):
        """Initialize the base model."""
        self.model: Optional[Any] = None
        self.is_fitted: bool = False
        self.feature_names: Optional[list] = None
        self.n_features: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
    
    @abstractmethod
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], 
                                 Union[pd.Series, np.ndarray]]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features. Shape (n_samples, n_features).
            y: Training labels. Shape (n_samples,).
            sample_weights: Optional sample weights from Triple Barrier method.
                           Shape (n_samples,). Higher weights for samples that
                           exited early (hit profit/loss barriers quickly).
            eval_set: Optional validation set (X_val, y_val) for early stopping.
                     Used to prevent overfitting.
            **kwargs: Additional model-specific parameters.
        
        Returns:
            Self for method chaining.
            
        Example:
            >>> model = XGBoostClassifier()
            >>> model.train(
            ...     X_train, y_train,
            ...     sample_weights=barrier_weights,
            ...     eval_set=(X_val, y_val),
            ...     early_stopping_rounds=20
            ... )
            
        Notes:
            - sample_weights should be computed from Triple Barrier method
            - For event-based labeling, weight = 1 / holding_period
            - eval_set enables early stopping to prevent overfitting
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on. Shape (n_samples, n_features).
        
        Returns:
            Predictions. Shape (n_samples,).
            For classification: class labels
            For regression: continuous values
            
        Raises:
            ValueError: If model is not fitted.
            
        Example:
            >>> predictions = model.predict(X_test)
            >>> # For classification: [0, 1, -1, 1, ...]
            >>> # For regression: [0.023, -0.015, 0.008, ...]
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Features to predict on. Shape (n_samples, n_features).
        
        Returns:
            Class probabilities. Shape (n_samples, n_classes).
            Each row sums to 1.0.
            
        Raises:
            ValueError: If model is not fitted or is regression model.
            
        Example:
            >>> proba = model.predict_proba(X_test)
            >>> # For 3-class problem: [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], ...]
            >>> # Column 0: P(Short), Column 1: P(Neutral), Column 2: P(Long)
            
        Notes:
            - Used for meta-labeling (secondary model uses primary probabilities)
            - Higher probability = higher confidence
            - Can be used for position sizing based on confidence
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model. Should include extension.
                 For XGBoost: .json or .ubj
                 For PyTorch: .pt or .pth
                 For generic: .pkl
        
        Example:
            >>> model.save('models/xgboost_primary.json')
            >>> model.save('models/lstm_meta.pt')
            
        Notes:
            - Also saves metadata (feature_names, classes_, etc.)
            - Creates parent directories if they don't exist
            - Overwrites existing files
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from.
        
        Returns:
            Self for method chaining.
            
        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If file is corrupted or incompatible.
            
        Example:
            >>> model = XGBoostClassifier()
            >>> model.load('models/xgboost_primary.json')
            >>> predictions = model.predict(X_test)
            
        Notes:
            - Restores all model state including metadata
            - Validates model integrity after loading
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores (if available).
        
        Returns:
            Series with feature names as index and importance scores as values.
            Returns None if not available for this model type.
            
        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance.sort_values(ascending=False).head(10))
            ofi_1m           0.245
            spread_bps       0.183
            volume_imbalance 0.156
            ...
            
        Notes:
            - Available for tree-based models (XGBoost, CatBoost)
            - Not available for neural networks (use SHAP instead)
            - Useful for feature selection and debugging
        """
        return None
    
    def validate_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """
        Validate input data for NaN, infinite values, and shape consistency.
        
        Args:
            X: Features to validate.
            y: Optional labels to validate.
            
        Raises:
            ValueError: If data contains NaN/inf or shape mismatch.
            
        Notes:
            - Called automatically by train() and predict()
            - Checks for NaN and infinite values
            - Validates feature count matches training
            - Ensures no empty datasets
        """
        # Convert to numpy for uniform handling
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
        else:
            y_array = None
        
        # Check for empty data
        if X_array.shape[0] == 0:
            raise ValueError("Input data is empty (0 samples)")
        
        # Check for NaN values
        if np.isnan(X_array).any():
            n_nan = np.isnan(X_array).sum()
            raise ValueError(f"Input features contain {n_nan} NaN values. "
                           f"Please handle missing data before training/prediction.")
        
        # Check for infinite values
        if np.isinf(X_array).any():
            n_inf = np.isinf(X_array).sum()
            raise ValueError(f"Input features contain {n_inf} infinite values. "
                           f"Please handle outliers before training/prediction.")
        
        # Check labels if provided
        if y_array is not None:
            if len(y_array) != X_array.shape[0]:
                raise ValueError(
                    f"Feature and label count mismatch: "
                    f"X has {X_array.shape[0]} samples, y has {len(y_array)} samples"
                )
            
            if np.isnan(y_array).any():
                raise ValueError("Labels contain NaN values")
            
            if np.isinf(y_array).any():
                raise ValueError("Labels contain infinite values")
        
        # Check feature count consistency (only when fitted)
        if self.is_fitted and self.n_features is not None:
            if X_array.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature count mismatch: model was trained on {self.n_features} "
                    f"features, but got {X_array.shape[1]} features"
                )
        
        logger.debug(f"Input validation passed: {X_array.shape[0]} samples, "
                    f"{X_array.shape[1]} features")
    
    def _extract_feature_names(self, X: Union[pd.DataFrame, np.ndarray]) -> list:
        """
        Extract feature names from input data.
        
        Args:
            X: Input features.
            
        Returns:
            List of feature names.
        """
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        else:
            return [f"feature_{i}" for i in range(X.shape[1])]
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        features = f"{self.n_features} features" if self.n_features else "unknown features"
        return f"{self.__class__.__name__}({status}, {features})"
