"""
Gradient Boosting models (XGBoost and CatBoost) for Bitcoin trading.

This module provides production-ready wrappers around XGBoost and CatBoost
that implement the BaseModel interface. Key features:
- Automatic GPU acceleration detection
- Early stopping support
- Feature importance extraction
- Hyperparameter tuning ready (Optuna integration)
- Sample weights from Triple Barrier method

Performance Notes:
- XGBoost: Excellent for tabular data, GPU acceleration available
- CatBoost: Handles categorical features natively, robust to overfitting
- Both support early stopping to prevent overfitting

References:
    Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

from src.bitcoin_scalper.models.base import BaseModel

logger = logging.getLogger(__name__)

# Optional imports
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    logger.warning("CatBoost not available. Install with: pip install catboost")


class XGBoostClassifier(BaseModel):
    """
    XGBoost classifier wrapper implementing BaseModel interface.
    
    This wrapper provides a production-ready interface to XGBoost with:
    - Automatic GPU detection and usage
    - Early stopping support
    - Sample weights from Triple Barrier
    - Feature importance extraction
    - Hyperparameter tuning ready
    
    Attributes:
        model: XGBoost Booster object
        params: Hyperparameters for XGBoost
        use_gpu: Whether GPU acceleration is enabled
        
    Example:
        >>> # Basic usage
        >>> model = XGBoostClassifier(
        ...     n_estimators=100,
        ...     max_depth=6,
        ...     learning_rate=0.1
        ... )
        >>> model.train(
        ...     X_train, y_train,
        ...     sample_weights=barrier_weights,
        ...     eval_set=(X_val, y_val),
        ...     early_stopping_rounds=20
        ... )
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
        
        >>> # Feature importance
        >>> importance = model.get_feature_importance()
        >>> print(importance.sort_values(ascending=False).head(10))
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        use_gpu: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds (trees).
            max_depth: Maximum tree depth. Higher = more complex, risk overfitting.
            learning_rate: Step size shrinkage. Lower = more conservative.
            subsample: Fraction of samples for each tree. Prevents overfitting.
            colsample_bytree: Fraction of features for each tree. Prevents overfitting.
            gamma: Minimum loss reduction for split. Regularization parameter.
            min_child_weight: Minimum sum of instance weight in child. Regularization.
            reg_alpha: L1 regularization (Lasso). Promotes sparsity.
            reg_lambda: L2 regularization (Ridge). Prevents large weights.
            use_gpu: Whether to use GPU acceleration if available.
            random_state: Random seed for reproducibility.
            **kwargs: Additional XGBoost parameters.
            
        Notes:
            - GPU acceleration requires CUDA-enabled GPU and xgboost compiled with GPU support
            - Start with defaults, then tune with Optuna
            - For imbalanced data, consider scale_pos_weight parameter
        """
        super().__init__()
        
        if not _HAS_XGBOOST:
            raise ImportError("XGBoost is required but not installed")
        
        # Check GPU availability
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        # Store hyperparameters
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            **kwargs
        }
        
        logger.info(f"Initialized XGBoostClassifier (GPU: {self.use_gpu})")
    
    def _check_gpu_available(self) -> bool:
        """
        Check if GPU is available for XGBoost.
        
        Uses nvidia-smi to check for GPU availability, which is much faster
        than attempting to train a model. The 2-second timeout ensures we
        don't block if nvidia-smi is slow or unavailable.
        
        Returns:
            True if GPU is available, False otherwise.
        """
        try:
            # Check for CUDA availability via nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv'],
                capture_output=True, 
                timeout=2  # 2 seconds is enough for nvidia-smi to respond
            )
            if result.returncode == 0:
                logger.info("GPU acceleration available and enabled")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        logger.info("GPU not available, using CPU")
        return False
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = True,
        **kwargs
    ) -> 'XGBoostClassifier':
        """
        Train XGBoost classifier.
        
        Args:
            X: Training features.
            y: Training labels.
            sample_weights: Optional weights from Triple Barrier method.
            eval_set: Optional (X_val, y_val) for early stopping.
            early_stopping_rounds: Stop if no improvement for N rounds.
            verbose: Whether to print training progress.
            **kwargs: Additional training parameters.
            
        Returns:
            Self for method chaining.
        """
        # Validate inputs
        self.validate_inputs(X, y)
        
        # Extract feature names
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Store classes
        self.classes_ = np.unique(y_array)
        n_classes = len(self.classes_)
        
        # Map labels to 0, 1, 2, ... for XGBoost
        label_map = {label: i for i, label in enumerate(self.classes_)}
        y_mapped = np.array([label_map[label] for label in y_array])
        
        # Create DMatrix
        if sample_weights is not None:
            if isinstance(sample_weights, pd.Series):
                sample_weights = sample_weights.values
            dtrain = xgb.DMatrix(X_array, label=y_mapped, weight=sample_weights,
                                feature_names=self.feature_names)
        else:
            dtrain = xgb.DMatrix(X_array, label=y_mapped,
                                feature_names=self.feature_names)
        
        # Setup parameters
        train_params = self.params.copy()
        if n_classes > 2:
            train_params['objective'] = 'multi:softprob'
            train_params['num_class'] = n_classes
        else:
            train_params['objective'] = 'binary:logistic'
        
        train_params['eval_metric'] = 'logloss'
        
        # Prepare evaluation set
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            X_val, y_val = eval_set
            self.validate_inputs(X_val, y_val)
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Map validation labels
            y_val_mapped = np.array([label_map[label] for label in y_val])
            dval = xgb.DMatrix(X_val, label=y_val_mapped,
                              feature_names=self.feature_names)
            evals.append((dval, 'val'))
        
        # Train model
        logger.info(f"Training XGBoost with {X_array.shape[0]} samples, "
                   f"{self.n_features} features, {n_classes} classes")
        
        evals_result = {}
        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=train_params['n_estimators'],
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=10 if verbose else False
        )
        
        self.is_fitted = True
        
        # Log final performance
        if eval_set is not None and 'val' in evals_result:
            final_loss = evals_result['val']['logloss'][-1]
            logger.info(f"Training completed. Final validation loss: {final_loss:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        
        # Get probabilities and convert to labels
        proba = self.predict_proba(X)
        predicted_indices = np.argmax(proba, axis=1)
        return self.classes_[predicted_indices]
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Class probabilities. Shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_array, feature_names=self.feature_names)
        
        # Predict
        proba = self.model.predict(dtest)
        
        # Handle binary vs multiclass
        if len(self.classes_) == 2 and proba.ndim == 1:
            # Binary classification - convert to 2D array
            proba = np.vstack([1 - proba, proba]).T
        
        return proba
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'use_gpu': self.use_gpu
        }
        
        metadata_path = path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> 'XGBoostClassifier':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from.
            
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        
        # Load metadata
        metadata_path = path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.params = metadata.get('params', {})
            self.feature_names = metadata.get('feature_names')
            self.n_features = metadata.get('n_features')
            classes = metadata.get('classes')
            self.classes_ = np.array(classes) if classes else None
            self.use_gpu = metadata.get('use_gpu', False)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return self
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance metric.
                - 'weight': Number of times a feature is used in splits
                - 'gain': Average gain of splits using the feature
                - 'cover': Average coverage of splits using the feature
                
        Returns:
            Series with feature names and importance scores.
            
        Example:
            >>> importance = model.get_feature_importance('gain')
            >>> print(importance.sort_values(ascending=False).head(10))
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting importance")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        # Create series with all features (missing features get 0)
        importance = pd.Series(0.0, index=self.feature_names)
        for feature, score in importance_dict.items():
            if feature in importance.index:
                importance[feature] = score
        
        return importance


class XGBoostRegressor(XGBoostClassifier):
    """
    XGBoost regressor wrapper implementing BaseModel interface.
    
    Similar to XGBoostClassifier but for regression tasks (predicting returns).
    
    Example:
        >>> model = XGBoostRegressor()
        >>> model.train(X_train, returns_train, sample_weights=weights)
        >>> predicted_returns = model.predict(X_test)
    """
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = True,
        **kwargs
    ) -> 'XGBoostRegressor':
        """Train XGBoost regressor."""
        # Validate inputs
        self.validate_inputs(X, y)
        
        # Extract feature names
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Create DMatrix
        if sample_weights is not None:
            if isinstance(sample_weights, pd.Series):
                sample_weights = sample_weights.values
            dtrain = xgb.DMatrix(X_array, label=y_array, weight=sample_weights,
                                feature_names=self.feature_names)
        else:
            dtrain = xgb.DMatrix(X_array, label=y_array,
                                feature_names=self.feature_names)
        
        # Setup parameters for regression
        train_params = self.params.copy()
        train_params['objective'] = 'reg:squarederror'
        train_params['eval_metric'] = 'rmse'
        
        # Prepare evaluation set
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            X_val, y_val = eval_set
            self.validate_inputs(X_val, y_val)
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            dval = xgb.DMatrix(X_val, label=y_val,
                              feature_names=self.feature_names)
            evals.append((dval, 'val'))
        
        # Train model
        logger.info(f"Training XGBoost Regressor with {X_array.shape[0]} samples, "
                   f"{self.n_features} features")
        
        evals_result = {}
        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=train_params['n_estimators'],
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=10 if verbose else False
        )
        
        self.is_fitted = True
        
        if eval_set is not None and 'val' in evals_result:
            final_rmse = evals_result['val']['rmse'][-1]
            logger.info(f"Training completed. Final validation RMSE: {final_rmse:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict continuous values."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        dtest = xgb.DMatrix(X_array, feature_names=self.feature_names)
        return self.model.predict(dtest)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("predict_proba not available for regression models")


class CatBoostClassifierWrapper(BaseModel):
    """
    CatBoost classifier wrapper implementing BaseModel interface.
    
    CatBoost is particularly good at:
    - Handling categorical features natively
    - Robust to overfitting (ordered boosting)
    - Fast inference
    - Handling missing values
    
    Example:
        >>> model = CatBoostClassifierWrapper()
        >>> model.train(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize CatBoost classifier."""
        super().__init__()
        
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost is required but not installed")
        
        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'random_state': random_state,
            'verbose': False,
            **kwargs
        }
        
        self.model = CatBoostClassifier(**self.params)
        logger.info("Initialized CatBoostClassifier")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = 20,
        **kwargs
    ) -> 'CatBoostClassifierWrapper':
        """Train CatBoost classifier."""
        self.validate_inputs(X, y)
        
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        
        # Prepare eval set
        eval_set_cb = None
        if eval_set is not None:
            X_val, y_val = eval_set
            self.validate_inputs(X_val, y_val)
            eval_set_cb = (X_val, y_val)
        
        # Train
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=eval_set_cb,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        self.classes_ = self.model.classes_
        self.is_fitted = True
        
        logger.info(f"CatBoost training completed")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        return self.model.predict_proba(X)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path))
        logger.info(f"CatBoost model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> 'CatBoostClassifierWrapper':
        """Load model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        self.is_fitted = True
        
        logger.info(f"CatBoost model loaded from {path}")
        return self
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting importance")
        
        importance = self.model.get_feature_importance()
        return pd.Series(importance, index=self.feature_names)


class CatBoostRegressorWrapper(BaseModel):
    """CatBoost regressor wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize CatBoost regressor."""
        super().__init__()
        
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost is required but not installed")
        
        self.params = kwargs
        self.model = CatBoostRegressor(**kwargs, verbose=False)
        logger.info("Initialized CatBoostRegressor")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = 20,
        **kwargs
    ) -> 'CatBoostRegressorWrapper':
        """Train CatBoost regressor."""
        self.validate_inputs(X, y)
        
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        
        # Prepare eval set
        eval_set_cb = None
        if eval_set is not None:
            X_val, y_val = eval_set
            self.validate_inputs(X_val, y_val)
            eval_set_cb = (X_val, y_val)
        
        # Train
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=eval_set_cb,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        self.is_fitted = True
        logger.info("CatBoost regressor training completed")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict continuous values."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        return self.model.predict(X)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path))
        logger.info(f"CatBoost regressor saved to {path}")
    
    def load(self, path: Union[str, Path]) -> 'CatBoostRegressorWrapper':
        """Load model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = CatBoostRegressor()
        self.model.load_model(str(path))
        self.is_fitted = True
        
        logger.info(f"CatBoost regressor loaded from {path}")
        return self
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting importance")
        
        importance = self.model.get_feature_importance()
        return pd.Series(importance, index=self.feature_names)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("predict_proba not available for regression models")
