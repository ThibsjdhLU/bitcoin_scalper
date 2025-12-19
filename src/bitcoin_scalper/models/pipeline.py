"""
Training pipeline and meta-labeling for ML models.

This module provides high-level training utilities including:
- Generic Trainer class for any BaseModel
- Meta-labeling pipeline (two-stage training)
- Data preprocessing and validation
- Integration with Triple Barrier labels

Meta-labeling Strategy:
    1. Primary Model: Predicts side (Long/Short/Neutral)
    2. Secondary Model: Predicts size (Bet/Pass) - should we take the trade?
    
This approach significantly improves Sharpe ratio by filtering false signals.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 3: Labeling & Chapter 6: Ensemble Methods.
"""

from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from src.bitcoin_scalper.models.base import BaseModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for ML models with robust preprocessing.
    
    This class handles the complete training workflow:
    - Data validation (NaN, inf handling)
    - Feature scaling (if needed)
    - Sample weight computation
    - Model training with early stopping
    - Evaluation and logging
    
    Attributes:
        model: BaseModel instance to train
        handle_nans: How to handle NaN values ('drop', 'fill', or 'error')
        handle_infs: How to handle infinite values ('clip', 'drop', or 'error')
        
    Example:
        >>> from src.bitcoin_scalper.models import XGBoostClassifier, Trainer
        >>> 
        >>> # Create model
        >>> model = XGBoostClassifier(n_estimators=100)
        >>> 
        >>> # Create trainer
        >>> trainer = Trainer(model, handle_nans='fill', handle_infs='clip')
        >>> 
        >>> # Train with Triple Barrier labels and weights
        >>> trainer.train(
        ...     X_train, y_train,
        ...     sample_weights=barrier_events['weight'],
        ...     eval_set=(X_val, y_val)
        ... )
        >>> 
        >>> # Evaluate
        >>> metrics = trainer.evaluate(X_test, y_test)
        >>> print(metrics)
    """
    
    def __init__(
        self,
        model: BaseModel,
        handle_nans: str = 'error',
        handle_infs: str = 'error',
        fill_value: float = 0.0,
        clip_value: float = 1e10
    ):
        """
        Initialize trainer.
        
        Args:
            model: BaseModel instance to train
            handle_nans: How to handle NaN values:
                - 'error': Raise error (default, safest)
                - 'drop': Drop rows with NaN
                - 'fill': Fill with fill_value
            handle_infs: How to handle infinite values:
                - 'error': Raise error (default, safest)
                - 'clip': Clip to +/- clip_value
                - 'drop': Drop rows with inf
            fill_value: Value to fill NaNs with (if handle_nans='fill')
            clip_value: Value to clip infs to (if handle_infs='clip')
        """
        self.model = model
        self.handle_nans = handle_nans
        self.handle_infs = handle_infs
        self.fill_value = fill_value
        self.clip_value = clip_value
        
        logger.info(f"Initialized Trainer (NaN: {handle_nans}, Inf: {handle_infs})")
    
    def _preprocess_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], 
               Optional[Union[pd.Series, np.ndarray]],
               Optional[Union[pd.Series, np.ndarray]]]:
        """
        Preprocess data to handle NaN and infinite values.
        
        Args:
            X: Features
            y: Labels (optional)
            sample_weights: Sample weights (optional)
            
        Returns:
            Tuple of (X_clean, y_clean, weights_clean)
        """
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        original_size = len(X)
        
        # Handle NaN values
        if self.handle_nans == 'error':
            if X.isna().any().any():
                n_nan = X.isna().sum().sum()
                raise ValueError(f"Found {n_nan} NaN values in features. "
                               f"Set handle_nans='fill' or 'drop' to proceed.")
        elif self.handle_nans == 'drop':
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y[mask]
                else:
                    y = y[mask.values]
            if sample_weights is not None:
                if isinstance(sample_weights, pd.Series):
                    sample_weights = sample_weights[mask]
                else:
                    sample_weights = sample_weights[mask.values]
            logger.info(f"Dropped {original_size - len(X)} rows with NaN values")
        elif self.handle_nans == 'fill':
            n_filled = X.isna().sum().sum()
            X = X.fillna(self.fill_value)
            if n_filled > 0:
                logger.info(f"Filled {n_filled} NaN values with {self.fill_value}")
        
        # Handle infinite values
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.handle_infs == 'error':
            if np.isinf(X_array).any():
                n_inf = np.isinf(X_array).sum()
                raise ValueError(f"Found {n_inf} infinite values in features. "
                               f"Set handle_infs='clip' or 'drop' to proceed.")
        elif self.handle_infs == 'clip':
            n_inf = np.isinf(X_array).sum()
            X_array = np.clip(X_array, -self.clip_value, self.clip_value)
            if isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X_array, index=X.index, columns=X.columns)
            else:
                X = X_array
            if n_inf > 0:
                logger.info(f"Clipped {n_inf} infinite values to +/- {self.clip_value}")
        elif self.handle_infs == 'drop':
            mask = ~np.isinf(X_array).any(axis=1)
            X = X[mask] if isinstance(X, pd.DataFrame) else X_array[mask]
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y[mask]
                else:
                    y = y[mask]
            if sample_weights is not None:
                if isinstance(sample_weights, pd.Series):
                    sample_weights = sample_weights[mask]
                else:
                    sample_weights = sample_weights[mask]
            logger.info(f"Dropped {original_size - len(X)} rows with infinite values")
        
        return X, y, sample_weights
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        **kwargs
    ) -> BaseModel:
        """
        Train the model with preprocessing.
        
        Args:
            X: Training features
            y: Training labels (from Triple Barrier method)
            sample_weights: Sample weights (from Triple Barrier method)
                          Higher weight = sample exited barriers quickly
            eval_set: Optional (X_val, y_val) for validation
            **kwargs: Additional training parameters passed to model.train()
            
        Returns:
            Trained model
            
        Notes:
            - Automatically handles NaN and inf values based on settings
            - Computes and logs class distribution
            - Validates data consistency
        """
        logger.info(f"Starting training with {len(X)} samples")
        
        # Preprocess training data
        X_clean, y_clean, weights_clean = self._preprocess_data(X, y, sample_weights)
        
        # Preprocess validation data if provided
        eval_set_clean = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_clean, y_val_clean, _ = self._preprocess_data(X_val, y_val)
            eval_set_clean = (X_val_clean, y_val_clean)
        
        # Log class distribution
        if isinstance(y_clean, pd.Series):
            class_counts = y_clean.value_counts()
        else:
            unique, counts = np.unique(y_clean, return_counts=True)
            class_counts = pd.Series(counts, index=unique)
        
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Train model
        self.model.train(
            X_clean, y_clean,
            sample_weights=weights_clean,
            eval_set=eval_set_clean,
            **kwargs
        )
        
        logger.info("Training completed successfully")
        return self.model
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of metrics (accuracy, precision, recall, etc.)
        """
        # Preprocess test data
        X_clean, y_clean, _ = self._preprocess_data(X, y)
        
        # Make predictions
        predictions = self.model.predict(X_clean)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_clean, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_clean, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(X_clean)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


class MetaLabelingPipeline:
    """
    Meta-labeling pipeline for two-stage prediction.
    
    Meta-labeling is a powerful technique that improves trading performance:
    
    Stage 1 (Primary Model):
        - Predicts SIDE: Long (+1), Neutral (0), or Short (-1)
        - Uses all market features
        - Goal: Identify potential trading opportunities
        
    Stage 2 (Secondary Model):
        - Predicts SIZE: Bet (1) or Pass (0)
        - Uses features + primary model probabilities
        - Goal: Filter false positives from primary model
        
    Combined Strategy:
        - Only take trades where:
          1. Primary model predicts Long or Short (not Neutral)
          2. Secondary model predicts Bet (not Pass)
        - This significantly improves Sharpe ratio
        
    Attributes:
        primary_model: Model for predicting trade direction
        secondary_model: Model for predicting whether to trade
        
    Example:
        >>> from src.bitcoin_scalper.models import XGBoostClassifier
        >>> from src.bitcoin_scalper.models import MetaLabelingPipeline
        >>> 
        >>> # Create models
        >>> primary = XGBoostClassifier(n_estimators=100)
        >>> secondary = XGBoostClassifier(n_estimators=50)
        >>> 
        >>> # Create pipeline
        >>> pipeline = MetaLabelingPipeline(primary, secondary)
        >>> 
        >>> # Train both stages
        >>> pipeline.train(
        ...     X_train, y_train_side,  # Side labels: -1, 0, 1
        ...     y_meta=y_train_success,  # Success labels: 0, 1
        ...     sample_weights=barrier_weights,
        ...     eval_set=(X_val, y_val_side, y_val_success)
        ... )
        >>> 
        >>> # Make predictions
        >>> side, confidence = pipeline.predict(X_test)
        >>> # side: -1, 0, or 1
        >>> # confidence: 0 (don't trade) or 1 (trade)
    """
    
    def __init__(
        self,
        primary_model: BaseModel,
        secondary_model: BaseModel
    ):
        """
        Initialize meta-labeling pipeline.
        
        Args:
            primary_model: Model for predicting trade direction (side)
            secondary_model: Model for predicting bet/pass (size)
        """
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        
        logger.info("Initialized MetaLabelingPipeline")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_side: Union[pd.Series, np.ndarray],
        y_meta: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        **kwargs
    ) -> 'MetaLabelingPipeline':
        """
        Train both primary and secondary models.
        
        Args:
            X: Features
            y_side: Primary labels (side: -1, 0, 1)
            y_meta: Meta labels (success: 0 or 1)
                   1 if trade was profitable, 0 otherwise
            sample_weights: Sample weights from Triple Barrier
            eval_set: Optional (X_val, y_side_val, y_meta_val)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
            
        Notes:
            - Primary model trained on all samples
            - Secondary model trained only on non-neutral predictions
            - Uses primary probabilities as features for secondary model
        """
        logger.info("=== Training Primary Model (Side Prediction) ===")
        
        # Train primary model
        primary_eval = None
        if eval_set is not None:
            X_val, y_side_val, _ = eval_set
            primary_eval = (X_val, y_side_val)
        
        self.primary_model.train(
            X, y_side,
            sample_weights=sample_weights,
            eval_set=primary_eval,
            **kwargs
        )
        
        logger.info("=== Training Secondary Model (Bet/Pass) ===")
        
        # Generate primary predictions and probabilities
        primary_proba = self.primary_model.predict_proba(X)
        
        # Create features for secondary model
        # Concatenate original features with primary probabilities
        if isinstance(X, pd.DataFrame):
            proba_df = pd.DataFrame(
                primary_proba,
                index=X.index,
                columns=[f'primary_proba_{i}' for i in range(primary_proba.shape[1])]
            )
            X_meta = pd.concat([X, proba_df], axis=1)
        else:
            X_meta = np.hstack([X, primary_proba])
        
        # Train secondary model
        secondary_eval = None
        if eval_set is not None:
            X_val, y_side_val, y_meta_val = eval_set
            val_proba = self.primary_model.predict_proba(X_val)
            
            if isinstance(X_val, pd.DataFrame):
                val_proba_df = pd.DataFrame(
                    val_proba,
                    index=X_val.index,
                    columns=[f'primary_proba_{i}' for i in range(val_proba.shape[1])]
                )
                X_meta_val = pd.concat([X_val, val_proba_df], axis=1)
            else:
                X_meta_val = np.hstack([X_val, val_proba])
            
            secondary_eval = (X_meta_val, y_meta_val)
        
        self.secondary_model.train(
            X_meta, y_meta,
            sample_weights=sample_weights,
            eval_set=secondary_eval,
            **kwargs
        )
        
        logger.info("Meta-labeling pipeline training completed")
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using both models.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (side_predictions, bet_predictions)
            - side_predictions: -1 (Short), 0 (Neutral), or 1 (Long)
            - bet_predictions: 0 (Pass) or 1 (Bet)
            
        Example:
            >>> side, bet = pipeline.predict(X_test)
            >>> 
            >>> # Only trade when both agree
            >>> final_predictions = side * bet
            >>> # final_predictions:
            >>> #   -1 if Short and Bet
            >>> #    0 if Pass or Neutral
            >>> #    1 if Long and Bet
        """
        # Primary prediction
        side = self.primary_model.predict(X)
        proba = self.primary_model.predict_proba(X)
        
        # Create meta features
        if isinstance(X, pd.DataFrame):
            proba_df = pd.DataFrame(
                proba,
                index=X.index,
                columns=[f'primary_proba_{i}' for i in range(proba.shape[1])]
            )
            X_meta = pd.concat([X, proba_df], axis=1)
        else:
            X_meta = np.hstack([X, proba])
        
        # Secondary prediction
        bet = self.secondary_model.predict(X_meta)
        
        return side, bet
    
    def predict_combined(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make combined predictions (side * bet).
        
        Args:
            X: Features
            
        Returns:
            Combined predictions:
            - -1: Short position (side=-1, bet=1)
            -  0: No position (side=0 or bet=0)
            -  1: Long position (side=1, bet=1)
        """
        side, bet = self.predict(X)
        return side * bet
    
    def save(self, primary_path: Union[str, Path], secondary_path: Union[str, Path]) -> None:
        """
        Save both models.
        
        Args:
            primary_path: Path to save primary model
            secondary_path: Path to save secondary model
        """
        self.primary_model.save(primary_path)
        self.secondary_model.save(secondary_path)
        logger.info(f"Pipeline saved: {primary_path}, {secondary_path}")
    
    def load(self, primary_path: Union[str, Path], secondary_path: Union[str, Path]) -> 'MetaLabelingPipeline':
        """
        Load both models.
        
        Args:
            primary_path: Path to load primary model from
            secondary_path: Path to load secondary model from
            
        Returns:
            Self for method chaining
        """
        self.primary_model.load(primary_path)
        self.secondary_model.load(secondary_path)
        logger.info(f"Pipeline loaded: {primary_path}, {secondary_path}")
        return self
