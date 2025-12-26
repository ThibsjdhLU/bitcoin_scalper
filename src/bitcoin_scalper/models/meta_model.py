"""
Modern Meta-Labeling Model for Bitcoin Trading.

This module provides a production-ready meta-labeling implementation adapted from
legacy MetaLabelingPipeline, optimized for integration with the TradingEngine
and CatBoost models.

Meta-labeling Strategy (López de Prado, 2018):
    Stage 1 (Primary Model): Predicts DIRECTION (Buy=1, Sell=-1, Neutral=0)
    Stage 2 (Meta Model): Predicts SUCCESS probability (1=take trade, 0=pass)
    
The meta model filters false signals from the primary model, significantly
improving Sharpe ratio and reducing false positives.

Key Improvements over Legacy:
    - Modern type hints and error handling
    - CatBoost-specific optimizations
    - Integration with TradingEngine structure
    - Enhanced logging and monitoring
    - Threshold-based signal filtering
    
References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 3: Meta-Labeling, Chapter 6: Ensemble Methods.
"""

from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Import central label utils
from bitcoin_scalper.core.label_utils import decode_primary, PRIMARY_LABEL_MAPPING

logger = logging.getLogger(__name__)

# Optional CatBoost import
try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    logger.warning("CatBoost not available. Install with: pip install catboost")


class MetaModel:
    """
    Modern meta-labeling model with two-stage prediction architecture.
    
    This class implements the meta-labeling strategy where:
    1. Primary model predicts trade direction (Buy/Sell/Neutral)
    2. Meta model predicts success probability (should we take this trade?)
    
    The meta model is trained on primary predictions + original features,
    learning to filter out false positives from the primary model.
    
    Attributes:
        primary_model: Model for predicting trade direction
        meta_model: Model for predicting trade success probability
        meta_threshold: Confidence threshold for meta model (default: 0.5)
        
    Example:
        >>> from catboost import CatBoostClassifier
        >>> 
        >>> # Create models
        >>> primary = CatBoostClassifier(iterations=100, depth=6, verbose=False)
        >>> meta = CatBoostClassifier(iterations=50, depth=4, verbose=False)
        >>> 
        >>> # Create meta model
        >>> meta_model = MetaModel(primary, meta, meta_threshold=0.6)
        >>> 
        >>> # Train both stages
        >>> meta_model.train(
        ...     X_train, 
        ...     y_direction,  # Direction labels: -1, 0, 1
        ...     y_success,    # Success labels: 0, 1
        ...     sample_weights=weights,
        ...     eval_set=(X_val, y_direction_val, y_success_val)
        ... )
        >>> 
        >>> # Make predictions with filtering
        >>> result = meta_model.predict_meta(X_test)
        >>> print(result['final_signal'])   # Filtered signal
        >>> print(result['meta_conf'])      # Meta confidence
        >>> print(result['raw_signal'])     # Original signal
    """
    
    def __init__(
        self,
        primary_model: Any,
        meta_model: Any,
        meta_threshold: float = 0.5
    ):
        """
        Initialize meta-labeling model.
        
        Args:
            primary_model: Model for predicting direction (Buy/Sell/Neutral)
                         Should have .train(), .predict(), .predict_proba() methods
                         Expected to output: -1 (Sell), 0 (Neutral), 1 (Buy)
            meta_model: Model for predicting success probability (0/1)
                       Should have .train(), .predict(), .predict_proba() methods
                       Expected to output: 0 (Pass), 1 (Take trade)
            meta_threshold: Confidence threshold for meta model
                          Only take trades where meta_conf >= threshold
                          Higher = more conservative (fewer but better trades)
        """
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.meta_threshold = meta_threshold
        
        # Track training state
        self.is_trained = False
        self.feature_names = None
        self.n_features = None
        
        logger.info(
            f"Initialized MetaModel with threshold={meta_threshold:.2f}"
        )
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_direction: Union[pd.Series, np.ndarray],
        y_success: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.Series, np.ndarray],
            Union[pd.Series, np.ndarray]
        ]] = None,
        **kwargs
    ) -> 'MetaModel':
        """
        Train both primary and meta models.
        
        Training Strategy:
        1. Train primary model on (X, y_direction) to predict Buy/Sell/Neutral
        2. Generate primary predictions and probabilities
        3. Create augmented features: X + primary_probabilities
        4. Train meta model on (X_augmented, y_success) to predict success
        
        Args:
            X: Original features (market data, technical indicators, etc.)
            y_direction: Direction labels for primary model
                        -1 = Sell, 0 = Neutral, 1 = Buy
            y_success: Success labels for meta model
                      0 = Trade failed/loss, 1 = Trade succeeded/profit
            sample_weights: Optional sample weights (from Triple Barrier)
                          Higher weight for samples that exited quickly
            eval_set: Optional validation set (X_val, y_direction_val, y_success_val)
                     Used for early stopping
            **kwargs: Additional training parameters passed to both models
                     e.g., early_stopping_rounds, verbose, etc.
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If models don't have required methods or data is invalid
            
        Notes:
            - y_direction and y_success should be aligned (same indices)
            - Meta model learns which primary predictions to trust
            - Use sample_weights from Triple Barrier for better results
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: Training Primary Model (Direction Prediction)")
        logger.info("=" * 60)
        
        # Store feature information
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.n_features = len(self.feature_names)
        else:
            self.n_features = X.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
        logger.info(f"Training on {len(X)} samples, {self.n_features} features")
        
        # Log class distribution for direction
        if isinstance(y_direction, pd.Series):
            direction_dist = y_direction.value_counts().to_dict()
        else:
            unique, counts = np.unique(y_direction, return_counts=True)
            direction_dist = dict(zip(unique, counts))
        logger.info(f"Direction distribution: {direction_dist}")
        
        # Train primary model
        try:
            primary_eval = None
            if eval_set is not None:
                X_val, y_direction_val, _ = eval_set
                primary_eval = (X_val, y_direction_val)
            
            # Check if using CatBoost with its native API
            if _HAS_CATBOOST and isinstance(self.primary_model, CatBoostClassifier):
                # CatBoost native training
                if sample_weights is not None:
                    self.primary_model.fit(
                        X, y_direction,
                        sample_weight=sample_weights,
                        eval_set=primary_eval if primary_eval else None,
                        **kwargs
                    )
                else:
                    self.primary_model.fit(
                        X, y_direction,
                        eval_set=primary_eval if primary_eval else None,
                        **kwargs
                    )
            elif hasattr(self.primary_model, 'train'):
                # BaseModel interface
                self.primary_model.train(
                    X, y_direction,
                    sample_weights=sample_weights,
                    eval_set=primary_eval,
                    **kwargs
                )
            elif hasattr(self.primary_model, 'fit'):
                # Scikit-learn interface
                if sample_weights is not None:
                    self.primary_model.fit(X, y_direction, sample_weight=sample_weights)
                else:
                    self.primary_model.fit(X, y_direction)
            else:
                raise ValueError("Primary model must have 'train' or 'fit' method")
            
            logger.info("Primary model training completed")
            
        except Exception as e:
            logger.error(f"Primary model training failed: {e}")
            raise
        
        logger.info("=" * 60)
        logger.info("STAGE 2: Training Meta Model (Success Prediction)")
        logger.info("=" * 60)
        
        # Generate primary predictions and probabilities for meta features
        try:
            primary_proba = self.primary_model.predict_proba(X)
            logger.info(f"Generated primary probabilities: shape={primary_proba.shape}")
            
            # Create augmented features for meta model
            X_meta = self._create_meta_features(X, primary_proba)
            
            # Log class distribution for success
            if isinstance(y_success, pd.Series):
                success_dist = y_success.value_counts().to_dict()
            else:
                unique, counts = np.unique(y_success, return_counts=True)
                success_dist = dict(zip(unique, counts))
            logger.info(f"Success distribution: {success_dist}")
            
            # Prepare validation set for meta model
            meta_eval = None
            if eval_set is not None:
                X_val, y_direction_val, y_success_val = eval_set
                val_proba = self.primary_model.predict_proba(X_val)
                X_meta_val = self._create_meta_features(X_val, val_proba)
                meta_eval = (X_meta_val, y_success_val)
            
            # Train meta model
            if _HAS_CATBOOST and isinstance(self.meta_model, CatBoostClassifier):
                # CatBoost native training
                if sample_weights is not None:
                    self.meta_model.fit(
                        X_meta, y_success,
                        sample_weight=sample_weights,
                        eval_set=meta_eval if meta_eval else None,
                        **kwargs
                    )
                else:
                    self.meta_model.fit(
                        X_meta, y_success,
                        eval_set=meta_eval if meta_eval else None,
                        **kwargs
                    )
            elif hasattr(self.meta_model, 'train'):
                # BaseModel interface
                self.meta_model.train(
                    X_meta, y_success,
                    sample_weights=sample_weights,
                    eval_set=meta_eval,
                    **kwargs
                )
            elif hasattr(self.meta_model, 'fit'):
                # Scikit-learn interface
                if sample_weights is not None:
                    self.meta_model.fit(X_meta, y_success, sample_weight=sample_weights)
                else:
                    self.meta_model.fit(X_meta, y_success)
            else:
                raise ValueError("Meta model must have 'train' or 'fit' method")
            
            logger.info("Meta model training completed")
            
        except Exception as e:
            logger.error(f"Meta model training failed: {e}")
            raise
        
        self.is_trained = True
        logger.info("=" * 60)
        logger.info("Meta-labeling training completed successfully")
        logger.info("=" * 60)
        
        return self
    
    def _create_meta_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        primary_proba: np.ndarray
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Create augmented features for meta model.
        
        Combines original features with primary model probabilities.
        This allows the meta model to learn which market conditions
        lead to successful vs. failed primary predictions.
        
        Args:
            X: Original features
            primary_proba: Primary model probability predictions
            
        Returns:
            Augmented feature matrix: [X | primary_probabilities]
        """
        if isinstance(X, pd.DataFrame):
            # Create DataFrame for probabilities with proper column names
            proba_cols = [f'primary_proba_{i}' for i in range(primary_proba.shape[1])]
            proba_df = pd.DataFrame(
                primary_proba,
                index=X.index,
                columns=proba_cols
            )
            # Concatenate original features with probabilities
            X_meta = pd.concat([X, proba_df], axis=1)
        else:
            # Numpy array concatenation
            X_meta = np.hstack([X, primary_proba])
        
        return X_meta
    
    def predict_meta(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Make filtered predictions using meta-labeling.
        
        This is the main prediction method that combines both models:
        1. Get direction prediction from primary model (Buy/Sell/Neutral)
        2. Get success probability from meta model
        3. Filter signal based on meta confidence threshold
        4. Return final signal, confidence, and raw signal
        
        Signal Filtering Logic:
            - If meta_conf < threshold: final_signal = 0 (don't trade)
            - If meta_conf >= threshold: final_signal = raw_signal (trade)
            - Neutral signals (raw_signal=0) always result in final_signal=0
        
        Returns AUTOMATICALLY DECODED labels in {-1, 0, 1} format if input model
        was trained on encoded labels {0, 1, 2}.

        Args:
            X: Features to predict on
            return_all: If True, return additional diagnostic information
            
        Returns:
            Dictionary with:
                - final_signal: Filtered trading signal (DECODED {-1, 0, 1})
                - meta_conf: Meta model confidence (success probability)
                - raw_signal: Original primary model prediction (DECODED {-1, 0, 1})
                - primary_proba: (optional) Full probability distribution
                - meta_proba: (optional) Full probability distribution
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        try:
            # Step 1: Get primary model predictions
            raw_signal_encoded = self.primary_model.predict(X)
            primary_proba = self.primary_model.predict_proba(X)
            
            # Step 2: Create augmented features for meta model
            X_meta = self._create_meta_features(X, primary_proba)
            
            # Step 3: Get meta model predictions (success probability)
            meta_proba_full = self.meta_model.predict_proba(X_meta)
            
            # Extract probability of success (class 1)
            if meta_proba_full.shape[1] == 2:
                meta_conf = meta_proba_full[:, 1]
            else:
                meta_conf = np.max(meta_proba_full, axis=1)
            
            # Step 4: Filter signals based on meta confidence
            final_signal_encoded = np.where(
                meta_conf >= self.meta_threshold,
                raw_signal_encoded,
                0 # Neutral
            )
            
            # Step 5: DECODE LABELS back to {-1, 0, 1}
            # We assume the internal model works on {0, 1, 2}
            raw_signal_decoded = decode_primary(raw_signal_encoded)
            final_signal_decoded = decode_primary(final_signal_encoded)

            # Build result dictionary
            result = {
                'final_signal': final_signal_decoded,
                'meta_conf': meta_conf,
                'raw_signal': raw_signal_decoded,
            }
            
            # Add full probability distributions if requested
            if return_all:
                result['primary_proba'] = primary_proba
                result['meta_proba'] = meta_proba_full
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy compatibility: returns (direction, success) predictions.
        DECODED labels {-1, 0, 1}.
        """
        result = self.predict_meta(X)
        success = (result['meta_conf'] >= self.meta_threshold).astype(int)
        return result['raw_signal'], success
    
    def predict_combined(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make combined predictions. DECODED labels {-1, 0, 1}.
        """
        result = self.predict_meta(X)
        return result['final_signal']
    
    def save(
        self,
        primary_path: Union[str, Path],
        meta_path: Union[str, Path]
    ) -> None:
        """
        Save both models to disk.
        
        Args:
            primary_path: Path to save primary model
            meta_path: Path to save meta model
            
        Example:
            >>> meta_model.save(
            ...     'models/primary_direction.cbm',
            ...     'models/meta_success.cbm'
            ... )
        """
        try:
            # Save primary model
            if _HAS_CATBOOST and isinstance(self.primary_model, CatBoostClassifier):
                self.primary_model.save_model(str(primary_path))
            elif hasattr(self.primary_model, 'save'):
                self.primary_model.save(primary_path)
            else:
                import joblib
                joblib.dump(self.primary_model, primary_path)
            
            # Save meta model
            if _HAS_CATBOOST and isinstance(self.meta_model, CatBoostClassifier):
                self.meta_model.save_model(str(meta_path))
            elif hasattr(self.meta_model, 'save'):
                self.meta_model.save(meta_path)
            else:
                import joblib
                joblib.dump(self.meta_model, meta_path)
            
            logger.info(f"Models saved: {primary_path}, {meta_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load(
        self,
        primary_path: Union[str, Path],
        meta_path: Union[str, Path]
    ) -> 'MetaModel':
        """
        Load both models from disk.
        
        Args:
            primary_path: Path to load primary model from
            meta_path: Path to load meta model from
            
        Returns:
            Self for method chaining
            
        Example:
            >>> meta_model = MetaModel(
            ...     CatBoostClassifier(),
            ...     CatBoostClassifier()
            ... )
            >>> meta_model.load(
            ...     'models/primary_direction.cbm',
            ...     'models/meta_success.cbm'
            ... )
        """
        try:
            # Load primary model
            if _HAS_CATBOOST and isinstance(self.primary_model, CatBoostClassifier):
                self.primary_model.load_model(str(primary_path))
            elif hasattr(self.primary_model, 'load'):
                self.primary_model.load(primary_path)
            else:
                import joblib
                self.primary_model = joblib.load(primary_path)
            
            # Load meta model
            if _HAS_CATBOOST and isinstance(self.meta_model, CatBoostClassifier):
                self.meta_model.load_model(str(meta_path))
            elif hasattr(self.meta_model, 'load'):
                self.meta_model.load(meta_path)
            else:
                import joblib
                self.meta_model = joblib.load(meta_path)
            
            self.is_trained = True
            logger.info(f"Models loaded: {primary_path}, {meta_path}")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def get_feature_importance(
        self,
        model_type: str = 'primary'
    ) -> Optional[pd.Series]:
        """
        Get feature importance from primary or meta model.
        
        Args:
            model_type: Which model to get importance from ('primary' or 'meta')
            
        Returns:
            Series with feature importance scores, or None if not available
        """
        model = self.primary_model if model_type == 'primary' else self.meta_model
        
        try:
            if _HAS_CATBOOST and isinstance(model, CatBoostClassifier):
                importance = model.get_feature_importance()
                feature_names = model.feature_names_
                return pd.Series(importance, index=feature_names)
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                if self.feature_names is not None:
                    return pd.Series(importance, index=self.feature_names)
                return pd.Series(importance)
            else:
                logger.warning(f"Feature importance not available for {type(model)}")
                return None
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "not trained"
        return (
            f"MetaModel({status}, threshold={self.meta_threshold:.2f}, "
            f"features={self.n_features})"
        )
