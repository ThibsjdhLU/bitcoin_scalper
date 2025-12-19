"""
ML Models Module for Bitcoin Scalper.

This module provides a unified interface for machine learning models including:
- Classical ML (XGBoost, CatBoost)
- Deep Learning (LSTM, Transformer via PyTorch)
- Training pipelines with meta-labeling support
- Integration with Triple Barrier labeling method

The architecture follows the Model Factory pattern, allowing seamless model
swapping while maintaining consistent interfaces.
"""

from src.bitcoin_scalper.models.base import BaseModel
from src.bitcoin_scalper.models.gradient_boosting import (
    XGBoostClassifier,
    XGBoostRegressor,
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper
)
from src.bitcoin_scalper.models.pipeline import Trainer, MetaLabelingPipeline

__all__ = [
    'BaseModel',
    'XGBoostClassifier',
    'XGBoostRegressor',
    'CatBoostClassifierWrapper',
    'CatBoostRegressorWrapper',
    'Trainer',
    'MetaLabelingPipeline'
]
