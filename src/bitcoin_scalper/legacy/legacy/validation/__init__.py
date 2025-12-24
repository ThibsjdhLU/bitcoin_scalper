"""
Validation module for financial machine learning.

This module provides tools for scientifically valid backtesting and model validation:
- Combinatorial Purged Cross-Validation (CPCV) for time series
- Drift detection using ADWIN algorithm
- Event-driven backtesting engine

These tools help prevent common pitfalls like look-ahead bias and overfitting.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from .cross_val import PurgedKFold, CombinatorialPurgedCV
from .drift import DriftScanner, ADWINDetector
from .backtest import Backtester, BacktestResult

__all__ = [
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'DriftScanner',
    'ADWINDetector',
    'Backtester',
    'BacktestResult',
]
