"""
Labeling module for financial machine learning.

This module implements advanced labeling techniques from quantitative finance literature,
including the Triple Barrier Method, meta-labeling, and dynamic volatility estimation.

Modules:
    volatility: Daily volatility estimation using Exponential Weighted Moving Average (EWMA)
    barriers: Triple Barrier Method for event-based labeling
    labels: Primary and meta-labeling functions for supervised learning

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from .volatility import (
    calculate_daily_volatility,
    estimate_ewma_volatility
)

from .barriers import (
    get_events,
    apply_triple_barrier
)

from .labels import (
    get_labels,
    get_meta_labels,
    generate_labels_from_barriers
)

__all__ = [
    # Volatility
    'calculate_daily_volatility',
    'estimate_ewma_volatility',
    
    # Barriers
    'get_events',
    'apply_triple_barrier',
    
    # Labels
    'get_labels',
    'get_meta_labels',
    'generate_labels_from_barriers',
]
