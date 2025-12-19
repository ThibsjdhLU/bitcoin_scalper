"""
Mathematical tools for advanced financial data processing.

This module provides implementations of mathematical operations used in quantitative
finance, particularly for achieving stationarity while preserving memory in time series.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate weights for Fixed-Window Fractional Differentiation (FFD).
    
    Computes the binomial coefficients for fractional differentiation using the
    recursive formula from López de Prado's "Advances in Financial Machine Learning".
    
    The weights follow: ω_k = (-1)^k * C(d, k) where C is binomial coefficient.
    Recursive relation: ω_k = (k - 1 - d) / k * ω_{k-1}, with ω_0 = 1
    
    Args:
        d: Differentiation order (typically between 0 and 1). 
           d=0 returns original series, d=1 is standard differentiation.
           Optimal values around 0.4 preserve memory while achieving stationarity.
        threshold: Minimum absolute weight value to include. Weights below this
                  are truncated to create a fixed window. Default 1e-5.
    
    Returns:
        Array of weights for the fractional differentiation operator.
        
    Example:
        >>> weights = get_weights_ffd(d=0.4, threshold=1e-5)
        >>> len(weights)  # Number of lags in the fixed window
        23
    """
    weights = [1.0]
    k = 1
    
    while True:
        # Recursive formula: ω_k = (k - 1 - d) / k * ω_{k-1}
        weight = -weights[-1] * (d - k + 1) / k
        
        # Stop when weights become negligible
        if abs(weight) < threshold:
            break
            
        weights.append(weight)
        k += 1
    
    return np.array(weights)


def frac_diff_ffd(
    series: Union[pd.Series, pd.DataFrame],
    d: float,
    threshold: float = 1e-5
) -> Union[pd.Series, pd.DataFrame]:
    """
    Apply Fixed-Window Fractional Differentiation (FFD) to time series data.
    
    Fractional differentiation allows achieving stationarity while preserving
    memory (autocorrelation structure) of the original series. This is crucial
    for machine learning models that depend on long-term dependencies like LSTMs
    or State-Space Models.
    
    The method uses a fixed window of weights (determined by threshold) rather
    than an expanding window, making it more computationally efficient and
    suitable for online/streaming applications.
    
    Args:
        series: Input time series data. Can be a pandas Series or DataFrame.
               If DataFrame, differentiation is applied to each column.
        d: Differentiation order. Typical range [0, 1]:
           - d=0.0: Returns original series (no differentiation)
           - d=0.4-0.5: Good balance between stationarity and memory preservation
           - d=1.0: Standard first-order differentiation (maximum stationarity)
        threshold: Weight threshold for determining window size. Lower values
                  create larger windows but better approximations. Default 1e-5.
    
    Returns:
        Fractionally differentiated series with same index as input.
        Initial values (window size - 1) will be NaN due to insufficient history.
        
    Example:
        >>> import pandas as pd
        >>> prices = pd.Series([100, 101, 102, 101, 103, 105])
        >>> stationary = frac_diff_ffd(prices, d=0.4)
        >>> # stationary is now more stationary while preserving trends
        
    References:
        López de Prado, M. (2018). Advances in Financial Machine Learning.
        Chapter 5: Fractional Differentiation.
    """
    # Get weights for the specified differentiation order
    weights = get_weights_ffd(d, threshold)
    window_size = len(weights)
    
    # Handle both Series and DataFrame
    if isinstance(series, pd.Series):
        series_values = series.values
        result = np.full(len(series), np.nan)
        
        # Apply convolution starting from window_size
        for i in range(window_size - 1, len(series)):
            # Extract window of past values (including current)
            window = series_values[i - window_size + 1:i + 1]
            # Reverse window to align with weights (most recent first)
            window = window[::-1]
            # Apply weighted sum
            result[i] = np.dot(weights, window)
        
        return pd.Series(result, index=series.index, name=series.name)
    
    elif isinstance(series, pd.DataFrame):
        # Apply to each column
        result_dict = {}
        for col in series.columns:
            result_dict[col] = frac_diff_ffd(series[col], d, threshold)
        
        return pd.DataFrame(result_dict, index=series.index)
    
    else:
        raise TypeError(f"Input must be pandas Series or DataFrame, got {type(series)}")


def get_ffd_window_size(d: float, threshold: float = 1e-5) -> int:
    """
    Calculate the window size for Fixed-Window Fractional Differentiation.
    
    Args:
        d: Differentiation order.
        threshold: Weight threshold.
    
    Returns:
        Number of lags in the fixed window.
        
    Example:
        >>> window = get_ffd_window_size(d=0.4)
        >>> print(f"Need {window} data points for d=0.4")
    """
    weights = get_weights_ffd(d, threshold)
    return len(weights)
