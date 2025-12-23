"""
Data preprocessing module for advanced financial time series.

This module implements preprocessing techniques from quantitative finance literature:
- Fractional Differentiation for stationarity with memory preservation
- Advanced sampling methods (Volume Bars, Dollar Bars) for noise reduction
- Statistical tests for stationarity verification

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging
from statsmodels.tsa.stattools import adfuller

from ..utils.math_tools import frac_diff_ffd

logger = logging.getLogger(__name__)


def is_stationary(
    series: pd.Series,
    significance_level: float = 0.05,
    max_lags: Optional[int] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Test if a time series is stationary using the Augmented Dickey-Fuller (ADF) test.
    
    The ADF test checks the null hypothesis that a unit root is present in the time
    series (i.e., the series is non-stationary). A p-value below the significance
    level allows us to reject the null hypothesis and conclude stationarity.
    
    Args:
        series: Time series to test. NaN values are automatically dropped.
        significance_level: Threshold for rejecting null hypothesis (default 0.05).
        max_lags: Maximum number of lags to use in ADF regression. If None,
                 uses automatic selection based on 12*(n/100)^(1/4).
    
    Returns:
        Tuple containing:
        - is_stationary (bool): True if series is stationary at given significance level
        - p_value (float): P-value from the ADF test
        - test_results (dict): Complete ADF test results including:
            - 'adf_statistic': Test statistic value
            - 'p_value': P-value
            - 'n_lags': Number of lags used
            - 'n_obs': Number of observations
            - 'critical_values': Critical values at 1%, 5%, and 10% levels
            - 'ic_best': Information criterion value
    
    Example:
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> is_stat, pval, results = is_stationary(prices)
        >>> print(f"Stationary: {is_stat}, p-value: {pval:.4f}")
        
    Notes:
        - Price series are typically non-stationary (random walk)
        - Returns and fractionally differentiated series should be stationary
        - Lower p-values provide stronger evidence of stationarity
    """
    # Remove NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        logger.warning("Series too short for reliable ADF test (<10 observations)")
        return False, 1.0, {}
    
    try:
        # Perform ADF test with automatic lag selection if not specified
        adf_result = adfuller(clean_series, maxlag=max_lags, autolag='AIC')
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        n_lags = adf_result[2]
        n_obs = adf_result[3]
        critical_values = adf_result[4]
        ic_best = adf_result[5]
        
        test_results = {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'n_lags': n_lags,
            'n_obs': n_obs,
            'critical_values': critical_values,
            'ic_best': ic_best
        }
        
        is_stat = p_value < significance_level
        
        logger.debug(
            f"ADF Test - Statistic: {adf_statistic:.4f}, "
            f"p-value: {p_value:.4f}, "
            f"Stationary: {is_stat}"
        )
        
        return is_stat, p_value, test_results
        
    except Exception as e:
        logger.error(f"Error in ADF test: {e}")
        return False, 1.0, {}


def frac_diff_with_adf_test(
    series: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.1,
    threshold: float = 1e-4,
    significance_level: float = 0.05
) -> Tuple[pd.Series, float, Dict[str, Any]]:
    """
    Find optimal fractional differentiation order and apply it to achieve stationarity.
    
    Iteratively tests different values of d to find the minimum differentiation order
    that achieves stationarity (as measured by ADF test). This preserves maximum
    memory while ensuring the series is suitable for ML models.
    
    Args:
        series: Input time series (e.g., price series).
        d_min: Minimum differentiation order to test (default 0.0).
        d_max: Maximum differentiation order to test (default 1.0).
        d_step: Step size for testing d values (default 0.1).
        threshold: Weight threshold for FFD (default 1e-4).
        significance_level: P-value threshold for ADF test (default 0.05).
    
    Returns:
        Tuple containing:
        - differentiated_series (pd.Series): Transformed series with optimal d
        - optimal_d (float): Differentiation order that achieved stationarity
        - results (dict): Dictionary with:
            - 'optimal_d': Best d value found
            - 'p_value': P-value at optimal d
            - 'adf_statistic': ADF statistic at optimal d
            - 'is_stationary': Whether stationarity was achieved
            - 'tested_d_values': List of all tested d values
            - 'p_values': P-values for each tested d
    
    Example:
        >>> import pandas as pd
        >>> prices = pd.Series([100, 101, 102, 103, 105, 108, 107])
        >>> stationary_series, d, info = frac_diff_with_adf_test(prices)
        >>> print(f"Optimal d: {d}, Stationary: {info['is_stationary']}")
        
    Notes:
        - Start with smallest d that achieves stationarity to preserve memory
        - If no d achieves stationarity, returns series with d=d_max
        - Typical optimal values are between 0.3 and 0.6 for financial time series
    """
    d_values = np.arange(d_min, d_max + d_step, d_step)
    p_values = []
    optimal_d = d_max  # Default to maximum if no stationary series found
    best_series = None
    
    logger.info(f"Testing fractional differentiation from d={d_min} to d={d_max}")
    
    for d in d_values:
        # Apply fractional differentiation
        diff_series = frac_diff_ffd(series, d=d, threshold=threshold)
        
        # Test for stationarity
        is_stat, p_val, _ = is_stationary(diff_series, significance_level)
        p_values.append(p_val)
        
        logger.debug(f"d={d:.2f}: p-value={p_val:.4f}, stationary={is_stat}")
        
        # Keep the first (minimum) d that achieves stationarity
        if is_stat and best_series is None:
            optimal_d = d
            best_series = diff_series
            logger.info(f"Found stationary series at d={optimal_d:.2f}")
            break
    
    # If no stationary series found, use maximum d
    if best_series is None:
        logger.warning(
            f"No stationary series found in range [{d_min}, {d_max}]. "
            f"Using d={d_max}"
        )
        best_series = frac_diff_ffd(series, d=d_max, threshold=threshold)
        is_stat, p_val, adf_results = is_stationary(best_series, significance_level)
    else:
        is_stat, p_val, adf_results = is_stationary(best_series, significance_level)
    
    results = {
        'optimal_d': optimal_d,
        'p_value': p_val,
        'adf_statistic': adf_results.get('adf_statistic', None),
        'is_stationary': is_stat,
        'tested_d_values': d_values.tolist(),
        'p_values': p_values
    }
    
    return best_series, optimal_d, results


class VolumeBars:
    """
    Generate Volume Bars for time series data.
    
    Volume Bars sample data at fixed volume intervals rather than fixed time intervals.
    This creates more uniform information content per bar and reduces the impact of
    low-activity periods, making the data more suitable for ML models.
    
    Each bar represents a fixed amount of volume traded, regardless of how long it takes.
    This approach:
    - Reduces noise during low-activity periods
    - Captures more information during high-activity periods
    - Creates more homoskedastic (constant variance) returns
    
    Attributes:
        volume_threshold: Volume required to complete one bar.
        
    Example:
        >>> volume_bars = VolumeBars(volume_threshold=1000.0)
        >>> df = pd.DataFrame({
        ...     'timestamp': [...],
        ...     'price': [...],
        ...     'volume': [...]
        ... })
        >>> bars = volume_bars.generate(df)
        
    References:
        López de Prado, M. (2018). Advances in Financial Machine Learning.
        Chapter 2: Financial Data Structures.
    """
    
    def __init__(self, volume_threshold: float):
        """
        Initialize Volume Bars generator.
        
        Args:
            volume_threshold: Total volume required to form one bar.
                            Should be calibrated to desired sampling frequency.
        """
        self.volume_threshold = volume_threshold
        
    def generate(
        self,
        data: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Generate Volume Bars from tick/trade data.
        
        Args:
            data: DataFrame with trade data, must be sorted by time.
            price_col: Name of price column.
            volume_col: Name of volume column.
            timestamp_col: Name of timestamp column.
        
        Returns:
            DataFrame with Volume Bars containing:
            - timestamp: Timestamp of bar close
            - open: First price in the bar
            - high: Highest price in the bar
            - low: Lowest price in the bar
            - close: Last price in the bar
            - volume: Total volume in the bar (≈ volume_threshold)
            - trades: Number of trades in the bar
            
        Raises:
            ValueError: If required columns are missing.
            ValueError: If data is empty.
        """
        # Validate input
        required_cols = [price_col, volume_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(data) == 0:
            raise ValueError("Input data is empty")
        
        bars = []
        accumulated_volume = 0.0
        bar_prices = []
        bar_start_time = None
        bar_trades = 0
        
        for idx, row in data.iterrows():
            price = row[price_col]
            volume = row[volume_col]
            timestamp = row[timestamp_col]
            
            # Initialize new bar if needed
            if bar_start_time is None:
                bar_start_time = timestamp
            
            # Accumulate data
            bar_prices.append(price)
            accumulated_volume += volume
            bar_trades += 1
            
            # Check if bar is complete
            if accumulated_volume >= self.volume_threshold:
                bars.append({
                    'timestamp': timestamp,
                    'open': bar_prices[0],
                    'high': max(bar_prices),
                    'low': min(bar_prices),
                    'close': bar_prices[-1],
                    'volume': accumulated_volume,
                    'trades': bar_trades
                })
                
                # Reset for next bar
                accumulated_volume = 0.0
                bar_prices = []
                bar_start_time = None
                bar_trades = 0
        
        # Handle remaining data (partial bar)
        if bar_prices:
            bars.append({
                'timestamp': data.iloc[-1][timestamp_col],
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                'volume': accumulated_volume,
                'trades': bar_trades
            })
        
        result = pd.DataFrame(bars)
        logger.info(f"Generated {len(result)} volume bars from {len(data)} trades")
        
        return result


class DollarBars:
    """
    Generate Dollar Bars for time series data.
    
    Dollar Bars (also called Value Bars) sample data at fixed dollar value intervals.
    Each bar represents a fixed monetary value traded (price * volume), providing
    even better normalization than Volume Bars for assets with significant price changes.
    
    This approach:
    - Adapts to price level changes automatically
    - More robust than Volume Bars for volatile assets
    - Provides consistent economic significance per bar
    
    Attributes:
        dollar_threshold: Dollar value required to complete one bar.
        
    Example:
        >>> dollar_bars = DollarBars(dollar_threshold=1000000.0)
        >>> df = pd.DataFrame({
        ...     'timestamp': [...],
        ...     'price': [...],
        ...     'volume': [...]
        ... })
        >>> bars = dollar_bars.generate(df)
        
    References:
        López de Prado, M. (2018). Advances in Financial Machine Learning.
        Chapter 2: Financial Data Structures.
    """
    
    def __init__(self, dollar_threshold: float):
        """
        Initialize Dollar Bars generator.
        
        Args:
            dollar_threshold: Total dollar value (price * volume) required to form one bar.
                            Should be calibrated based on market liquidity and desired frequency.
        """
        self.dollar_threshold = dollar_threshold
        
    def generate(
        self,
        data: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Generate Dollar Bars from tick/trade data.
        
        Args:
            data: DataFrame with trade data, must be sorted by time.
            price_col: Name of price column.
            volume_col: Name of volume column.
            timestamp_col: Name of timestamp column.
        
        Returns:
            DataFrame with Dollar Bars containing:
            - timestamp: Timestamp of bar close
            - open: First price in the bar
            - high: Highest price in the bar
            - low: Lowest price in the bar
            - close: Last price in the bar
            - volume: Total volume in the bar
            - dollar_value: Total dollar value in the bar (≈ dollar_threshold)
            - trades: Number of trades in the bar
            
        Raises:
            ValueError: If required columns are missing.
            ValueError: If data is empty.
        """
        # Validate input
        required_cols = [price_col, volume_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(data) == 0:
            raise ValueError("Input data is empty")
        
        bars = []
        accumulated_dollar_value = 0.0
        accumulated_volume = 0.0
        bar_prices = []
        bar_start_time = None
        bar_trades = 0
        
        for idx, row in data.iterrows():
            price = row[price_col]
            volume = row[volume_col]
            timestamp = row[timestamp_col]
            
            dollar_value = price * volume
            
            # Initialize new bar if needed
            if bar_start_time is None:
                bar_start_time = timestamp
            
            # Accumulate data
            bar_prices.append(price)
            accumulated_dollar_value += dollar_value
            accumulated_volume += volume
            bar_trades += 1
            
            # Check if bar is complete
            if accumulated_dollar_value >= self.dollar_threshold:
                bars.append({
                    'timestamp': timestamp,
                    'open': bar_prices[0],
                    'high': max(bar_prices),
                    'low': min(bar_prices),
                    'close': bar_prices[-1],
                    'volume': accumulated_volume,
                    'dollar_value': accumulated_dollar_value,
                    'trades': bar_trades
                })
                
                # Reset for next bar
                accumulated_dollar_value = 0.0
                accumulated_volume = 0.0
                bar_prices = []
                bar_start_time = None
                bar_trades = 0
        
        # Handle remaining data (partial bar)
        if bar_prices:
            bars.append({
                'timestamp': data.iloc[-1][timestamp_col],
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                'volume': accumulated_volume,
                'dollar_value': accumulated_dollar_value,
                'trades': bar_trades
            })
        
        result = pd.DataFrame(bars)
        logger.info(
            f"Generated {len(result)} dollar bars from {len(data)} trades "
            f"(threshold: ${self.dollar_threshold:,.0f})"
        )
        
        return result
