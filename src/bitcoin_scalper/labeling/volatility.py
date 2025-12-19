"""
Dynamic volatility estimation for financial time series.

This module implements volatility estimators used to set dynamic barriers in the
Triple Barrier Method. Volatility is estimated using Exponential Weighted Moving
Average (EWMA) which gives more weight to recent observations.

The volatility estimate is crucial for:
- Setting appropriate profit-taking (upper barrier) levels
- Setting appropriate stop-loss (lower barrier) levels
- Adapting to changing market conditions (high vs low volatility regimes)

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    RiskMetrics Technical Document (1996). J.P. Morgan/Reuters.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_daily_volatility(
    close: pd.Series,
    span: int = 100
) -> pd.Series:
    """
    Calculate daily volatility using Exponential Weighted Moving Average (EWMA).
    
    Daily volatility is estimated from intraday returns using an EWMA estimator.
    This provides a dynamic measure that adapts to recent market conditions while
    smoothing out noise.
    
    The EWMA gives exponentially decreasing weights to older observations:
        σ_t = √(EWMA(r_t²))
    
    where r_t are the log returns and the decay parameter is calculated from span.
    
    Args:
        close: Series of closing prices with DatetimeIndex.
        span: Number of periods for EWMA calculation (default 100).
              Higher values give smoother estimates, lower values adapt faster.
              Common values: 20 (fast), 100 (standard), 252 (slow).
    
    Returns:
        Series of daily volatility estimates aligned with input index.
        Values represent the standard deviation of returns.
    
    Example:
        >>> prices = pd.Series([100, 101, 99, 102, 98], 
        ...                    index=pd.date_range('2024-01-01', periods=5, freq='1min'))
        >>> vol = calculate_daily_volatility(prices, span=20)
        >>> print(f"Current volatility: {vol.iloc[-1]:.4f}")
        
    Notes:
        - Returns are calculated as log(P_t / P_{t-1})
        - First value will be NaN due to return calculation
        - Volatility is annualized for intraday data based on sampling frequency
        - For Bitcoin, volatility can vary significantly across regimes
    """
    if len(close) < 2:
        logger.warning("Insufficient data for volatility calculation (need at least 2 points)")
        return pd.Series(np.nan, index=close.index)
    
    # Calculate log returns
    returns = np.log(close / close.shift(1))
    
    # Calculate EWMA of squared returns
    # alpha = 2 / (span + 1) for EWMA
    returns_squared = returns ** 2
    ewma_var = returns_squared.ewm(span=span, min_periods=span).mean()
    
    # Volatility is square root of variance
    volatility = np.sqrt(ewma_var)
    
    # Handle any NaN or infinite values
    volatility = volatility.replace([np.inf, -np.inf], np.nan)
    
    logger.debug(
        f"Calculated daily volatility: "
        f"mean={volatility.mean():.6f}, "
        f"std={volatility.std():.6f}, "
        f"min={volatility.min():.6f}, "
        f"max={volatility.max():.6f}"
    )
    
    return volatility


def estimate_ewma_volatility(
    prices: Union[pd.Series, pd.DataFrame],
    span: int = 100,
    price_col: Optional[str] = None
) -> pd.Series:
    """
    Estimate volatility using EWMA for use in barrier calculations.
    
    This is a convenience wrapper around calculate_daily_volatility that handles
    both Series and DataFrame inputs. It's the primary function to use for
    volatility estimation in the Triple Barrier Method.
    
    Args:
        prices: Series of prices or DataFrame containing price column.
        span: EWMA span parameter (default 100). Controls adaptation speed:
              - Smaller values (20-50): Fast adaptation, more noise
              - Medium values (100): Good balance for most applications
              - Larger values (200+): Smooth but slow to adapt
        price_col: Column name if prices is a DataFrame. If None and DataFrame
                  is provided, will try common column names: 'close', '<CLOSE>',
                  '1min_<CLOSE>', '1min_close'.
    
    Returns:
        Series of volatility estimates with same index as input.
        
    Raises:
        ValueError: If price_col not found in DataFrame or prices is empty.
        
    Example:
        >>> # With Series
        >>> vol = estimate_ewma_volatility(price_series, span=100)
        
        >>> # With DataFrame
        >>> vol = estimate_ewma_volatility(df, span=100, price_col='close')
        
        >>> # With DataFrame, auto-detect column
        >>> vol = estimate_ewma_volatility(df, span=100)
        
    Notes:
        - Volatility is in the same units as the returns (typically decimal)
        - For barrier calculation, multiply by a factor (e.g., 2) to set width
        - Consider market regime when choosing span parameter
    """
    # Extract price series from DataFrame if needed
    if isinstance(prices, pd.DataFrame):
        if price_col is None:
            # Try common column names
            candidates = ['close', '<CLOSE>', '1min_<CLOSE>', '1min_close']
            price_col = next((col for col in candidates if col in prices.columns), None)
            
            if price_col is None:
                raise ValueError(
                    f"Could not find price column. Available columns: {list(prices.columns)}. "
                    f"Please specify price_col parameter."
                )
        
        if price_col not in prices.columns:
            raise ValueError(
                f"Column '{price_col}' not found in DataFrame. "
                f"Available columns: {list(prices.columns)}"
            )
        
        price_series = prices[price_col]
    else:
        price_series = prices
    
    if len(price_series) == 0:
        raise ValueError("Input prices are empty")
    
    # Calculate volatility
    volatility = calculate_daily_volatility(price_series, span=span)
    
    logger.info(
        f"Estimated EWMA volatility (span={span}): "
        f"current={volatility.iloc[-1] if len(volatility) > 0 else 'N/A':.6f}, "
        f"mean={volatility.mean():.6f}"
    )
    
    return volatility


def get_adaptive_span(
    market_regime: str = 'normal',
    frequency: str = '1min'
) -> int:
    """
    Get recommended EWMA span based on market regime and data frequency.
    
    Different market conditions require different adaptation speeds. This function
    provides sensible defaults that can be fine-tuned based on specific needs.
    
    Args:
        market_regime: Market condition, one of:
            - 'volatile': High volatility, fast changes (use short span)
            - 'normal': Normal market conditions (use medium span)
            - 'stable': Low volatility, stable conditions (use long span)
        frequency: Data frequency, one of:
            - '1min', '5min', '15min', '1h', '1d'
    
    Returns:
        Recommended span parameter for EWMA calculation.
        
    Example:
        >>> span = get_adaptive_span(market_regime='volatile', frequency='1min')
        >>> vol = estimate_ewma_volatility(prices, span=span)
        
    Notes:
        - These are starting points; backtesting should be used to optimize
        - Higher frequency data can use shorter spans
        - During market stress, consider switching to 'volatile' regime
    """
    # Base spans for 1-minute data
    regime_spans = {
        'volatile': 50,   # Fast adaptation for volatile markets
        'normal': 100,    # Balanced for normal conditions
        'stable': 200     # Smooth for stable markets
    }
    
    # Frequency multipliers (relative to 1min)
    freq_multipliers = {
        '1min': 1.0,
        '5min': 0.6,    # 5min has 1/5 the observations
        '15min': 0.4,   # 15min has 1/15 the observations
        '1h': 0.2,      # 1h has 1/60 the observations
        '1d': 0.05      # 1d has 1/1440 the observations
    }
    
    base_span = regime_spans.get(market_regime, 100)
    multiplier = freq_multipliers.get(frequency, 1.0)
    
    recommended_span = int(base_span * multiplier)
    
    # Ensure minimum span
    recommended_span = max(recommended_span, 10)
    
    logger.debug(
        f"Recommended span for {market_regime} regime at {frequency} frequency: "
        f"{recommended_span}"
    )
    
    return recommended_span
