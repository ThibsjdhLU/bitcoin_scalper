"""
Data Requirements Constants for Feature Engineering.

This module defines the minimum historical data requirements needed
for proper feature calculation in the trading system.

The requirements are based on the longest-window indicators used:
- SMA/EMA 200: 200 periods
- Multi-timeframe (5min from 1min): 5x multiplier
- NaN warm-up period: ~200-300 rows
- Safety buffer: Additional margin for stability
"""

# Individual indicator requirements (in periods)
INDICATOR_WINDOWS = {
    'fracdiff': 23,        # FracDiff with d=0.4, threshold=1e-4
    'rsi': 21,             # RSI max window (7, 14, 21)
    'macd': 34,            # MACD (12, 26, 9)
    'ema': 200,            # EMA max window (21, 50, 200)
    'sma': 200,            # SMA max window (20, 50, 200)
    'bollinger': 50,       # Bollinger Bands max window (20, 50)
    'atr': 21,             # ATR max window (14, 21)
    'supertrend': 7,       # SuperTrend length
    'ichimoku': 52,        # Ichimoku (9, 26, 52)
    'volume_sma': 20,      # Volume moving average
    'zscore': 100,         # Z-score max window (5, 20, 50, 100)
    'volatility': 20,      # Rolling volatility window
    'rolling_minmax': 100  # Rolling high/low windows
}

# Maximum lookback required across all indicators
MAX_LOOKBACK_WINDOW = max(INDICATOR_WINDOWS.values())  # 200

# Minimum data requirements for single timeframe
MIN_ROWS_SINGLE_TIMEFRAME = MAX_LOOKBACK_WINDOW + 100  # 300

# Minimum data requirements for multi-timeframe (1min + 5min)
# 5-minute resampling requires 5x more 1-minute data
RESAMPLING_MULTIPLIER = 5
MIN_ROWS_MULTI_TIMEFRAME = MAX_LOOKBACK_WINDOW * RESAMPLING_MULTIPLIER + 100  # 1100

# Safe minimum with buffer for NaN removal and edge cases
SAFE_MIN_ROWS = 1500

# Default fetch limit for connectors
DEFAULT_FETCH_LIMIT = 3000

# Minimum rows after feature engineering (post-NaN removal)
MIN_ROWS_AFTER_FEATURE_ENG = 300

# Error messages
ERROR_INSUFFICIENT_DATA = (
    "Insufficient historical data for feature engineering. "
    f"Minimum required: {SAFE_MIN_ROWS} candles"
)

ERROR_INSUFFICIENT_AFTER_NAN = (
    "Insufficient data after NaN removal in feature engineering. "
    f"Minimum required: {MIN_ROWS_AFTER_FEATURE_ENG} rows, but only {{rows}} remain. "
    "Try fetching more historical data."
)


def validate_data_requirements(
    df_len: int,
    stage: str = "pre_processing"
) -> tuple[bool, str]:
    """
    Validate if dataframe has sufficient rows for processing.
    
    Args:
        df_len: Length of the dataframe
        stage: Processing stage - "pre_processing" or "post_processing"
    
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> valid, msg = validate_data_requirements(len(df), "pre_processing")
        >>> if not valid:
        ...     logger.error(msg)
        ...     return pd.DataFrame()
    """
    if stage == "pre_processing":
        if df_len < SAFE_MIN_ROWS:
            msg = (
                f"Insufficient input data: {df_len} rows "
                f"(minimum required: {SAFE_MIN_ROWS}). "
                "Feature engineering requires at least 1500 historical candles "
                "for multi-timeframe analysis (1min + 5min). "
                "Please increase the 'limit' parameter when fetching data."
            )
            return False, msg
    elif stage == "post_processing":
        if df_len < MIN_ROWS_AFTER_FEATURE_ENG:
            msg = ERROR_INSUFFICIENT_AFTER_NAN.format(rows=df_len)
            return False, msg
    
    return True, ""


def get_recommended_fetch_limit(timeframe: str = "1m") -> int:
    """
    Get recommended fetch limit based on timeframe.
    
    Args:
        timeframe: Target timeframe (e.g., "1m", "5m", "1h")
    
    Returns:
        Recommended number of candles to fetch
        
    Example:
        >>> limit = get_recommended_fetch_limit("1m")
        >>> df = connector.fetch_ohlcv(symbol, "1m", limit)
    """
    # For 1-minute data with multi-timeframe features
    if timeframe in ["1m", "1min", "M1"]:
        return SAFE_MIN_ROWS
    # For 5-minute data
    elif timeframe in ["5m", "5min", "M5"]:
        return MIN_ROWS_SINGLE_TIMEFRAME
    # For higher timeframes, use single timeframe minimum
    else:
        return MIN_ROWS_SINGLE_TIMEFRAME
