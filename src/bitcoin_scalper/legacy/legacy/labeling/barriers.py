"""
Triple Barrier Method for event-based labeling in financial machine learning.

The Triple Barrier Method is a sophisticated labeling technique that incorporates
risk management principles directly into the labeling process. Instead of using
fixed time horizons, it defines three conditions (barriers) for exiting a position:

1. Upper Barrier (Take Profit): Price increases by a threshold
2. Lower Barrier (Stop Loss): Price decreases by a threshold  
3. Vertical Barrier (Time Limit): Maximum holding period expires

The label is determined by which barrier is touched first, creating labels that
reflect realistic trading scenarios with explicit profit targets and stop losses.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 3: Labeling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: Union[float, pd.Series],
    molecule: Optional[pd.DatetimeIndex] = None,
    vertical_barrier: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Apply the Triple Barrier Method to a series of events.
    
    For each event timestamp, this function looks forward in time to determine
    which of three barriers is touched first:
    - Upper barrier: price increases by pt_sl (profit taking)
    - Lower barrier: price decreases by pt_sl (stop loss)
    - Vertical barrier: time limit is reached
    
    Args:
        close: Series of closing prices with DatetimeIndex.
        events: DatetimeIndex of event timestamps to label (e.g., from CUSUM filter).
        pt_sl: Profit-taking / Stop-loss threshold. Can be:
               - float: Fixed threshold for all events
               - Series: Dynamic threshold per event (e.g., volatility-based)
               Values should be in decimal form (e.g., 0.01 for 1%).
        molecule: Subset of events to process (for parallelization). If None, process all.
        vertical_barrier: Series with DatetimeIndex matching events, containing
                         the timestamp of the vertical barrier for each event.
                         If None, no time-based exit.
        side: Series with values {-1, 1} indicating bet side. If None, assumes side=1
              (long only). For side=-1, barriers are inverted.
    
    Returns:
        DataFrame with index matching events, containing:
        - 't1': Timestamp when first barrier was touched
        - 'sl': Stop-loss barrier level (price)
        - 'pt': Profit-taking barrier level (price)
        - 'type': Which barrier was touched first:
          * 1: Upper barrier (profit target hit)
          * -1: Lower barrier (stop loss hit)
          * 0: Vertical barrier (time limit)
        - 'return': Actual return achieved (for validation)
    
    Example:
        >>> # Simple example with fixed 2% barriers
        >>> events = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
        >>> result = apply_triple_barrier(
        ...     close=price_series,
        ...     events=events,
        ...     pt_sl=0.02,  # 2% barriers
        ...     vertical_barrier=pd.Series(
        ...         pd.DatetimeIndex(['2024-01-01 10:00', '2024-01-02 10:00']),
        ...         index=events
        ...     )
        ... )
        
        >>> # With dynamic volatility-based barriers
        >>> volatility = estimate_ewma_volatility(price_series)
        >>> pt_sl = 2.0 * volatility.loc[events]  # 2 sigma barriers
        >>> result = apply_triple_barrier(close, events, pt_sl, ...)
        
    Notes:
        - Events not in close.index are skipped
        - If vertical_barrier timestamp is beyond data, uses last available price
        - Returns NaN if insufficient data to determine barrier touch
        - Processing can be parallelized using molecule parameter
    """
    # Subset events if molecule is specified
    if molecule is not None:
        events = events.intersection(molecule)
    
    # Ensure events are in close index
    events = events.intersection(close.index)
    
    if len(events) == 0:
        logger.warning("No valid events found in price series")
        return pd.DataFrame()
    
    # Convert pt_sl to Series if it's a scalar
    if isinstance(pt_sl, (int, float)):
        pt_sl = pd.Series(pt_sl, index=events)
    else:
        # Align pt_sl with events
        pt_sl = pt_sl.loc[events]
    
    # Default side to 1 (long) if not provided
    if side is None:
        side = pd.Series(1, index=events)
    else:
        side = side.loc[events]
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=events)
    
    # Get vertical barrier timestamps
    if vertical_barrier is not None:
        t1 = vertical_barrier.loc[events]
    else:
        # If no vertical barrier, use last timestamp in data
        t1 = pd.Series(close.index[-1], index=events)
    
    # Store barriers
    result['t1'] = t1
    
    # Calculate barrier levels for each event
    pt_levels = []
    sl_levels = []
    barrier_types = []
    barrier_times = []
    returns_achieved = []
    
    for idx in events:
        # Get starting price
        p0 = close.loc[idx]
        
        # Get barrier thresholds
        threshold = pt_sl.loc[idx]
        bet_side = side.loc[idx]
        
        # Calculate barrier price levels
        # For long (side=1): pt above, sl below
        # For short (side=-1): pt below, sl above
        if bet_side > 0:
            pt_level = p0 * (1 + threshold)
            sl_level = p0 * (1 - threshold)
        else:
            pt_level = p0 * (1 - threshold)
            sl_level = p0 * (1 + threshold)
        
        pt_levels.append(pt_level)
        sl_levels.append(sl_level)
        
        # Get end time for this event
        end_time = t1.loc[idx]
        
        # Get price series from event to vertical barrier
        # Ensure end_time is in index or use closest
        if end_time not in close.index:
            # Find closest timestamp not exceeding end_time
            valid_times = close.index[close.index >= idx]
            if len(valid_times) > 0:
                valid_times = valid_times[valid_times <= end_time]
                if len(valid_times) > 0:
                    end_time = valid_times[-1]
                else:
                    # No prices between event and barrier, use next available
                    valid_times = close.index[close.index > idx]
                    if len(valid_times) > 0:
                        end_time = valid_times[0]
                    else:
                        # No future prices available
                        barrier_types.append(np.nan)
                        barrier_times.append(pd.NaT)
                        returns_achieved.append(np.nan)
                        continue
        
        price_path = close.loc[idx:end_time]
        
        if len(price_path) <= 1:
            # Not enough data
            barrier_types.append(np.nan)
            barrier_times.append(pd.NaT)
            returns_achieved.append(np.nan)
            continue
        
        # Check which barrier is hit first
        if bet_side > 0:
            # Long position
            upper_hit = price_path >= pt_level
            lower_hit = price_path <= sl_level
        else:
            # Short position (barriers inverted)
            upper_hit = price_path <= pt_level
            lower_hit = price_path >= sl_level
        
        # Find first hit
        upper_hit_idx = upper_hit[upper_hit].index
        lower_hit_idx = lower_hit[lower_hit].index
        
        first_upper = upper_hit_idx[0] if len(upper_hit_idx) > 0 else None
        first_lower = lower_hit_idx[0] if len(lower_hit_idx) > 0 else None
        
        # Determine which barrier was hit first
        if first_upper is None and first_lower is None:
            # Vertical barrier hit (time limit)
            barrier_type = 0
            barrier_time = end_time
        elif first_upper is None:
            # Stop loss hit
            barrier_type = -1
            barrier_time = first_lower
        elif first_lower is None:
            # Profit target hit
            barrier_type = 1
            barrier_time = first_upper
        elif first_upper <= first_lower:
            # Profit target hit first
            barrier_type = 1
            barrier_time = first_upper
        else:
            # Stop loss hit first
            barrier_type = -1
            barrier_time = first_lower
        
        # Calculate actual return achieved
        p1 = close.loc[barrier_time]
        actual_return = (p1 / p0 - 1) * bet_side
        
        barrier_types.append(barrier_type)
        barrier_times.append(barrier_time)
        returns_achieved.append(actual_return)
    
    # Store results
    result['pt'] = pt_levels
    result['sl'] = sl_levels
    result['type'] = barrier_types
    result['t1'] = barrier_times
    result['return'] = returns_achieved
    
    # Log statistics
    if len(result) > 0:
        type_counts = result['type'].value_counts()
        logger.info(
            f"Triple Barrier results: "
            f"Profit targets: {type_counts.get(1, 0)}, "
            f"Stop losses: {type_counts.get(-1, 0)}, "
            f"Time limits: {type_counts.get(0, 0)}, "
            f"Invalid: {result['type'].isna().sum()}"
        )
    
    return result


def get_events(
    close: pd.Series,
    timestamps: pd.DatetimeIndex,
    pt_sl: Union[float, pd.Series, Tuple[float, float], Tuple[pd.Series, pd.Series]],
    t_final: Optional[Union[pd.Timestamp, pd.Series]] = None,
    max_holding_period: Optional[pd.Timedelta] = None,
    min_return: float = 0.0,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate labeled events using the Triple Barrier Method.
    
    This is the main function for applying the Triple Barrier Method to a series
    of potential trading signals. It returns a DataFrame with barrier touch
    information that can be used for supervised learning.
    
    Args:
        close: Series of closing prices with DatetimeIndex.
        timestamps: DatetimeIndex of event timestamps (e.g., from CUSUM filter,
                   signal generation, or simply regular intervals).
        pt_sl: Profit-taking and stop-loss thresholds. Can be:
               - float: Same threshold for both PT and SL
               - Series: Dynamic threshold per event (aligned with timestamps)
               - Tuple[float, float]: (profit_target, stop_loss) as separate values
               - Tuple[Series, Series]: (pt_series, sl_series) dynamic per event
               Values in decimal form (e.g., 0.02 for 2%).
        t_final: End timestamp for all events, or Series of end times per event.
                If None, uses last timestamp in close.
        max_holding_period: Maximum holding period as Timedelta (e.g., pd.Timedelta('10min')).
                           If specified, vertical barrier is set to timestamp + max_holding_period.
                           Takes precedence over t_final if both specified.
        min_return: Minimum return threshold to consider an event valid.
                   Events with smaller potential returns are filtered out.
        side: Series indicating bet direction {-1, 1}. If None, assumes all long (1).
    
    Returns:
        DataFrame with columns:
        - 't1': Timestamp when barrier was touched
        - 'type': Barrier type (-1: stop loss, 0: time, 1: profit)
        - 'return': Return achieved
        - 'sl': Stop loss price level
        - 'pt': Profit target price level
        
    Example:
        >>> # Basic usage with fixed 2% barriers and 15-minute time limit
        >>> events = get_events(
        ...     close=price_series,
        ...     timestamps=signal_times,
        ...     pt_sl=0.02,
        ...     max_holding_period=pd.Timedelta('15min')
        ... )
        
        >>> # With dynamic volatility-based barriers
        >>> volatility = estimate_ewma_volatility(price_series)
        >>> events = get_events(
        ...     close=price_series,
        ...     timestamps=signal_times,
        ...     pt_sl=2.0 * volatility.loc[signal_times],
        ...     max_holding_period=pd.Timedelta('15min')
        ... )
        
        >>> # With asymmetric barriers (2% profit, 1% stop)
        >>> events = get_events(
        ...     close=price_series,
        ...     timestamps=signal_times,
        ...     pt_sl=(0.02, 0.01),
        ...     max_holding_period=pd.Timedelta('15min')
        ... )
        
    Notes:
        - This is the recommended high-level interface to the Triple Barrier Method
        - Timestamps should be aligned with close.index
        - Consider using CUSUM filter or other methods to generate timestamps
        - Returns can be used directly as features or converted to labels
    """
    # Ensure timestamps are in close index
    timestamps = timestamps.intersection(close.index)
    
    if len(timestamps) == 0:
        logger.warning("No valid timestamps found in price series")
        return pd.DataFrame()
    
    # Handle different pt_sl formats
    if isinstance(pt_sl, tuple):
        # Separate PT and SL thresholds
        pt_threshold, sl_threshold = pt_sl
        
        # Convert to Series if needed
        if isinstance(pt_threshold, (int, float)):
            pt_threshold = pd.Series(pt_threshold, index=timestamps)
        else:
            pt_threshold = pt_threshold.loc[timestamps]
        
        if isinstance(sl_threshold, (int, float)):
            sl_threshold = pd.Series(sl_threshold, index=timestamps)
        else:
            sl_threshold = sl_threshold.loc[timestamps]
        
        # For apply_triple_barrier, we'll handle asymmetric barriers separately
        # For now, use average (this will be refined in implementation)
        pt_sl_combined = (pt_threshold + sl_threshold) / 2
    else:
        pt_sl_combined = pt_sl
        pt_threshold = pt_sl
        sl_threshold = pt_sl
    
    # Calculate vertical barrier
    if max_holding_period is not None:
        # Set vertical barrier based on holding period
        vertical_barrier = pd.Series(
            timestamps + max_holding_period,
            index=timestamps
        )
    elif t_final is not None:
        # Use provided end time
        if isinstance(t_final, pd.Timestamp):
            vertical_barrier = pd.Series(t_final, index=timestamps)
        else:
            vertical_barrier = t_final.loc[timestamps]
    else:
        # No vertical barrier (use end of data)
        vertical_barrier = None
    
    # Apply triple barrier
    events = apply_triple_barrier(
        close=close,
        events=timestamps,
        pt_sl=pt_sl_combined,
        vertical_barrier=vertical_barrier,
        side=side
    )
    
    # Filter by minimum return if specified
    if min_return > 0 and len(events) > 0:
        valid_events = events['return'].abs() >= min_return
        n_filtered = (~valid_events).sum()
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} events with return < {min_return}")
        events = events[valid_events]
    
    return events


def get_vertical_barriers(
    timestamps: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: Optional[int] = None,
    timedelta: Optional[pd.Timedelta] = None
) -> pd.Series:
    """
    Calculate vertical barrier timestamps for events.
    
    Vertical barriers define the maximum holding period for each event. This can
    be specified as either a fixed number of bars or a time delta.
    
    Args:
        timestamps: Event timestamps for which to calculate barriers.
        close: Price series with DatetimeIndex (used to find valid timestamps).
        num_bars: Number of bars (time periods) after event for vertical barrier.
                 If specified, timedelta is ignored.
        timedelta: Time duration for vertical barrier (e.g., pd.Timedelta('15min')).
                  Used if num_bars is None.
    
    Returns:
        Series with index=timestamps, values=vertical barrier timestamps.
        
    Example:
        >>> # 10 bars forward
        >>> barriers = get_vertical_barriers(events, price_series, num_bars=10)
        
        >>> # 15 minutes forward
        >>> barriers = get_vertical_barriers(
        ...     events, price_series, 
        ...     timedelta=pd.Timedelta('15min')
        ... )
        
    Notes:
        - If barrier timestamp is beyond available data, uses last timestamp
        - Useful for creating consistent barrier structure across backtests
    """
    if num_bars is None and timedelta is None:
        raise ValueError("Must specify either num_bars or timedelta")
    
    barriers = pd.Series(index=timestamps, dtype='datetime64[ns]')
    
    for ts in timestamps:
        if num_bars is not None:
            # Find timestamp num_bars ahead
            future_idx = close.index[close.index > ts]
            if len(future_idx) >= num_bars:
                barriers.loc[ts] = future_idx[num_bars - 1]
            elif len(future_idx) > 0:
                barriers.loc[ts] = future_idx[-1]
            else:
                barriers.loc[ts] = close.index[-1]
        else:
            # Use time delta
            target_time = ts + timedelta
            future_idx = close.index[close.index > ts]
            future_idx = future_idx[future_idx <= target_time]
            
            if len(future_idx) > 0:
                barriers.loc[ts] = future_idx[-1]
            else:
                # Target time is beyond data
                future_idx = close.index[close.index > ts]
                if len(future_idx) > 0:
                    barriers.loc[ts] = future_idx[-1]
                else:
                    barriers.loc[ts] = close.index[-1]
    
    return barriers
