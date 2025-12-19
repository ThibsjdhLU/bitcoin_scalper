"""
Unit tests for Triple Barrier Method implementation.

These tests validate the core labeling logic, particularly focusing on:
- Correct barrier touch detection
- Edge cases and off-by-one errors
- Handling of various market scenarios
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.bitcoin_scalper.labeling.barriers import (
    apply_triple_barrier,
    get_events,
    get_vertical_barriers
)
from src.bitcoin_scalper.labeling.labels import (
    generate_labels_from_barriers,
    get_labels,
    get_meta_labels
)


class TestTripleBarrier:
    """Test suite for Triple Barrier Method."""
    
    @pytest.fixture
    def simple_price_series(self):
        """Create a simple price series for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        # Create price series with known pattern
        prices = pd.Series(100.0, index=dates)
        return prices
    
    @pytest.fixture
    def uptrend_price_series(self):
        """Create an uptrending price series."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        # Steady uptrend
        prices = pd.Series(100 + np.arange(100) * 0.1, index=dates)
        return prices
    
    @pytest.fixture
    def downtrend_price_series(self):
        """Create a downtrending price series."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        # Steady downtrend
        prices = pd.Series(100 - np.arange(100) * 0.1, index=dates)
        return prices
    
    @pytest.fixture
    def volatile_price_series(self):
        """Create a volatile price series."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        # Random walk with drift
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02  # 2% volatility
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        return prices
    
    def test_profit_target_hit(self, uptrend_price_series):
        """Test that profit target is correctly detected."""
        close = uptrend_price_series
        events = pd.DatetimeIndex([close.index[0]])
        
        # Set 2% profit target - should be hit quickly in uptrend
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events)
        )
        
        assert len(result) == 1
        assert result.loc[events[0], 'type'] == 1  # Profit target
        assert result.loc[events[0], 'return'] > 0  # Positive return
    
    def test_stop_loss_hit(self, downtrend_price_series):
        """Test that stop loss is correctly detected."""
        close = downtrend_price_series
        events = pd.DatetimeIndex([close.index[0]])
        
        # Set 2% stop loss - should be hit quickly in downtrend
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events)
        )
        
        assert len(result) == 1
        assert result.loc[events[0], 'type'] == -1  # Stop loss
        assert result.loc[events[0], 'return'] < 0  # Negative return
    
    def test_vertical_barrier_hit(self, simple_price_series):
        """Test that vertical barrier is hit when price doesn't move."""
        close = simple_price_series  # Flat price
        events = pd.DatetimeIndex([close.index[0]])
        
        # Set small vertical barrier
        vertical_barrier = pd.Series(close.index[10], index=events)
        
        # With 2% barriers and flat price, should hit vertical
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=vertical_barrier
        )
        
        assert len(result) == 1
        assert result.loc[events[0], 'type'] == 0  # Vertical barrier
        assert abs(result.loc[events[0], 'return']) < 0.001  # Near zero return
    
    def test_barrier_levels(self, simple_price_series):
        """Test that barrier levels are correctly calculated."""
        close = simple_price_series
        events = pd.DatetimeIndex([close.index[0]])
        
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events)
        )
        
        # Check barrier levels
        assert result.loc[events[0], 'pt'] == pytest.approx(102.0, rel=1e-5)  # 100 * 1.02
        assert result.loc[events[0], 'sl'] == pytest.approx(98.0, rel=1e-5)   # 100 * 0.98
    
    def test_multiple_events(self, volatile_price_series):
        """Test processing multiple events."""
        close = volatile_price_series
        # Select multiple event times
        events = pd.DatetimeIndex([
            close.index[0],
            close.index[25],
            close.index[50],
            close.index[75]
        ])
        
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events)
        )
        
        assert len(result) == 4
        # Check all events have valid barrier types
        assert result['type'].isin([-1, 0, 1]).all()
        # Check returns are computed
        assert result['return'].notna().all()
    
    def test_side_long(self, uptrend_price_series):
        """Test long position (side=1)."""
        close = uptrend_price_series
        events = pd.DatetimeIndex([close.index[0]])
        side = pd.Series(1, index=events)
        
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events),
            side=side
        )
        
        # In uptrend, long should hit profit target
        assert result.loc[events[0], 'type'] == 1
    
    def test_side_short(self, downtrend_price_series):
        """Test short position (side=-1)."""
        close = downtrend_price_series
        events = pd.DatetimeIndex([close.index[0]])
        side = pd.Series(-1, index=events)
        
        result = apply_triple_barrier(
            close=close,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(close.index[-1], index=events),
            side=side
        )
        
        # In downtrend, short should hit profit target (inverted barriers)
        assert result.loc[events[0], 'type'] == 1
        # Return should be positive for short in downtrend
        assert result.loc[events[0], 'return'] > 0
    
    def test_get_events_basic(self, volatile_price_series):
        """Test get_events function with basic parameters."""
        close = volatile_price_series
        timestamps = pd.DatetimeIndex([
            close.index[0],
            close.index[30],
            close.index[60]
        ])
        
        events = get_events(
            close=close,
            timestamps=timestamps,
            pt_sl=0.02,
            max_holding_period=pd.Timedelta('15min')
        )
        
        assert len(events) == 3
        assert 'type' in events.columns
        assert 'return' in events.columns
        assert 't1' in events.columns
    
    def test_get_events_min_return_filter(self, volatile_price_series):
        """Test that min_return filters out small moves."""
        close = volatile_price_series
        timestamps = pd.DatetimeIndex([
            close.index[i] for i in range(0, 90, 10)
        ])
        
        # Without filter
        events_all = get_events(
            close=close,
            timestamps=timestamps,
            pt_sl=0.02,
            max_holding_period=pd.Timedelta('5min'),
            min_return=0.0
        )
        
        # With filter
        events_filtered = get_events(
            close=close,
            timestamps=timestamps,
            pt_sl=0.02,
            max_holding_period=pd.Timedelta('5min'),
            min_return=0.01  # 1% minimum
        )
        
        # Filtered should have fewer or equal events
        assert len(events_filtered) <= len(events_all)
    
    def test_vertical_barriers_num_bars(self, simple_price_series):
        """Test vertical barrier calculation with number of bars."""
        close = simple_price_series
        timestamps = pd.DatetimeIndex([close.index[0], close.index[20]])
        
        barriers = get_vertical_barriers(
            timestamps=timestamps,
            close=close,
            num_bars=10
        )
        
        assert len(barriers) == 2
        # First barrier should be 10 bars after first timestamp
        assert barriers.iloc[0] == close.index[10]
    
    def test_vertical_barriers_timedelta(self, simple_price_series):
        """Test vertical barrier calculation with timedelta."""
        close = simple_price_series
        timestamps = pd.DatetimeIndex([close.index[0]])
        
        barriers = get_vertical_barriers(
            timestamps=timestamps,
            close=close,
            timedelta=pd.Timedelta('10min')
        )
        
        assert len(barriers) == 1
        # Barrier should be approximately 10 minutes after start
        time_diff = barriers.iloc[0] - timestamps[0]
        assert time_diff <= pd.Timedelta('10min')
    
    def test_off_by_one_exact_hit(self):
        """Test off-by-one errors with exact barrier hits."""
        # Create price series that hits barrier exactly
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series([100, 101, 102, 103, 102, 101, 100, 99, 98, 97], index=dates)
        
        events = pd.DatetimeIndex([dates[0]])
        
        # 2% barrier should be hit at 102 (index 2)
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(dates[-1], index=events)
        )
        
        # Should detect profit target hit
        assert result.loc[events[0], 'type'] == 1
        # Should be at correct timestamp
        assert result.loc[events[0], 't1'] == dates[2]
    
    def test_first_barrier_priority(self):
        """Test that first barrier touched takes precedence."""
        # Create price that would hit stop loss before profit target
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series([100, 99, 98, 97, 103, 104, 105, 106, 107, 108], index=dates)
        
        events = pd.DatetimeIndex([dates[0]])
        
        # 2% barriers
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(dates[-1], index=events)
        )
        
        # Stop loss at 98 (index 2) comes before profit target at 102
        assert result.loc[events[0], 'type'] == -1  # Stop loss
        assert result.loc[events[0], 't1'] == dates[2]


class TestLabels:
    """Test suite for label generation."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events DataFrame."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        events = pd.DataFrame({
            'type': [1, -1, 0, 1, 1, -1, 0, -1, 1, 0],
            'return': [0.02, -0.02, 0.001, 0.025, 0.015, -0.018, -0.002, -0.022, 0.03, 0.0005],
            't1': dates,
            'pt': [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],
            'sl': [98, 98, 98, 98, 98, 98, 98, 98, 98, 98]
        }, index=dates)
        return events
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series(100 + np.arange(10), index=dates)
        return prices
    
    def test_generate_labels_from_barriers(self, sample_events, sample_prices):
        """Test basic label generation from barriers."""
        labels = generate_labels_from_barriers(sample_events, sample_prices)
        
        assert len(labels) == len(sample_events)
        # Labels should match types
        assert (labels == sample_events['type']).all()
    
    def test_get_labels_fixed(self, sample_events, sample_prices):
        """Test fixed label generation."""
        labels = get_labels(sample_events, sample_prices, label_type='fixed')
        
        assert len(labels) == len(sample_events)
        assert labels.isin([-1, 0, 1]).all()
    
    def test_get_labels_sign(self, sample_events, sample_prices):
        """Test sign-based label generation."""
        labels = get_labels(sample_events, sample_prices, label_type='sign')
        
        assert len(labels) == len(sample_events)
        # Should match sign of returns
        expected = np.sign(sample_events['return'])
        assert (labels == expected).all()
    
    def test_get_labels_binary(self, sample_events, sample_prices):
        """Test binary label generation (no neutrals)."""
        labels = get_labels(sample_events, sample_prices, label_type='binary')
        
        # Should have fewer labels (neutrals removed)
        assert len(labels) < len(sample_events)
        # Should only have -1 and 1
        assert labels.isin([-1, 1]).all()
        assert 0 not in labels.values
    
    def test_get_labels_threshold(self, sample_events, sample_prices):
        """Test threshold-based label generation."""
        labels = get_labels(
            sample_events, 
            sample_prices, 
            label_type='threshold',
            return_threshold=0.015
        )
        
        # Check that small returns are labeled as 0
        small_return_idx = sample_events[sample_events['return'].abs() < 0.015].index
        assert (labels.loc[small_return_idx] == 0).all()
        
        # Check that large positive returns are labeled as 1
        large_pos_idx = sample_events[sample_events['return'] > 0.015].index
        if len(large_pos_idx) > 0:
            assert (labels.loc[large_pos_idx] == 1).all()
    
    def test_get_meta_labels_basic(self, sample_events, sample_prices):
        """Test basic meta-label generation."""
        meta_labels = get_meta_labels(sample_events, sample_prices)
        
        assert len(meta_labels) == len(sample_events)
        # Should be binary (0 or 1)
        assert meta_labels.isin([0, 1]).all()
        # Successful trades should be labeled 1
        assert (meta_labels == (sample_events['return'] > 0).astype(int)).all()
    
    def test_get_meta_labels_with_predictions(self, sample_events, sample_prices):
        """Test meta-label generation with primary model predictions."""
        # Create sample predictions
        predictions = pd.Series([1, 1, -1, 1, 1, -1, -1, 1, 1, -1], index=sample_events.index)
        
        meta_labels = get_meta_labels(
            sample_events, 
            sample_prices, 
            primary_model_predictions=predictions,
            side_from_predictions=True
        )
        
        assert len(meta_labels) == len(sample_events)
        assert meta_labels.isin([0, 1]).all()
        
        # Check logic: if prediction matches return direction, label is 1
        adjusted_return = sample_events['return'] * predictions
        expected = (adjusted_return > 0).astype(int)
        assert (meta_labels == expected).all()
    
    def test_label_distribution(self, sample_events, sample_prices):
        """Test that label distribution is reasonable."""
        labels = get_labels(sample_events, sample_prices, label_type='fixed')
        
        # Should have some of each label type in our sample data
        unique_labels = labels.unique()
        assert len(unique_labels) >= 2  # At least 2 different labels


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_events(self):
        """Test handling of empty events."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series(100.0, index=dates)
        events = pd.DatetimeIndex([])
        
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.02,
            vertical_barrier=None
        )
        
        assert len(result) == 0
    
    def test_single_event(self):
        """Test handling of single event."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series(100.0, index=dates)
        events = pd.DatetimeIndex([dates[0]])
        
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.02,
            vertical_barrier=pd.Series(dates[-1], index=events)
        )
        
        assert len(result) == 1
    
    def test_event_beyond_data(self):
        """Test event timestamp beyond available data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1min')
        prices = pd.Series(100.0, index=dates)
        # Event at last timestamp - no future data
        events = pd.DatetimeIndex([dates[-1]])
        
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.02,
            vertical_barrier=None
        )
        
        # Should handle gracefully (may return NaN or empty)
        assert len(result) >= 0
    
    def test_very_small_barriers(self):
        """Test with very small barrier thresholds."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100) * 0.1, index=dates)
        events = pd.DatetimeIndex([dates[0]])
        
        # Very small barrier - should be hit quickly
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.0001,  # 0.01%
            vertical_barrier=pd.Series(dates[-1], index=events)
        )
        
        assert len(result) == 1
        # Should hit a barrier (not timeout)
        assert result.loc[events[0], 'type'] in [-1, 1]
    
    def test_very_large_barriers(self):
        """Test with very large barrier thresholds."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1min')
        prices = pd.Series(100 + np.arange(20) * 0.01, index=dates)
        events = pd.DatetimeIndex([dates[0]])
        
        # Very large barriers - should hit vertical
        result = apply_triple_barrier(
            close=prices,
            events=events,
            pt_sl=0.5,  # 50% - unlikely to hit
            vertical_barrier=pd.Series(dates[10], index=events)
        )
        
        assert len(result) == 1
        # Should hit vertical barrier
        assert result.loc[events[0], 'type'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
