"""
Unit tests for volatility estimation.

Tests the EWMA volatility estimation functions.
"""

import pytest
import numpy as np
import pandas as pd

from src.bitcoin_scalper.labeling.volatility import (
    calculate_daily_volatility,
    estimate_ewma_volatility,
    get_adaptive_span
)


class TestVolatility:
    """Test suite for volatility estimation."""
    
    @pytest.fixture
    def constant_price_series(self):
        """Create a constant price series (zero volatility)."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        prices = pd.Series(100.0, index=dates)
        return prices
    
    @pytest.fixture
    def volatile_price_series(self):
        """Create a volatile price series."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02  # 2% volatility
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        return prices
    
    @pytest.fixture
    def trending_price_series(self):
        """Create a trending price series."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        prices = pd.Series(100 + np.arange(200) * 0.1, index=dates)
        return prices
    
    def test_calculate_daily_volatility_basic(self, volatile_price_series):
        """Test basic volatility calculation."""
        vol = calculate_daily_volatility(volatile_price_series, span=100)
        
        assert len(vol) == len(volatile_price_series)
        # First value should be NaN (need previous for return)
        assert pd.isna(vol.iloc[0])
        # Subsequent values should be valid
        assert vol.iloc[-1] > 0
        assert not np.isinf(vol.iloc[-1])
    
    def test_calculate_daily_volatility_zero(self, constant_price_series):
        """Test volatility calculation with constant prices."""
        vol = calculate_daily_volatility(constant_price_series, span=100)
        
        # After warmup period, volatility should be very close to zero
        assert vol.iloc[-1] == pytest.approx(0, abs=1e-10)
    
    def test_calculate_daily_volatility_span_effect(self, volatile_price_series):
        """Test that different spans produce different volatility estimates."""
        vol_fast = calculate_daily_volatility(volatile_price_series, span=20)
        vol_slow = calculate_daily_volatility(volatile_price_series, span=100)
        
        # Fast should have more variation (after dropping NaN)
        assert vol_fast.dropna().std() > vol_slow.dropna().std()
    
    def test_estimate_ewma_volatility_series(self, volatile_price_series):
        """Test volatility estimation with Series input."""
        vol = estimate_ewma_volatility(volatile_price_series, span=100)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(volatile_price_series)
        assert vol.iloc[-1] > 0
    
    def test_estimate_ewma_volatility_dataframe(self, volatile_price_series):
        """Test volatility estimation with DataFrame input."""
        df = pd.DataFrame({'close': volatile_price_series})
        
        vol = estimate_ewma_volatility(df, span=100, price_col='close')
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(df)
    
    def test_estimate_ewma_volatility_autodetect(self, volatile_price_series):
        """Test automatic column detection."""
        # Test with common column names
        for col_name in ['close', '<CLOSE>', '1min_close', '1min_<CLOSE>']:
            df = pd.DataFrame({col_name: volatile_price_series})
            vol = estimate_ewma_volatility(df, span=100)
            assert len(vol) == len(df)
    
    def test_estimate_ewma_volatility_missing_column(self, volatile_price_series):
        """Test error handling for missing column."""
        df = pd.DataFrame({'price': volatile_price_series})
        
        with pytest.raises(ValueError, match="Could not find price column"):
            estimate_ewma_volatility(df, span=100)
    
    def test_estimate_ewma_volatility_empty(self):
        """Test error handling for empty series."""
        empty_series = pd.Series([], dtype=float)
        
        with pytest.raises(ValueError, match="Input prices are empty"):
            estimate_ewma_volatility(empty_series, span=100)
    
    def test_estimate_ewma_volatility_insufficient_data(self):
        """Test volatility calculation with very little data."""
        dates = pd.date_range('2024-01-01', periods=5, freq='1min')
        prices = pd.Series([100, 101, 99, 102, 98], index=dates)
        
        vol = estimate_ewma_volatility(prices, span=100)
        
        # Should still return a series
        assert len(vol) == len(prices)
        # Most values will be NaN due to insufficient warmup
        assert vol.isna().sum() > 0
    
    def test_get_adaptive_span_regimes(self):
        """Test adaptive span recommendations for different regimes."""
        span_volatile = get_adaptive_span(market_regime='volatile', frequency='1min')
        span_normal = get_adaptive_span(market_regime='normal', frequency='1min')
        span_stable = get_adaptive_span(market_regime='stable', frequency='1min')
        
        # Volatile should have shortest span
        assert span_volatile < span_normal < span_stable
    
    def test_get_adaptive_span_frequencies(self):
        """Test adaptive span recommendations for different frequencies."""
        span_1min = get_adaptive_span(market_regime='normal', frequency='1min')
        span_5min = get_adaptive_span(market_regime='normal', frequency='5min')
        span_1h = get_adaptive_span(market_regime='normal', frequency='1h')
        
        # Higher frequency should have longer span (more observations)
        assert span_1min > span_5min > span_1h
    
    def test_get_adaptive_span_minimum(self):
        """Test that adaptive span has a minimum value."""
        # Even for very low frequency and stable regime, should have min span
        span = get_adaptive_span(market_regime='stable', frequency='1d')
        assert span >= 10
    
    def test_volatility_positive(self, volatile_price_series):
        """Test that volatility estimates are always non-negative."""
        vol = calculate_daily_volatility(volatile_price_series, span=100)
        
        # Remove NaN values
        vol_valid = vol.dropna()
        
        # All values should be >= 0
        assert (vol_valid >= 0).all()
    
    def test_volatility_adaptation(self):
        """Test that volatility adapts to changing conditions."""
        # Create price series with regime change
        dates = pd.date_range('2024-01-01', periods=300, freq='1min')
        np.random.seed(42)
        
        # First half: low volatility
        returns1 = np.random.randn(150) * 0.005
        # Second half: high volatility
        returns2 = np.random.randn(150) * 0.03
        
        returns = np.concatenate([returns1, returns2])
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        vol = calculate_daily_volatility(prices, span=50)
        
        # Volatility in second half should be higher
        vol_first_half = vol.iloc[100:150].mean()
        vol_second_half = vol.iloc[250:].mean()
        
        assert vol_second_half > vol_first_half


class TestVolatilityIntegration:
    """Integration tests for volatility in barrier calculations."""
    
    def test_volatility_based_barriers(self):
        """Test using volatility for dynamic barrier sizing."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        # Calculate volatility
        vol = estimate_ewma_volatility(prices, span=100)
        
        # Use 2-sigma barriers
        pt_sl = 2.0 * vol
        
        # Barriers should adapt to volatility (after dropping NaN)
        pt_sl_valid = pt_sl.dropna()
        assert pt_sl_valid.std() > 0  # Should vary
        assert (pt_sl_valid > 0).all()  # Should be positive
        
        # Select a few events
        event_times = pd.DatetimeIndex([dates[100], dates[150]])
        barriers = pt_sl.loc[event_times]
        
        # Should have valid barriers
        assert len(barriers) == 2
        assert barriers.notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
