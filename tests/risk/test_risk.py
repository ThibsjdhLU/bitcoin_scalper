"""Tests for risk management module."""

import pytest
import numpy as np
import pandas as pd

from src.bitcoin_scalper.risk.sizing import KellySizer, TargetVolatilitySizer


class TestKellySizer:
    """Test Kelly Criterion position sizer."""
    
    def test_initialization(self):
        """Test sizer initialization."""
        sizer = KellySizer(kelly_fraction=0.5, max_leverage=1.0)
        assert sizer.kelly_fraction == 0.5
        assert sizer.max_leverage == 1.0
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=0)  # Must be > 0
        
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=1.5)  # Must be <= 1
        
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=0.5, max_leverage=0)  # Must be > 0
    
    def test_positive_edge_sizing(self):
        """Test sizing with positive edge."""
        sizer = KellySizer(kelly_fraction=1.0, max_leverage=2.0)
        
        # 60% win rate, 2:1 payoff ratio
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            win_prob=0.60,
            payoff_ratio=2.0
        )
        
        # Should return positive size
        assert size > 0
        
        # Kelly formula: f* = 0.6 - 0.4/2 = 0.6 - 0.2 = 0.4
        # So 40% of capital = $4000 / $50000 = 0.08 BTC
        expected_size = (10000 * 0.4) / 50000
        assert abs(size - expected_size) < 1e-6
    
    def test_no_edge_sizing(self):
        """Test sizing with no edge (should return 0)."""
        sizer = KellySizer(kelly_fraction=1.0)
        
        # 50% win rate, 1:1 payoff (no edge)
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            win_prob=0.50,
            payoff_ratio=1.0
        )
        
        # Kelly formula: f* = 0.5 - 0.5/1 = 0
        assert size == 0.0
    
    def test_negative_edge_sizing(self):
        """Test sizing with negative edge (should return 0)."""
        sizer = KellySizer(kelly_fraction=1.0)
        
        # 40% win rate, 1:1 payoff (negative edge)
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            win_prob=0.40,
            payoff_ratio=1.0
        )
        
        # Kelly formula: f* = 0.4 - 0.6/1 = -0.2 (negative)
        # Should return 0
        assert size == 0.0
    
    def test_fractional_kelly(self):
        """Test fractional Kelly reduces position size."""
        full_kelly = KellySizer(kelly_fraction=1.0)
        half_kelly = KellySizer(kelly_fraction=0.5)
        
        params = {
            'capital': 10000,
            'price': 50000,
            'win_prob': 0.60,
            'payoff_ratio': 2.0
        }
        
        full_size = full_kelly.calculate_size(**params)
        half_size = half_kelly.calculate_size(**params)
        
        # Half Kelly should be half of full Kelly
        assert abs(half_size - full_size * 0.5) < 1e-6
    
    def test_max_leverage_cap(self):
        """Test max leverage caps position size."""
        sizer = KellySizer(kelly_fraction=1.0, max_leverage=0.5)
        
        # Very favorable odds would normally give large position
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            win_prob=0.80,
            payoff_ratio=3.0
        )
        
        # Position should be capped at max_leverage
        # Max position value = 10000 * 0.5 = 5000
        # Max size = 5000 / 50000 = 0.1
        expected_max = (10000 * 0.5) / 50000
        assert size <= expected_max + 1e-6
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        sizer = KellySizer(kelly_fraction=0.5)
        
        # Invalid capital
        assert sizer.calculate_size(
            capital=0, price=50000, win_prob=0.6, payoff_ratio=2.0
        ) == 0.0
        
        # Invalid price
        assert sizer.calculate_size(
            capital=10000, price=0, win_prob=0.6, payoff_ratio=2.0
        ) == 0.0
        
        # Invalid win_prob
        assert sizer.calculate_size(
            capital=10000, price=50000, win_prob=1.5, payoff_ratio=2.0
        ) == 0.0
        
        # Invalid payoff_ratio
        assert sizer.calculate_size(
            capital=10000, price=50000, win_prob=0.6, payoff_ratio=-1.0
        ) == 0.0
    
    def test_model_confidence_interface(self):
        """Test convenience method for model confidence."""
        sizer = KellySizer(kelly_fraction=0.5)
        
        size = sizer.calculate_from_model_confidence(
            capital=10000,
            price=50000,
            confidence=0.70,
            expected_return=0.03,
            stop_loss_pct=0.02
        )
        
        # Should return positive size
        assert size > 0


class TestTargetVolatilitySizer:
    """Test Target Volatility position sizer."""
    
    def test_initialization(self):
        """Test sizer initialization."""
        sizer = TargetVolatilitySizer(target_volatility=0.40, max_leverage=1.0)
        assert sizer.target_volatility == 0.40
        assert sizer.max_leverage == 1.0
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            TargetVolatilitySizer(target_volatility=0)
        
        with pytest.raises(ValueError):
            TargetVolatilitySizer(target_volatility=0.4, max_leverage=0)
    
    def test_high_volatility_reduces_size(self):
        """Test that high volatility reduces position size."""
        sizer = TargetVolatilitySizer(target_volatility=0.40, max_leverage=2.0)
        
        # Low volatility case
        low_vol_size = sizer.calculate_size(
            capital=10000,
            price=50000,
            asset_volatility=0.40  # Matches target
        )
        
        # High volatility case
        high_vol_size = sizer.calculate_size(
            capital=10000,
            price=50000,
            asset_volatility=0.80  # Double the target
        )
        
        # Higher volatility should give smaller position
        assert high_vol_size < low_vol_size
        
        # Specifically, should be half
        assert abs(high_vol_size - low_vol_size * 0.5) < 1e-6
    
    def test_target_volatility_matching(self):
        """Test position when asset vol matches target vol."""
        sizer = TargetVolatilitySizer(target_volatility=0.40, max_leverage=2.0)
        
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            asset_volatility=0.40
        )
        
        # When asset vol = target vol, position fraction = 1.0
        # Position value = 10000, size = 10000/50000 = 0.2
        expected_size = 10000 / 50000
        assert abs(size - expected_size) < 1e-6
    
    def test_max_leverage_cap(self):
        """Test max leverage caps position size."""
        sizer = TargetVolatilitySizer(target_volatility=0.80, max_leverage=0.5)
        
        # Very low volatility would normally give large position
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
            asset_volatility=0.10  # Very low
        )
        
        # Should be capped at max_leverage
        expected_max = (10000 * 0.5) / 50000
        assert size <= expected_max + 1e-6
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        sizer = TargetVolatilitySizer(target_volatility=0.40)
        
        # Invalid capital
        assert sizer.calculate_size(
            capital=0, price=50000, asset_volatility=0.60
        ) == 0.0
        
        # Invalid price
        assert sizer.calculate_size(
            capital=10000, price=0, asset_volatility=0.60
        ) == 0.0
        
        # Invalid volatility
        assert sizer.calculate_size(
            capital=10000, price=50000, asset_volatility=0
        ) == 0.0
    
    def test_volatility_estimation(self):
        """Test volatility estimation from returns."""
        # Create sample returns series
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1D')
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        
        vol = TargetVolatilitySizer.estimate_volatility(
            returns, window=20, annualization_factor=365
        )
        
        # Should return positive volatility
        assert vol > 0
        
        # Should be reasonable (not extreme)
        assert 0.01 < vol < 5.0
    
    def test_volatility_estimation_insufficient_data(self):
        """Test volatility estimation with insufficient data."""
        # Too few returns
        returns = pd.Series([0.01, 0.02])
        
        vol = TargetVolatilitySizer.estimate_volatility(
            returns, window=20, min_periods=10
        )
        
        # Should return 0 or very small value
        assert vol == 0.0
