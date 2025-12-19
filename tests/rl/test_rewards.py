"""
Unit tests for reward functions.
"""

import pytest
import numpy as np

from src.bitcoin_scalper.rl.rewards import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    DifferentialSharpeRatio,
    calculate_differential_sharpe_ratio,
    calculate_step_penalty,
)


class TestSharpeRatio:
    """Test suite for Sharpe ratio calculation."""
    
    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0, "Sharpe should be positive for positive returns"
    
    def test_negative_returns(self):
        """Test Sharpe with negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.008, -0.012])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0, "Sharpe should be negative for negative returns"
    
    def test_zero_std(self):
        """Test Sharpe with zero standard deviation."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0, "Sharpe should be 0 when std is 0"
    
    def test_insufficient_data(self):
        """Test Sharpe with insufficient data."""
        returns = np.array([0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0, "Sharpe should be 0 with single return"
    
    def test_mixed_returns(self):
        """Test Sharpe with mixed positive/negative returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float), "Sharpe should return float"
    
    def test_annualization(self):
        """Test that annualization affects result."""
        returns = np.array([0.01, 0.02, 0.015, 0.008])
        sharpe_minute = calculate_sharpe_ratio(returns, periods_per_year=525600)
        sharpe_daily = calculate_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe_minute != sharpe_daily, "Different periods should give different results"


class TestSortinoRatio:
    """Test suite for Sortino ratio calculation."""
    
    def test_positive_returns(self):
        """Test Sortino with positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        sortino = calculate_sortino_ratio(returns)
        assert np.isinf(sortino) or sortino > 0, "Sortino should be positive/inf for all positive returns"
    
    def test_mixed_returns(self):
        """Test Sortino with mixed returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
        sortino = calculate_sortino_ratio(returns)
        assert isinstance(sortino, float), "Sortino should return float"
        assert sortino > 0, "Sortino should be positive for positive mean return"
    
    def test_downside_only(self):
        """Test that Sortino only penalizes downside."""
        # Two series with same mean but different upside volatility
        returns1 = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
        returns2 = np.array([0.05, -0.01, 0.001, -0.01, 0.001])
        
        sortino1 = calculate_sortino_ratio(returns1)
        sortino2 = calculate_sortino_ratio(returns2)
        
        # Both have similar downside, so similar Sortino
        assert abs(sortino1 - sortino2) < abs(
            calculate_sharpe_ratio(returns1) - calculate_sharpe_ratio(returns2)
        ), "Sortino should be less sensitive to upside volatility than Sharpe"
    
    def test_no_downside(self):
        """Test Sortino when no returns below target."""
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        sortino = calculate_sortino_ratio(returns, target_return=0.0)
        assert np.isinf(sortino), "Sortino should be infinite with no downside"
    
    def test_insufficient_data(self):
        """Test Sortino with insufficient data."""
        returns = np.array([0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 0.0, "Sortino should be 0 with single return"


class TestDifferentialSharpeRatio:
    """Test suite for Differential Sharpe Ratio."""
    
    def test_initialization(self):
        """Test DSR initialization."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        assert dsr.eta == 0.01
        assert dsr.A == 0.0
        assert dsr.B == 1.0
        assert dsr.t == 0
    
    def test_invalid_eta(self):
        """Test that invalid eta raises error."""
        with pytest.raises(ValueError):
            DifferentialSharpeRatio(eta=0.0)
        with pytest.raises(ValueError):
            DifferentialSharpeRatio(eta=1.0)
        with pytest.raises(ValueError):
            DifferentialSharpeRatio(eta=-0.1)
    
    def test_update(self):
        """Test DSR update with returns."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        
        # Update with positive return
        dsr_value = dsr.update(0.01)
        assert isinstance(dsr_value, float)
        assert dsr.t == 1
        
        # Update with negative return
        dsr_value = dsr.update(-0.005)
        assert isinstance(dsr_value, float)
        assert dsr.t == 2
    
    def test_reset(self):
        """Test DSR reset."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        
        # Update a few times
        for _ in range(10):
            dsr.update(np.random.randn() * 0.01)
        
        # Reset
        dsr.reset()
        assert dsr.A == 0.0
        assert dsr.B == 1.0
        assert dsr.t == 0
    
    def test_get_sharpe_estimate(self):
        """Test getting Sharpe estimate from DSR."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        
        # Update with positive returns
        for _ in range(100):
            dsr.update(0.01 + np.random.randn() * 0.002)
        
        sharpe_est = dsr.get_sharpe_estimate()
        assert sharpe_est > 0, "Should have positive Sharpe for positive returns"
        assert isinstance(sharpe_est, float)
    
    def test_clipping(self):
        """Test that DSR values are clipped."""
        dsr = DifferentialSharpeRatio(eta=0.01)
        
        # Update with extreme returns
        for _ in range(20):
            dsr.update(0.001)
        
        dsr_value = dsr.update(10.0)  # Extreme return
        assert -10.0 <= dsr_value <= 10.0, "DSR should be clipped to [-10, 10]"


class TestDifferentialSharpeRatioBatch:
    """Test batch DSR calculation."""
    
    def test_batch_calculation(self):
        """Test DSR calculation for return sequence."""
        returns = [0.01, -0.005, 0.02, 0.015, -0.003]
        dsr_values, final_sharpe = calculate_differential_sharpe_ratio(returns)
        
        assert len(dsr_values) == len(returns)
        assert isinstance(final_sharpe, float)
    
    def test_empty_returns(self):
        """Test batch DSR with empty returns."""
        returns = []
        dsr_values, final_sharpe = calculate_differential_sharpe_ratio(returns)
        
        assert len(dsr_values) == 0
        assert final_sharpe == 0.0


class TestStepPenalty:
    """Test suite for step penalty calculation."""
    
    def test_penalty_increases_with_duration(self):
        """Test that penalty increases with position duration."""
        penalty1 = calculate_step_penalty(10)
        penalty2 = calculate_step_penalty(50)
        penalty3 = calculate_step_penalty(100)
        
        # Penalties are negative, so more negative (smaller value) = larger penalty
        assert penalty1 > penalty2 > penalty3, "Penalty should be more negative for longer duration"
        assert penalty3 < penalty2 < penalty1, "Verify ordering: longer duration = more negative"
    
    def test_penalty_caps_at_max_duration(self):
        """Test that penalty caps at max duration."""
        penalty1 = calculate_step_penalty(100, max_duration=100)
        penalty2 = calculate_step_penalty(150, max_duration=100)
        
        assert penalty1 == penalty2, "Penalty should cap at max_duration"
    
    def test_penalty_is_negative(self):
        """Test that penalty is always negative."""
        penalty = calculate_step_penalty(50)
        assert penalty < 0, "Penalty should be negative"
    
    def test_zero_duration(self):
        """Test penalty with zero duration."""
        penalty = calculate_step_penalty(0)
        assert penalty == 0.0, "Penalty should be 0 for zero duration"
    
    def test_custom_penalty_rate(self):
        """Test custom penalty rate."""
        penalty1 = calculate_step_penalty(50, penalty_rate=0.0001)
        penalty2 = calculate_step_penalty(50, penalty_rate=0.001)
        
        assert penalty2 < penalty1, "Higher penalty rate should give more negative penalty"
