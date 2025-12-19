"""Tests for validation module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.bitcoin_scalper.validation.cross_val import PurgedKFold, CombinatorialPurgedCV
from src.bitcoin_scalper.validation.drift import ADWINDetector, DriftScanner
from src.bitcoin_scalper.validation.backtest import Backtester, BacktestResult, SignalType


class TestPurgedKFold:
    """Test PurgedKFold cross-validator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        
        X = np.random.randn(n_samples, 5)
        y = np.random.randint(0, 2, n_samples)
        
        # t1: event end times (use same as event time for simplicity)
        # In practice, these would be from Triple Barrier method
        t1 = pd.Series(dates, index=range(n_samples))
        
        return X, y, t1
    
    def test_initialization(self):
        """Test PurgedKFold initialization."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        assert cv.n_splits == 5
        assert cv.embargo_pct == 0.01
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)  # Too few splits
        
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=5, embargo_pct=1.5)  # Invalid embargo
    
    def test_split_generation(self, sample_data):
        """Test split generation."""
        X, y, t1 = sample_data
        cv = PurgedKFold(n_splits=3, embargo_pct=0.01)
        
        splits = list(cv.split(X, y, t1=t1))
        
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_purging_effect(self, sample_data):
        """Test that purging reduces training set size."""
        X, y, t1 = sample_data
        
        # Without purging (standard KFold behavior)
        cv_no_purge = PurgedKFold(n_splits=3, embargo_pct=0.0)
        
        # With purging
        cv_purge = PurgedKFold(n_splits=3, embargo_pct=0.1)
        
        splits_no_purge = list(cv_no_purge.split(X, y, t1=t1))
        splits_purge = list(cv_purge.split(X, y, t1=t1))
        
        # Purged splits should have smaller training sets
        for i in range(3):
            train_no_purge, _ = splits_no_purge[i]
            train_purge, _ = splits_purge[i]
            
            # With embargo, training set should be smaller or equal
            assert len(train_purge) <= len(train_no_purge)


class TestCombinatorialPurgedCV:
    """Test CombinatorialPurgedCV."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n_samples = 80
        X = np.random.randn(n_samples, 3)
        y = np.random.randint(0, 2, n_samples)
        t1 = pd.Series(range(n_samples), index=range(n_samples))
        return X, y, t1
    
    def test_initialization(self):
        """Test initialization."""
        cv = CombinatorialPurgedCV(n_splits=8, n_test_groups=2)
        assert cv.n_splits == 8
        assert cv.n_test_groups == 2
        
        # Check number of combinations: C(8, 2) = 28
        assert cv.n_combinations == 28
    
    def test_split_generation(self, sample_data):
        """Test split generation."""
        X, y, t1 = sample_data
        cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=1)
        
        splits = list(cv.split(X, y, t1=t1))
        
        # Should have C(4, 1) = 4 combinations
        assert len(splits) == 4
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestADWINDetector:
    """Test ADWIN drift detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = ADWINDetector(delta=0.002, max_window=1000)
        assert detector.delta == 0.002
        assert detector.max_window == 1000
        assert detector.n_detections == 0
    
    def test_stable_stream(self):
        """Test on stable data stream."""
        detector = ADWINDetector(delta=0.05)
        
        # Generate stable stream
        for _ in range(100):
            value = np.random.normal(0.1, 0.01)
            drift = detector.update(value)
            
            # Should not detect drift in stable stream
            # (though probabilistic, so very unlikely but possible)
        
        # Most likely no drift
        assert detector.n_detections <= 1
    
    def test_drift_detection(self):
        """Test drift detection on changing stream."""
        detector = ADWINDetector(delta=0.002)
        
        # Stable period
        for _ in range(50):
            detector.update(np.random.normal(0.1, 0.01))
        
        # Change in distribution (drift)
        drift_detected = False
        for _ in range(50):
            value = np.random.normal(0.5, 0.05)  # Much different
            if detector.update(value):
                drift_detected = True
                break
        
        # Should detect drift
        assert drift_detected or detector.n_detections > 0
    
    def test_reset(self):
        """Test detector reset."""
        detector = ADWINDetector()
        
        for _ in range(20):
            detector.update(np.random.randn())
        
        assert detector.get_window_size() > 0
        
        detector.reset()
        
        assert detector.get_window_size() == 0
        assert detector.n_detections == 0


class TestDriftScanner:
    """Test DriftScanner."""
    
    def test_initialization(self):
        """Test scanner initialization."""
        scanner = DriftScanner(delta=0.002, use_river=False)
        assert scanner.delta == 0.002
        assert len(scanner.history) == 0
    
    def test_scan(self):
        """Test scanning for drift."""
        scanner = DriftScanner(delta=0.05, use_river=False)
        
        # Scan some values
        for i in range(30):
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
            scanner.scan(0.1, timestamp)
        
        # Check history (may or may not have drift)
        history = scanner.get_drift_history()
        assert isinstance(history, pd.DataFrame)


class TestBacktester:
    """Test Backtester."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        # Random walk price
        returns = np.random.randn(100) * 0.01
        prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
        
        # Simple signals: buy when price below MA, sell when above
        ma = prices.rolling(10).mean()
        signals = pd.Series(0, index=dates)
        signals[prices < ma] = 1  # Long
        signals[prices > ma] = -1  # Short
        
        return prices, signals
    
    def test_initialization(self):
        """Test backtester initialization."""
        bt = Backtester(
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        assert bt.initial_capital == 10000
        assert bt.capital == 10000
        assert bt.position.is_flat()
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            Backtester(initial_capital=0)
        
        with pytest.raises(ValueError):
            Backtester(initial_capital=1000, commission_pct=-0.1)
    
    def test_run_backtest(self, sample_market_data):
        """Test running a backtest."""
        prices, signals = sample_market_data
        
        bt = Backtester(
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        result = bt.run(prices, signals)
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0
        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics
    
    def test_result_summary(self, sample_market_data):
        """Test result summary generation."""
        prices, signals = sample_market_data
        
        bt = Backtester(initial_capital=10000)
        result = bt.run(prices, signals)
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert 'BACKTEST RESULTS' in summary
        assert 'Sharpe Ratio' in summary
    
    def test_reset(self):
        """Test backtester reset."""
        bt = Backtester(initial_capital=10000)
        
        # Simulate some activity
        bt.capital = 8000
        bt.trades.append(None)
        
        bt.reset()
        
        assert bt.capital == 10000
        assert len(bt.trades) == 0
        assert bt.position.is_flat()


class TestBacktestResult:
    """Test BacktestResult."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample result."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1D')
        equity = pd.Series(10000 * (1 + np.random.randn(50) * 0.01).cumprod(), index=dates)
        returns = equity.pct_change().dropna()
        
        return BacktestResult(
            equity_curve=equity,
            trades=[],
            returns=returns
        )
    
    def test_metrics_calculation(self, sample_result):
        """Test automatic metrics calculation."""
        assert 'sharpe_ratio' in sample_result.metrics
        assert 'sortino_ratio' in sample_result.metrics
        assert 'max_drawdown' in sample_result.metrics
        assert 'total_return' in sample_result.metrics
    
    def test_summary_generation(self, sample_result):
        """Test summary string generation."""
        summary = sample_result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
