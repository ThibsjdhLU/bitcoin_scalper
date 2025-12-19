"""
Unit tests for validation wrapper.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.bitcoin_scalper.rl.env import TradingEnv
from src.bitcoin_scalper.rl.validation import ValidationWrapper, validate_agent


class DummyAgent:
    """Dummy agent for testing validation without SB3."""
    
    def __init__(self, strategy='random'):
        """
        Initialize dummy agent.
        
        Args:
            strategy: 'random', 'always_hold', 'always_buy', 'always_sell'
        """
        self.strategy = strategy
    
    def predict(self, obs, deterministic=True):
        """Predict action."""
        if self.strategy == 'random':
            action = np.random.randint(0, 3)
        elif self.strategy == 'always_hold':
            action = 0
        elif self.strategy == 'always_buy':
            action = 1
        elif self.strategy == 'always_sell':
            action = 2
        else:
            action = 0
        
        return action, None


class TestValidationWrapper:
    """Test suite for ValidationWrapper."""
    
    @pytest.fixture
    def sample_env(self):
        """Create a sample trading environment."""
        np.random.seed(42)
        
        # Generate sample data
        prices = [10000.0]
        for _ in range(200):
            change = np.random.randn() * 50
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.rand(len(prices)) * 1000,
        })
        
        return TradingEnv(df, window_size=30, reward_mode='sortino')
    
    @pytest.fixture
    def dummy_agent(self):
        """Create a dummy agent."""
        return DummyAgent(strategy='random')
    
    def test_initialization(self, sample_env, dummy_agent):
        """Test wrapper initialization."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        
        assert validator.env is sample_env
        assert validator.agent is dummy_agent
        assert validator.trade_log is None
        assert validator.episode_metrics == {}
    
    def test_run_single_episode(self, sample_env, dummy_agent):
        """Test running a single validation episode."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        metrics = validator.run_validation(n_episodes=1, render=False)
        
        assert 'mean_return' in metrics
        assert 'mean_sharpe' in metrics
        assert 'mean_sortino' in metrics
        assert 'win_rate' in metrics
        assert metrics['n_episodes'] == 1
    
    def test_run_multiple_episodes(self, sample_env, dummy_agent):
        """Test running multiple episodes."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        metrics = validator.run_validation(n_episodes=3, render=False)
        
        assert metrics['n_episodes'] == 3
        assert 'all_episodes' in metrics
        assert len(metrics['all_episodes']) == 3
    
    def test_trade_logging(self, sample_env, dummy_agent):
        """Test that trades are logged."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        metrics = validator.run_validation(n_episodes=1, log_trades=True)
        
        trade_log = validator.get_trade_log()
        assert trade_log is not None
        assert len(trade_log) > 0
        
        # Check columns
        expected_cols = ['episode', 'step', 'action', 'price', 
                        'position', 'balance', 'equity', 'reward']
        for col in expected_cols:
            assert col in trade_log.columns
    
    def test_no_trade_logging(self, sample_env, dummy_agent):
        """Test validation without trade logging."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        metrics = validator.run_validation(n_episodes=1, log_trades=False)
        
        trade_log = validator.get_trade_log()
        assert trade_log is None
    
    def test_get_metrics(self, sample_env, dummy_agent):
        """Test getting metrics."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        validator.run_validation(n_episodes=2)
        
        metrics = validator.get_metrics()
        assert 'mean_return' in metrics
        assert 'mean_sharpe' in metrics
    
    def test_save_results(self, sample_env, dummy_agent):
        """Test saving validation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'validation_results'
            
            validator = ValidationWrapper(sample_env, dummy_agent)
            validator.run_validation(n_episodes=2, log_trades=True)
            validator.save_results(output_dir)
            
            # Check files exist
            assert (output_dir / 'metrics.json').exists()
            assert (output_dir / 'trade_log.csv').exists()
    
    def test_compare_with_buy_and_hold(self, sample_env, dummy_agent):
        """Test comparison with buy-and-hold baseline."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        validator.run_validation(n_episodes=1)
        
        comparison = validator.compare_with_baseline('buy_and_hold')
        
        assert 'baseline_strategy' in comparison
        assert 'agent_return' in comparison
        assert 'baseline_return' in comparison
        assert 'outperformance' in comparison
        assert 'sharpe_improvement' in comparison
    
    def test_compare_without_validation(self, sample_env, dummy_agent):
        """Test that comparison without validation raises error."""
        validator = ValidationWrapper(sample_env, dummy_agent)
        
        with pytest.raises(ValueError):
            validator.compare_with_baseline('buy_and_hold')
    
    def test_different_agent_strategies(self, sample_env):
        """Test validation with different agent strategies."""
        strategies = ['always_hold', 'always_buy', 'random']
        
        for strategy in strategies:
            agent = DummyAgent(strategy=strategy)
            validator = ValidationWrapper(sample_env, agent)
            
            metrics = validator.run_validation(n_episodes=1, render=False)
            assert isinstance(metrics['mean_return'], float)
    
    def test_deterministic_flag(self, sample_env):
        """Test deterministic vs non-deterministic prediction."""
        agent = DummyAgent(strategy='random')
        
        # Deterministic
        validator_det = ValidationWrapper(sample_env, agent, deterministic=True)
        assert validator_det.deterministic is True
        
        # Non-deterministic
        validator_nondet = ValidationWrapper(sample_env, agent, deterministic=False)
        assert validator_nondet.deterministic is False


class TestValidateAgentFunction:
    """Test the convenience validate_agent function."""
    
    def test_validate_agent_function(self):
        """Test validate_agent convenience function."""
        np.random.seed(42)
        
        # Create sample data
        df = pd.DataFrame({
            'close': np.random.randn(150) + 10000,
            'volume': np.random.rand(150) * 1000,
        })
        
        # Create dummy agent
        agent = DummyAgent(strategy='random')
        
        # Validate
        metrics = validate_agent(
            agent,
            df,
            n_episodes=2,
            reward_mode='sortino',
            output_dir=None
        )
        
        assert 'mean_return' in metrics
        assert 'n_episodes' in metrics
        assert metrics['n_episodes'] == 2
    
    def test_validate_agent_with_save(self):
        """Test validate_agent with saving results."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': np.random.randn(150) + 10000,
            'volume': np.random.rand(150) * 1000,
        })
        
        agent = DummyAgent(strategy='always_hold')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test_results'
            
            metrics = validate_agent(
                agent,
                df,
                n_episodes=1,
                output_dir=output_dir
            )
            
            # Check that files were saved
            assert (output_dir / 'metrics.json').exists()


class TestValidationEdgeCases:
    """Test edge cases for validation."""
    
    def test_empty_episode(self):
        """Test with very short episodes."""
        # Create minimal data
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 10,
            'volume': [1000] * 50,
        })
        
        env = TradingEnv(df, window_size=5)
        agent = DummyAgent(strategy='always_hold')
        
        validator = ValidationWrapper(env, agent)
        metrics = validator.run_validation(n_episodes=1)
        
        # Should complete without errors
        assert 'mean_return' in metrics
    
    def test_agent_always_hold(self):
        """Test agent that always holds."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': np.random.randn(100) + 10000,
            'volume': np.random.rand(100) * 1000,
        })
        
        env = TradingEnv(df, window_size=20)
        agent = DummyAgent(strategy='always_hold')
        
        validator = ValidationWrapper(env, agent)
        metrics = validator.run_validation(n_episodes=1)
        
        # Return should be approximately 0 (no trades)
        assert abs(metrics['mean_return']) < 0.1  # Small due to no position
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': np.random.randn(150) + 10000,
            'volume': np.random.rand(150) * 1000,
        })
        
        env = TradingEnv(df, window_size=30)
        agent = DummyAgent(strategy='random')
        
        validator = ValidationWrapper(env, agent)
        metrics = validator.run_validation(n_episodes=10)
        
        # Win rate should be between 0 and 1
        assert 0.0 <= metrics['win_rate'] <= 1.0
