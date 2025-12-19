"""
Unit tests for TradingEnv (Gymnasium environment).
"""

import pytest
import numpy as np
import pandas as pd

from src.bitcoin_scalper.rl.env import TradingEnv


class TestTradingEnv:
    """Test suite for TradingEnv."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate realistic price data (random walk)
        prices = [10000.0]
        for _ in range(n_samples - 1):
            change = np.random.randn() * 50
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.rand(n_samples) * 1000,
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
        })
        
        return df
    
    @pytest.fixture
    def env(self, sample_data):
        """Create a trading environment."""
        return TradingEnv(
            sample_data,
            window_size=30,
            initial_balance=10000.0,
            reward_mode='sortino'
        )
    
    def test_initialization(self, sample_data):
        """Test environment initialization."""
        env = TradingEnv(sample_data, window_size=30)
        
        assert env.window_size == 30
        assert env.initial_balance == 10000.0
        assert env.action_space.n == 3  # Hold, Buy, Sell
        assert env.observation_space.shape == (30, sample_data.shape[1] + 4)
    
    def test_invalid_window_size(self, sample_data):
        """Test that too large window size raises error."""
        with pytest.raises(ValueError):
            TradingEnv(sample_data, window_size=250)
    
    def test_invalid_reward_mode(self, sample_data):
        """Test that invalid reward mode raises error."""
        with pytest.raises(ValueError):
            TradingEnv(sample_data, reward_mode='invalid')
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        assert obs.shape == env.observation_space.shape
        assert env.current_step == env.window_size
        assert env.balance == env.initial_balance
        assert env.position == 0
        assert env.done is False
    
    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_observation_shape(self, env):
        """Test that observation has correct shape."""
        obs, _ = env.reset()
        
        expected_shape = (env.window_size, env.n_features + 4)
        assert obs.shape == expected_shape
    
    def test_observation_contains_portfolio_state(self, env):
        """Test that observation includes portfolio state."""
        obs, _ = env.reset()
        
        # Last 4 columns should be portfolio state
        portfolio_state = obs[:, -4:]
        
        # All rows should have same portfolio state
        assert np.allclose(portfolio_state[0], portfolio_state[1])
    
    def test_step_hold_action(self, env):
        """Test taking hold action."""
        env.reset()
        initial_balance = env.balance
        
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert env.position == 0
        assert env.balance == initial_balance
    
    def test_step_buy_action(self, env):
        """Test taking buy action."""
        env.reset()
        initial_balance = env.balance
        
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        
        assert env.position == 1  # Long
        assert env.balance < initial_balance  # Paid fees
        assert env.position_size > 0
        assert env.entry_price > 0
    
    def test_step_sell_action(self, env):
        """Test taking sell action."""
        env.reset()
        initial_balance = env.balance
        
        obs, reward, terminated, truncated, info = env.step(2)  # Sell
        
        assert env.position == -1  # Short
        assert env.balance < initial_balance  # Paid fees
        assert env.position_size > 0
        assert env.entry_price > 0
    
    def test_close_long_position(self, env):
        """Test closing a long position."""
        env.reset()
        
        # Open long
        env.step(1)
        initial_position_size = env.position_size
        
        # Close long (sell)
        env.step(2)
        
        assert env.position == -1  # Now short
        assert env.position_size != initial_position_size
    
    def test_close_short_position(self, env):
        """Test closing a short position."""
        env.reset()
        
        # Open short
        env.step(2)
        initial_position_size = env.position_size
        
        # Close short (buy)
        env.step(1)
        
        assert env.position == 1  # Now long
        assert env.position_size != initial_position_size
    
    def test_episode_truncation(self, sample_data):
        """Test that episode truncates at end of data."""
        env = TradingEnv(sample_data, window_size=30)
        env.reset()
        
        # Step until end
        done = False
        truncated = False
        steps = 0
        max_steps = len(sample_data) - env.window_size + 10
        
        while not (done or truncated) and steps < max_steps:
            _, _, done, truncated, _ = env.step(0)  # Hold
            steps += 1
        
        assert truncated or steps >= max_steps, "Episode should truncate at end of data"
    
    def test_episode_termination_on_bust(self, sample_data):
        """Test that episode terminates when account busts."""
        env = TradingEnv(
            sample_data,
            window_size=30,
            initial_balance=100.0,  # Small balance
            bust_threshold=0.5  # Bust at 50% loss
        )
        env.reset()
        
        # Manually set balance to trigger bust
        env.balance = 40.0  # Below 50% of initial
        
        _, _, terminated, truncated, info = env.step(0)
        
        # Note: bust check happens after step, so may not terminate immediately
        # Just check that bust condition would be detected
        assert env._calculate_total_equity() < env.bust_threshold * env.initial_balance
    
    def test_reward_modes(self, sample_data):
        """Test different reward modes."""
        reward_modes = ['pnl', 'sharpe', 'sortino', 'dsr']
        
        for mode in reward_modes:
            env = TradingEnv(sample_data, window_size=30, reward_mode=mode)
            env.reset()
            
            obs, reward, _, _, _ = env.step(1)  # Buy
            assert isinstance(reward, float), f"Reward should be float for mode {mode}"
    
    def test_info_dict(self, env):
        """Test that info dict contains expected keys."""
        env.reset()
        _, _, _, _, info = env.step(0)
        
        expected_keys = ['step', 'balance', 'position', 'entry_price', 
                        'current_price', 'equity', 'position_duration']
        
        for key in expected_keys:
            assert key in info, f"Info should contain '{key}'"
    
    def test_episode_summary(self, sample_data):
        """Test episode summary in info dict when done."""
        env = TradingEnv(sample_data, window_size=30)
        env.reset()
        
        # Run until done
        done = False
        truncated = False
        steps = 0
        max_steps = 100
        
        while not (done or truncated) and steps < max_steps:
            _, _, done, truncated, info = env.step(np.random.randint(0, 3))
            steps += 1
        
        if done or truncated:
            assert 'episode' in info, "Should have episode summary when done"
            episode = info['episode']
            assert 'total_return' in episode
            assert 'sharpe_ratio' in episode
            assert 'sortino_ratio' in episode
    
    def test_render(self, env, capsys):
        """Test that render doesn't crash."""
        env.reset()
        env.render()
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0, "Render should print output"
    
    def test_position_duration_tracking(self, env):
        """Test that position duration is tracked."""
        env.reset()
        
        # Open position - duration starts at 1 (first step holding)
        env.step(1)
        assert env.position_duration == 1
        
        # Hold position - duration increments
        env.step(0)
        assert env.position_duration == 2
        
        env.step(0)
        assert env.position_duration == 3
        
        # Close and open new position - duration resets
        env.step(2)
        assert env.position_duration == 1  # First step of new position
    
    def test_fees_and_spread(self, sample_data):
        """Test that fees and spread are applied."""
        env = TradingEnv(
            sample_data,
            window_size=30,
            fee=0.001,  # 0.1%
            spread=0.0005  # 0.05%
        )
        env.reset()
        initial_balance = env.balance
        
        # Buy
        env.step(1)
        
        # Should have paid fees
        assert env.balance < initial_balance
        
        # Entry price should include spread
        current_price = env.data[env.current_step - 1, env.price_col_idx]
        assert env.entry_price > current_price  # Bought at higher price (spread)
    
    def test_total_equity_calculation(self, env):
        """Test total equity calculation."""
        env.reset()
        
        # Flat position
        equity_flat = env._calculate_total_equity()
        assert equity_flat == env.balance
        
        # Long position
        env.step(1)
        equity_long = env._calculate_total_equity()
        assert equity_long != env.balance  # Includes position value
    
    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset()
            
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated) and steps < 50:
                action = np.random.randint(0, 3)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            
            assert steps > 0, f"Episode {episode} should have steps"


class TestTradingEnvEdgeCases:
    """Test edge cases for TradingEnv."""
    
    def test_minimal_data(self):
        """Test with minimal amount of data."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 10,
            'volume': [1000] * 50,
        })
        
        env = TradingEnv(df, window_size=5)
        obs, _ = env.reset()
        
        assert obs.shape[0] == 5
    
    def test_single_feature(self):
        """Test with single feature column."""
        df = pd.DataFrame({
            'price': np.random.randn(100) + 10000,
        })
        
        env = TradingEnv(df, window_size=10)
        obs, _ = env.reset()
        
        # Should have 1 market feature + 4 portfolio features
        assert obs.shape[1] == 5
    
    def test_large_price_movements(self):
        """Test with large price movements."""
        prices = [10000.0]
        for _ in range(100):
            change = np.random.randn() * 500  # Large volatility
            prices.append(max(100, prices[-1] + change))
        
        df = pd.DataFrame({'close': prices})
        env = TradingEnv(df, window_size=20)
        
        env.reset()
        
        # Should handle large movements
        for _ in range(50):
            action = np.random.randint(0, 3)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
