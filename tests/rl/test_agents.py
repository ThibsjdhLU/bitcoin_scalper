"""
Unit tests for RL agents and factory.

Note: These tests may require stable-baselines3 to be installed.
They will be skipped if the library is not available.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.bitcoin_scalper.rl.env import TradingEnv

try:
    from src.bitcoin_scalper.rl.agents import (
        RLAgentFactory,
        create_ppo_agent,
        create_dqn_agent,
    )
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
class TestRLAgentFactory:
    """Test suite for RLAgentFactory."""
    
    @pytest.fixture
    def sample_env(self):
        """Create a sample trading environment."""
        np.random.seed(42)
        
        # Generate sample data
        prices = [10000.0]
        for _ in range(150):
            change = np.random.randn() * 50
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.rand(len(prices)) * 1000,
        })
        
        return TradingEnv(df, window_size=30, reward_mode='sortino')
    
    def test_initialization_ppo(self, sample_env):
        """Test factory initialization with PPO."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        
        assert factory.agent_type == 'ppo'
        assert factory.env is sample_env
        assert factory.model is None
    
    def test_initialization_dqn(self, sample_env):
        """Test factory initialization with DQN."""
        factory = RLAgentFactory(sample_env, agent_type='dqn')
        
        assert factory.agent_type == 'dqn'
        assert factory.env is sample_env
        assert factory.model is None
    
    def test_invalid_agent_type(self, sample_env):
        """Test that invalid agent type raises error."""
        with pytest.raises(ValueError):
            RLAgentFactory(sample_env, agent_type='invalid')
    
    def test_create_ppo_agent(self, sample_env):
        """Test creating PPO agent."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        model = factory.create_agent(learning_rate=1e-4)
        
        assert model is not None
        assert factory.model is model
    
    def test_create_dqn_agent(self, sample_env):
        """Test creating DQN agent."""
        factory = RLAgentFactory(sample_env, agent_type='dqn')
        model = factory.create_agent(learning_rate=1e-4)
        
        assert model is not None
        assert factory.model is model
    
    def test_create_agent_with_custom_params(self, sample_env):
        """Test creating agent with custom hyperparameters."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        model = factory.create_agent(
            learning_rate=1e-4,
            n_steps=1024,
            gamma=0.95,
        )
        
        assert model is not None
    
    def test_train_minimal(self, sample_env):
        """Test minimal training run."""
        factory = RLAgentFactory(sample_env, agent_type='ppo', verbose=0)
        factory.create_agent()
        
        # Train for minimal steps
        factory.train(total_timesteps=100, log_interval=100)
        
        assert factory.model is not None
    
    def test_predict(self, sample_env):
        """Test prediction."""
        factory = RLAgentFactory(sample_env, agent_type='ppo', verbose=0)
        factory.create_agent()
        factory.train(total_timesteps=100, log_interval=100)
        
        obs, _ = sample_env.reset()
        action, _states = factory.predict(obs)
        
        # Action can be int, np.integer, or np.ndarray with single element
        if isinstance(action, np.ndarray):
            action = action.item()
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3
    
    def test_predict_without_model(self, sample_env):
        """Test that predict without model raises error."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        
        obs, _ = sample_env.reset()
        
        with pytest.raises(ValueError):
            factory.predict(obs)
    
    def test_save_and_load(self, sample_env):
        """Test saving and loading agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_agent'
            
            # Create and train agent
            factory1 = RLAgentFactory(sample_env, agent_type='ppo', verbose=0)
            factory1.create_agent()
            factory1.train(total_timesteps=100, log_interval=100)
            
            # Save
            factory1.save(save_path)
            
            # Load in new factory
            factory2 = RLAgentFactory(sample_env, agent_type='ppo', verbose=0)
            factory2.load(save_path)
            
            # Test prediction
            obs, _ = sample_env.reset()
            action, _ = factory2.predict(obs)
            if isinstance(action, np.ndarray):
                action = action.item()
            assert isinstance(action, (int, np.integer))
    
    def test_load_nonexistent_file(self, sample_env):
        """Test that loading nonexistent file raises error."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        
        with pytest.raises(FileNotFoundError):
            factory.load('nonexistent_model.zip')
    
    def test_save_without_model(self, sample_env):
        """Test that save without model raises error."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_agent'
            
            with pytest.raises(ValueError):
                factory.save(save_path)
    
    def test_set_env(self, sample_env):
        """Test setting environment."""
        factory = RLAgentFactory(sample_env, agent_type='ppo')
        factory.create_agent()
        
        # Create new environment
        df = pd.DataFrame({
            'close': np.random.randn(100) + 10000,
            'volume': np.random.rand(100) * 1000,
        })
        new_env = TradingEnv(df, window_size=30)
        
        # Set new environment
        factory.set_env(new_env)
        
        assert factory.env is new_env


@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
class TestConvenienceFunctions:
    """Test convenience functions for agent creation."""
    
    @pytest.fixture
    def sample_env(self):
        """Create a sample trading environment."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': np.random.randn(100) + 10000,
            'volume': np.random.rand(100) * 1000,
        })
        
        return TradingEnv(df, window_size=30)
    
    def test_create_ppo_agent_function(self, sample_env):
        """Test create_ppo_agent convenience function."""
        factory, model = create_ppo_agent(sample_env, learning_rate=1e-4, verbose=0)
        
        assert isinstance(factory, RLAgentFactory)
        assert factory.agent_type == 'ppo'
        assert model is not None
    
    def test_create_dqn_agent_function(self, sample_env):
        """Test create_dqn_agent convenience function."""
        factory, model = create_dqn_agent(sample_env, learning_rate=1e-4, verbose=0)
        
        assert isinstance(factory, RLAgentFactory)
        assert factory.agent_type == 'dqn'
        assert model is not None


class TestAgentsWithoutSB3:
    """Test behavior when stable-baselines3 is not available."""
    
    def test_import_without_sb3(self):
        """Test that module can be imported even without SB3."""
        # This test just checks that import doesn't crash
        # The actual import happens at module level
        pass
