import numpy as np
import pytest
from bitcoin_scalper.core.rl_env import BitcoinScalperEnv

def test_rl_env_basic():
    # Génère un mini-dataset factice (prix + 2 features)
    data = np.hstack([
        np.linspace(10000, 10100, 100).reshape(-1, 1),
        np.random.randn(100, 2)
    ])
    env = BitcoinScalperEnv(data, fee=0.001, spread=0.0005, window_size=10, initial_balance=10000)
    obs = env.reset()
    assert obs.shape == (10, 3)
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs.shape == (10, 3)
        assert isinstance(reward, float)
        total_reward += reward
        steps += 1
    assert isinstance(total_reward, float)
    assert env.balance >= 0 