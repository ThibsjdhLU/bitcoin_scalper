"""
Deep Reinforcement Learning module for Bitcoin trading.

This module implements Section 4 of the ML Trading Bitcoin strategy:
- Custom Gymnasium trading environment
- RL agents (PPO, DQN) using Stable-Baselines3
- Advanced reward functions (Sharpe, Sortino, DSR)
- Validation wrappers for backtesting

References:
    Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
    Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
"""

from .env import TradingEnv
from .agents import RLAgentFactory
from .rewards import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_differential_sharpe_ratio,
)
from .validation import ValidationWrapper

__all__ = [
    "TradingEnv",
    "RLAgentFactory",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_differential_sharpe_ratio",
    "ValidationWrapper",
]
