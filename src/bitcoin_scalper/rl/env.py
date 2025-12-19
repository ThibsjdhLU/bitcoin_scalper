"""
Gymnasium-based trading environment for Deep Reinforcement Learning.

This environment simulates Bitcoin trading with realistic market conditions:
- Order book spreads and slippage
- Transaction fees
- Portfolio state tracking
- Risk-adjusted reward signals

Compatible with Stable-Baselines3 for training PPO and DQN agents.

References:
    Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.
    Towers, M., et al. (2023). Gymnasium.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging

from .rewards import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    DifferentialSharpeRatio,
    calculate_step_penalty,
)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for Bitcoin trading with RL agents.
    
    This environment implements the MDP formulation from Section 4.1:
    - State: Historical market data + portfolio state
    - Action: {0: Hold, 1: Buy, 2: Sell}
    - Reward: Risk-adjusted returns (Sharpe/Sortino/DSR)
    
    Features:
    - Realistic trading simulation with fees and spreads
    - Risk-adjusted reward functions
    - Portfolio state tracking (balance, position, P&L)
    - Episode termination on account bust
    - Compatible with stable-baselines3 PPO and DQN
    
    Attributes:
        df: DataFrame with market data (price, volume, indicators).
        window_size: Number of past timesteps in observation.
        initial_balance: Starting capital in USD.
        fee: Transaction fee rate (e.g., 0.001 = 0.1%).
        spread: Bid-ask spread rate (e.g., 0.0002 = 0.02%).
        reward_mode: Type of reward function ('sharpe', 'sortino', 'dsr', 'pnl').
        max_position_duration: Maximum steps to hold position (for penalty).
    
    Example:
        >>> df = pd.DataFrame({...})  # Market data
        >>> env = TradingEnv(df, window_size=30, reward_mode='sortino')
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()  # Random policy
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         break
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 10000.0,
        fee: float = 0.0005,
        spread: float = 0.0002,
        reward_mode: str = "sortino",
        max_position_duration: int = 100,
        bust_threshold: float = 0.05,  # Bust if balance < 5% of initial
        step_penalty_rate: float = 0.0001,
        dsr_eta: float = 0.001,
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with columns ['close', 'volume', ...features].
                Must have at least window_size + 1 rows.
            window_size: Number of timesteps in observation window.
            initial_balance: Starting capital in USD.
            fee: Transaction fee as fraction (0.0005 = 0.05% = 5 bps).
            spread: Bid-ask spread as fraction (0.0002 = 2 bps).
            reward_mode: Reward function type:
                'sharpe' - Sharpe ratio (penalizes all volatility)
                'sortino' - Sortino ratio (penalizes downside only, preferred)
                'dsr' - Differential Sharpe Ratio (online learning)
                'pnl' - Simple profit/loss (not recommended, too volatile)
            max_position_duration: Max steps before position penalty.
            bust_threshold: Episode ends if balance < bust_threshold * initial.
            step_penalty_rate: Penalty rate for holding positions.
            dsr_eta: Learning rate for DSR (if reward_mode='dsr').
        """
        super().__init__()
        
        # Validate inputs
        if len(df) <= window_size:
            raise ValueError(
                f"DataFrame must have more than window_size ({window_size}) rows, "
                f"got {len(df)} rows"
            )
        
        if reward_mode not in ['sharpe', 'sortino', 'dsr', 'pnl']:
            raise ValueError(
                f"reward_mode must be one of ['sharpe', 'sortino', 'dsr', 'pnl'], "
                f"got '{reward_mode}'"
            )
        
        # Store configuration
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.spread = spread
        self.reward_mode = reward_mode
        self.max_position_duration = max_position_duration
        self.bust_threshold = bust_threshold
        self.step_penalty_rate = step_penalty_rate
        
        # Extract price column (assume 'close' or first numeric column)
        if 'close' in self.df.columns:
            self.price_col_idx = self.df.columns.get_loc('close')
        else:
            # Find first numeric column
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("DataFrame must have at least one numeric column")
            self.price_col_idx = self.df.columns.get_loc(numeric_cols[0])
        
        # Convert to numpy for speed
        self.data = self.df.values.astype(np.float32)
        self.n_features = self.data.shape[1]
        
        # Define action space: {0: Hold, 1: Buy, 2: Sell}
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # Shape: (window_size, n_features + 4)
        # Features: market data + [balance, position, entry_price, unrealized_pnl]
        obs_shape = (window_size, self.n_features + 4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Initialize DSR if needed
        self.dsr = None
        if self.reward_mode == 'dsr':
            self.dsr = DifferentialSharpeRatio(eta=dsr_eta)
        
        # Episode state (initialized in reset())
        self.current_step = 0
        self.balance = 0.0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0
        self.position_size = 0.0  # Amount of BTC held
        self.position_duration = 0
        self.equity_curve: list = []
        self.returns: list = []
        self.done = False
        
        logger.info(
            f"Initialized TradingEnv: window={window_size}, "
            f"reward_mode={reward_mode}, fee={fee:.4f}, spread={spread:.4f}"
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        
        # Reset to starting position
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.position_duration = 0
        self.equity_curve = [self.initial_balance]
        self.returns = []
        self.done = False
        
        if self.dsr is not None:
            self.dsr.reset()
        
        obs = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset at step {self.current_step}")
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        Returns:
            Array of shape (window_size, n_features + 4).
            Last 4 columns are portfolio state [balance, position, entry_price, unrealized_pnl].
        """
        # Get historical market data window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        market_window = self.data[start_idx:end_idx, :].copy()
        
        # Get current price for unrealized P&L calculation
        current_price = self.data[self.current_step, self.price_col_idx]
        
        # Calculate unrealized P&L
        if self.position != 0 and self.position_size > 0:
            if self.position == 1:  # Long
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
        else:
            unrealized_pnl = 0.0
        
        # Normalize portfolio state values for better learning
        norm_balance = self.balance / self.initial_balance
        norm_position = float(self.position)  # Already in {-1, 0, 1}
        norm_entry = self.entry_price / self.initial_balance if self.entry_price > 0 else 0.0
        norm_pnl = unrealized_pnl / self.initial_balance
        
        # Create portfolio state columns (same for all timesteps in window)
        portfolio_state = np.array([
            [norm_balance, norm_position, norm_entry, norm_pnl]
        ] * self.window_size, dtype=np.float32)
        
        # Concatenate market data with portfolio state
        observation = np.concatenate([market_window, portfolio_state], axis=1)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        current_price = self.data[self.current_step, self.price_col_idx]
        total_equity = self._calculate_total_equity()
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'equity': total_equity,
            'position_duration': self.position_duration,
        }
    
    def _calculate_total_equity(self) -> float:
        """Calculate total account equity (balance + position value)."""
        current_price = self.data[self.current_step, self.price_col_idx]
        
        if self.position != 0 and self.position_size > 0:
            if self.position == 1:  # Long
                position_value = current_price * self.position_size
            else:  # Short
                # For short: value is (entry - current) * size
                position_value = (2 * self.entry_price - current_price) * self.position_size
            
            total = self.balance + position_value
        else:
            total = self.balance
        
        return float(total)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.
        
        Args:
            action: Action to take {0: Hold, 1: Buy, 2: Sell}.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.done:
            logger.warning("Called step() on done environment. Call reset() first.")
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Get current price
        current_price = self.data[self.current_step, self.price_col_idx]
        
        # Track equity before action
        equity_before = self._calculate_total_equity()
        
        # Execute action
        step_return = self._execute_action(action, current_price)
        
        # Track equity after action
        equity_after = self._calculate_total_equity()
        
        # Calculate step return
        if equity_before > 0:
            step_return = (equity_after - equity_before) / equity_before
        else:
            step_return = 0.0
        
        self.returns.append(step_return)
        self.equity_curve.append(equity_after)
        
        # Calculate reward based on mode
        reward = self._calculate_reward(step_return)
        
        # Update position duration
        if self.position != 0:
            self.position_duration += 1
            # Add step penalty for long-held positions
            penalty = calculate_step_penalty(
                self.position_duration,
                self.max_position_duration,
                self.step_penalty_rate
            )
            reward += penalty
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check if episode ends (ran out of data)
        if self.current_step >= len(self.data):
            truncated = True
            self.done = True
            logger.debug("Episode truncated: reached end of data")
        
        # Check if account busted
        if equity_after < self.bust_threshold * self.initial_balance:
            terminated = True
            self.done = True
            reward -= 10.0  # Large penalty for busting account
            logger.debug(f"Episode terminated: account bust (equity={equity_after:.2f})")
        
        # Get observation and info
        obs = self._get_observation() if not self.done else self._get_observation()
        info = self._get_info()
        
        # Add episode summary to info if done
        if terminated or truncated:
            info['episode'] = self._get_episode_summary()
        
        return obs, float(reward), terminated, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute trading action and update portfolio state.
        
        Args:
            action: Action to execute.
            current_price: Current market price.
        
        Returns:
            Immediate P&L from action (not including unrealized).
        """
        realized_pnl = 0.0
        
        if action == 0:  # Hold
            # Do nothing
            pass
        
        elif action == 1:  # Buy
            if self.position == 0:
                # Open long position
                # Apply spread: buy at higher price
                execution_price = current_price * (1 + self.spread)
                
                # Use 100% of balance (could be adjusted for risk management)
                position_value = self.balance * 0.95  # Keep 5% as buffer
                fee_cost = position_value * self.fee
                
                self.position_size = (position_value - fee_cost) / execution_price
                self.entry_price = execution_price
                self.position = 1
                self.balance -= position_value
                self.position_duration = 0
                
                logger.debug(f"Opened LONG: size={self.position_size:.6f} BTC @ {execution_price:.2f}")
            
            elif self.position == -1:
                # Close short and open long
                execution_price = current_price * (1 + self.spread)
                
                # Close short
                short_pnl = (self.entry_price - execution_price) * self.position_size
                close_fee = execution_price * self.position_size * self.fee
                realized_pnl = short_pnl - close_fee
                self.balance += realized_pnl
                
                # Open long with remaining balance
                position_value = self.balance * 0.95
                open_fee = position_value * self.fee
                
                self.position_size = (position_value - open_fee) / execution_price
                self.entry_price = execution_price
                self.position = 1
                self.balance -= position_value
                self.position_duration = 0
                
                logger.debug(f"Closed SHORT (PnL={realized_pnl:.2f}), Opened LONG")
            
            # else: already long, do nothing
        
        elif action == 2:  # Sell
            if self.position == 0:
                # Open short position
                # Apply spread: sell at lower price
                execution_price = current_price * (1 - self.spread)
                
                position_value = self.balance * 0.95
                fee_cost = position_value * self.fee
                
                self.position_size = (position_value - fee_cost) / execution_price
                self.entry_price = execution_price
                self.position = -1
                self.balance -= position_value
                self.position_duration = 0
                
                logger.debug(f"Opened SHORT: size={self.position_size:.6f} BTC @ {execution_price:.2f}")
            
            elif self.position == 1:
                # Close long and open short
                execution_price = current_price * (1 - self.spread)
                
                # Close long
                long_pnl = (execution_price - self.entry_price) * self.position_size
                close_fee = execution_price * self.position_size * self.fee
                realized_pnl = long_pnl - close_fee
                self.balance += realized_pnl + (self.entry_price * self.position_size)
                
                # Open short with remaining balance
                position_value = self.balance * 0.95
                open_fee = position_value * self.fee
                
                self.position_size = (position_value - open_fee) / execution_price
                self.entry_price = execution_price
                self.position = -1
                self.balance -= position_value
                self.position_duration = 0
                
                logger.debug(f"Closed LONG (PnL={realized_pnl:.2f}), Opened SHORT")
            
            # else: already short, do nothing
        
        return realized_pnl
    
    def _calculate_reward(self, step_return: float) -> float:
        """
        Calculate reward based on configured reward mode.
        
        Args:
            step_return: Return for current step.
        
        Returns:
            Reward value.
        """
        if self.reward_mode == 'pnl':
            # Simple P&L (not recommended, too volatile)
            return step_return * 100.0  # Scale up for better learning
        
        elif self.reward_mode == 'dsr':
            # Differential Sharpe Ratio (online)
            return self.dsr.update(step_return)
        
        elif self.reward_mode in ['sharpe', 'sortino']:
            # Calculate ratio over recent window (e.g., last 50 steps)
            if len(self.returns) < 10:
                return 0.0  # Not enough data yet
            
            recent_returns = np.array(self.returns[-50:])
            
            if self.reward_mode == 'sharpe':
                ratio = calculate_sharpe_ratio(recent_returns, periods_per_year=525600)
            else:  # sortino
                ratio = calculate_sortino_ratio(recent_returns, periods_per_year=525600)
            
            # Scale to reasonable range
            return ratio / 10.0
        
        return 0.0
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for completed episode."""
        returns_array = np.array(self.returns)
        equity_array = np.array(self.equity_curve)
        
        total_return = (equity_array[-1] - self.initial_balance) / self.initial_balance
        
        sharpe = calculate_sharpe_ratio(returns_array) if len(returns_array) > 1 else 0.0
        sortino = calculate_sortino_ratio(returns_array) if len(returns_array) > 1 else 0.0
        
        max_equity = np.max(equity_array)
        drawdown = (max_equity - equity_array[-1]) / max_equity if max_equity > 0 else 0.0
        
        return {
            'total_return': float(total_return),
            'final_equity': float(equity_array[-1]),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(drawdown),
            'n_steps': self.current_step - self.window_size,
        }
    
    def render(self, mode: str = "human") -> None:
        """
        Render environment state (for debugging).
        
        Args:
            mode: Render mode ('human' prints to console).
        """
        if mode == "human":
            info = self._get_info()
            equity = self._calculate_total_equity()
            print(
                f"Step: {info['step']:4d} | "
                f"Price: ${info['current_price']:8.2f} | "
                f"Position: {info['position']:2d} | "
                f"Balance: ${info['balance']:10.2f} | "
                f"Equity: ${equity:10.2f}"
            )
    
    def close(self) -> None:
        """Clean up resources."""
        pass
