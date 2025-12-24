"""
Hybrid reward system for Deep Reinforcement Learning.

This module implements risk-adjusted reward functions to guide RL agents
towards profitable and stable trading strategies.

Key Features:
- Sharpe Ratio: Risk-adjusted returns (penalizes all volatility)
- Sortino Ratio: Downside risk-adjusted returns (preferred for crypto)
- Differential Sharpe Ratio: Online learning with incremental updates
- Step penalty: Discourages indefinite holding without profit

References:
    Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
    Sortino, F., & Price, L. (1994). Performance measurement in a downside risk framework.
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525600  # Minutes in a year for M1 data
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    The Sharpe ratio measures risk-adjusted returns by dividing excess returns
    by their standard deviation. Higher values indicate better risk-adjusted performance.
    
    Formula:
        Sharpe = sqrt(periods) * (mean(returns) - risk_free) / std(returns)
    
    Args:
        returns: Array of period returns (e.g., per-step returns).
        risk_free_rate: Annual risk-free rate (default 0 for crypto).
        periods_per_year: Number of periods in a year for annualization.
                         525600 for minute data, 252 for daily data.
    
    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero or insufficient data.
    
    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.015])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
        
    Notes:
        - Standard Sharpe penalizes both upside and downside volatility
        - For crypto with high positive skew, Sortino may be more appropriate
        - Typical good Sharpe ratio in traditional finance: > 1.0
        - Crypto strategies: > 2.0 is excellent
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    # Convert risk-free rate to per-period rate
    rf_per_period = risk_free_rate / periods_per_year
    
    # Calculate annualized Sharpe ratio
    sharpe = np.sqrt(periods_per_year) * (mean_return - rf_per_period) / std_return
    
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525600,
    target_return: float = 0.0
) -> float:
    """
    Calculate annualized Sortino Ratio.
    
    The Sortino ratio is a variation of the Sharpe ratio that only penalizes
    downside volatility (below target return). This is more appropriate for
    assets like Bitcoin that have positive skew.
    
    Formula:
        Sortino = sqrt(periods) * (mean(returns) - target) / downside_std
        
    where downside_std only includes returns below the target.
    
    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate (default 0).
        periods_per_year: Number of periods in a year.
        target_return: Minimum acceptable return per period (default 0).
    
    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside_std is zero.
    
    Example:
        >>> returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
        >>> sortino = calculate_sortino_ratio(returns)
        >>> print(f"Sortino Ratio: {sortino:.2f}")
        
    Notes:
        - Sortino > Sharpe for positively skewed returns (Bitcoin)
        - Only penalizes "bad" volatility (losses)
        - Preferred metric for crypto optimization
        - López de Prado recommends this for asymmetric return distributions
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Calculate downside deviation (only negative deviations from target)
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        # No downside risk observed
        return float('inf') if mean_return > target_return else 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    # Convert risk-free rate to per-period rate
    rf_per_period = risk_free_rate / periods_per_year
    
    # Calculate annualized Sortino ratio
    sortino = np.sqrt(periods_per_year) * (mean_return - rf_per_period) / downside_std
    
    return float(sortino)


class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio (DSR) for online learning.
    
    DSR enables incremental computation of Sharpe ratio at each timestep,
    allowing RL agents to receive immediate risk-adjusted feedback without
    waiting for episode completion.
    
    The DSR approximates the gradient of the Sharpe ratio with respect to
    the policy parameters, enabling direct policy gradient optimization.
    
    Formula:
        DSR_t = (B_t * A_t - A_t * B_t) / (B_t - 1)^1.5
        
        where:
        A_t = exponentially weighted mean of returns
        B_t = exponentially weighted mean of squared returns
    
    Attributes:
        eta: Learning rate for exponential smoothing (0 < eta < 1).
             Smaller values = more smoothing (slower adaptation).
        A: Exponential moving average of returns.
        B: Exponential moving average of squared returns.
        t: Current timestep.
    
    Example:
        >>> dsr = DifferentialSharpeRatio(eta=0.01)
        >>> for return_t in episode_returns:
        ...     reward = dsr.update(return_t)
        ...     # Use reward for RL training
        
    References:
        Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
        Neural Networks, 14(4-5), 437-471.
    """
    
    def __init__(self, eta: float = 0.001):
        """
        Initialize DSR calculator.
        
        Args:
            eta: Learning rate for exponential smoothing.
                 Typical values: 0.001 - 0.01
                 Lower = smoother but slower adaptation
                 Higher = faster but noisier
        """
        if not 0 < eta < 1:
            raise ValueError(f"eta must be in (0, 1), got {eta}")
        
        self.eta = eta
        self.A: float = 0.0  # Exponential moving average of returns
        self.B: float = 1.0  # Exponential moving average of squared returns
        self.t: int = 0
        
        logger.debug(f"Initialized DSR with eta={eta}")
    
    def update(self, return_t: float) -> float:
        """
        Update DSR with new return and compute differential Sharpe.
        
        Args:
            return_t: Return at current timestep.
        
        Returns:
            Differential Sharpe ratio value (used as reward signal).
        
        Example:
            >>> dsr = DifferentialSharpeRatio(eta=0.01)
            >>> reward = dsr.update(0.005)  # 0.5% return
            >>> print(f"DSR reward: {reward:.4f}")
        """
        self.t += 1
        
        # Update exponential moving averages
        self.A = (1 - self.eta) * self.A + self.eta * return_t
        self.B = (1 - self.eta) * self.B + self.eta * (return_t ** 2)
        
        # Compute differential Sharpe ratio
        denominator = (self.B - self.A ** 2)
        
        if denominator <= 0 or self.t < 10:
            # Avoid division issues early in training or with zero variance
            return 0.0
        
        dsr = (return_t - self.A) / np.sqrt(denominator)
        
        # Clip to prevent extreme values
        dsr = np.clip(dsr, -10.0, 10.0)
        
        return float(dsr)
    
    def reset(self) -> None:
        """Reset the DSR state (call at episode start)."""
        self.A = 0.0
        self.B = 1.0
        self.t = 0
        logger.debug("DSR state reset")
    
    def get_sharpe_estimate(self) -> float:
        """
        Get current Sharpe ratio estimate.
        
        Returns:
            Estimated Sharpe ratio based on current A and B.
        """
        variance = self.B - self.A ** 2
        if variance <= 0:
            return 0.0
        return float(self.A / np.sqrt(variance))


def calculate_differential_sharpe_ratio(
    returns: List[float],
    eta: float = 0.001
) -> Tuple[List[float], float]:
    """
    Calculate DSR for a sequence of returns (batch mode).
    
    This is a convenience function for computing DSR rewards for
    a complete episode in hindsight (e.g., for analysis).
    
    For online RL training, use the DifferentialSharpeRatio class directly.
    
    Args:
        returns: List of returns for the episode.
        eta: Learning rate for exponential smoothing.
    
    Returns:
        Tuple of:
        - List of DSR values (one per timestep)
        - Final Sharpe ratio estimate
    
    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.003]
        >>> dsr_values, final_sharpe = calculate_differential_sharpe_ratio(returns)
        >>> print(f"DSR rewards: {dsr_values}")
        >>> print(f"Final Sharpe: {final_sharpe:.2f}")
    """
    dsr = DifferentialSharpeRatio(eta=eta)
    dsr_values = []
    
    for return_t in returns:
        dsr_val = dsr.update(return_t)
        dsr_values.append(dsr_val)
    
    final_sharpe = dsr.get_sharpe_estimate()
    
    return dsr_values, final_sharpe


def calculate_step_penalty(
    position_duration: int,
    max_duration: int = 100,
    penalty_rate: float = 0.0001
) -> float:
    """
    Calculate penalty for holding positions too long without profit.
    
    This penalty discourages the agent from holding positions indefinitely
    when there's no clear directional move. It encourages active trading
    and position management.
    
    Args:
        position_duration: Number of steps position has been held.
        max_duration: Duration at which penalty reaches maximum.
        penalty_rate: Base penalty per step (default 0.0001 = 0.01%).
    
    Returns:
        Penalty value (negative float).
    
    Example:
        >>> penalty = calculate_step_penalty(position_duration=50, max_duration=100)
        >>> print(f"Penalty: {penalty:.6f}")
        
    Notes:
        - Penalty increases linearly with duration
        - Caps at max_duration to avoid extreme penalties
        - Should be small relative to typical returns
        - Helps prevent "do nothing" strategies
    """
    duration = min(position_duration, max_duration)
    penalty = -penalty_rate * duration
    return float(penalty)
