"""
Position Sizing Strategies for Risk-Adjusted Capital Allocation.

This module implements mathematical methods for determining optimal position sizes:

1. **Kelly Criterion**: Maximizes long-term growth rate based on win probability
   and payoff ratio. Includes fractional Kelly to reduce volatility.

2. **Target Volatility Sizing**: Adjusts position size to maintain consistent
   portfolio volatility, regardless of individual asset volatility.

These methods answer "How much to buy?" based on model confidence, expected returns,
and market conditions.

References:
    Kelly, J. L. (1956). A New Interpretation of Information Rate.
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Tharp, V. (2007). Trade Your Way to Financial Freedom.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    
    All position sizers implement this interface to ensure consistency.
    """
    
    @abstractmethod
    def calculate_size(
        self,
        capital: float,
        price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size (number of units to buy/sell).
        
        Args:
            capital: Available capital for position.
            price: Current asset price.
            **kwargs: Strategy-specific parameters.
        
        Returns:
            Position size in number of units (can be fractional).
        """
        pass


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizer with fractional Kelly support.
    
    The Kelly Criterion maximizes long-term growth by sizing positions based on:
    - Win probability (p)
    - Payoff ratio (b = avg_win / avg_loss)
    
    Formula: f* = p - (1-p)/b = p - q/b
    where:
    - f* is the fraction of capital to risk
    - p is win probability
    - q = 1-p is loss probability
    - b is payoff ratio (win/loss)
    
    **Fractional Kelly** scales down the position to reduce volatility:
    - Full Kelly (fraction=1.0): Maximum growth, high volatility
    - Half Kelly (fraction=0.5): 75% of growth, 50% of volatility
    - Quarter Kelly (fraction=0.25): Conservative, stable
    
    Attributes:
        kelly_fraction: Scaling factor for Kelly sizing (0 < fraction <= 1).
                       Recommended: 0.25-0.5 for Bitcoin trading.
        max_leverage: Maximum allowed leverage/position size as fraction of capital.
        
    Example:
        >>> from bitcoin_scalper.risk import KellySizer
        >>> 
        >>> # Create Half-Kelly sizer
        >>> sizer = KellySizer(kelly_fraction=0.5, max_leverage=1.0)
        >>> 
        >>> # Calculate position size
        >>> # Model predicts 60% win probability with 2:1 payoff ratio
        >>> size = sizer.calculate_size(
        ...     capital=10000,
        ...     price=50000,
        ...     win_prob=0.60,
        ...     payoff_ratio=2.0
        ... )
        >>> print(f"Position size: {size:.4f} BTC")
        >>> print(f"Position value: ${size * 50000:.2f}")
        
    Notes:
        - Requires accurate estimates of win_prob and payoff_ratio
        - Use fractional Kelly to avoid over-leveraging
        - Returns 0 if Kelly formula gives negative or invalid result
        - Caps position at max_leverage to prevent excessive risk
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_leverage: float = 1.0,
    ):
        """
        Initialize Kelly Criterion position sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly bet to use (0 < fraction <= 1).
                          0.5 = Half Kelly (recommended)
                          0.25 = Quarter Kelly (conservative)
                          1.0 = Full Kelly (aggressive)
            max_leverage: Maximum position size as fraction of capital.
                         1.0 = no leverage, 2.0 = 2x leverage, etc.
        
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if not 0 < kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
        
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        
        logger.info(
            f"KellySizer initialized with fraction={kelly_fraction}, "
            f"max_leverage={max_leverage}"
        )
    
    def calculate_size(
        self,
        capital: float,
        price: float,
        win_prob: float,
        payoff_ratio: float,
        **kwargs
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            capital: Available capital ($).
            price: Current asset price ($/unit).
            win_prob: Probability of winning trade (0 < p < 1).
                     From model or historical win rate.
            payoff_ratio: Average win / average loss ratio (b > 0).
                         E.g., 2.0 means wins are 2x losses on average.
            **kwargs: Additional parameters (ignored).
        
        Returns:
            Position size in units (e.g., BTC).
            Returns 0.0 if parameters indicate not to trade.
            
        Example:
            >>> sizer = KellySizer(kelly_fraction=0.5)
            >>> 
            >>> # Model with 55% accuracy, 1.5:1 reward/risk
            >>> size = sizer.calculate_size(
            ...     capital=10000,
            ...     price=40000,
            ...     win_prob=0.55,
            ...     payoff_ratio=1.5
            ... )
        """
        if capital <= 0:
            logger.warning(f"Invalid capital: {capital}")
            return 0.0
        
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return 0.0
        
        if not 0 < win_prob < 1:
            logger.warning(f"Invalid win_prob: {win_prob}, must be in (0, 1)")
            return 0.0
        
        if payoff_ratio <= 0:
            logger.warning(f"Invalid payoff_ratio: {payoff_ratio}, must be > 0")
            return 0.0
        
        # Kelly formula: f* = p - q/b
        loss_prob = 1 - win_prob
        kelly_fraction_optimal = win_prob - (loss_prob / payoff_ratio)
        
        # Apply fractional Kelly
        kelly_fraction_actual = kelly_fraction_optimal * self.kelly_fraction
        
        # Ensure non-negative
        if kelly_fraction_actual <= 0:
            logger.debug(
                f"Kelly formula negative: {kelly_fraction_optimal:.4f}, "
                "no position"
            )
            return 0.0
        
        # Cap at max leverage
        kelly_fraction_actual = min(kelly_fraction_actual, self.max_leverage)
        
        # Calculate position value
        position_value = capital * kelly_fraction_actual
        
        # Convert to units
        position_size = position_value / price
        
        logger.debug(
            f"Kelly sizing: win_prob={win_prob:.3f}, payoff_ratio={payoff_ratio:.3f}, "
            f"kelly_optimal={kelly_fraction_optimal:.4f}, "
            f"kelly_actual={kelly_fraction_actual:.4f}, "
            f"position_size={position_size:.6f} units"
        )
        
        return position_size
    
    def calculate_from_model_confidence(
        self,
        capital: float,
        price: float,
        confidence: float,
        expected_return: float,
        stop_loss_pct: float = 0.02,
        **kwargs
    ) -> float:
        """
        Calculate position size from model confidence and expected return.
        
        Convenience method that converts model outputs to Kelly parameters:
        - confidence → win_prob
        - expected_return and stop_loss → payoff_ratio
        
        Args:
            capital: Available capital.
            price: Current price.
            confidence: Model confidence/probability (0-1).
            expected_return: Expected return if trade wins (e.g., 0.03 = 3%).
            stop_loss_pct: Stop loss as fraction (e.g., 0.02 = 2%).
            **kwargs: Additional parameters.
        
        Returns:
            Position size in units.
            
        Example:
            >>> # Model predicts 70% confidence with 3% expected return
            >>> size = sizer.calculate_from_model_confidence(
            ...     capital=10000,
            ...     price=45000,
            ...     confidence=0.70,
            ...     expected_return=0.03,
            ...     stop_loss_pct=0.02
            ... )
        """
        # Convert to Kelly parameters
        win_prob = confidence
        payoff_ratio = abs(expected_return / stop_loss_pct) if stop_loss_pct > 0 else 1.0
        
        return self.calculate_size(
            capital=capital,
            price=price,
            win_prob=win_prob,
            payoff_ratio=payoff_ratio
        )


class TargetVolatilitySizer(PositionSizer):
    """
    Target Volatility position sizer.
    
    Adjusts position size so that the portfolio volatility equals a target level,
    regardless of individual asset volatility. When asset volatility increases,
    position size decreases (and vice versa).
    
    Formula: position_size = (target_vol * capital) / (asset_vol * price)
    
    where:
    - target_vol: Target annualized portfolio volatility (e.g., 0.40 = 40%)
    - asset_vol: Annualized asset volatility (from historical data)
    
    Benefits:
    - Consistent risk across different volatility regimes
    - Automatic deleveraging in high volatility (protects capital)
    - Automatic leveraging in low volatility (maximizes returns)
    
    Attributes:
        target_volatility: Target annualized portfolio volatility (0 < vol < infinity).
                          Typical: 0.20-0.60 for Bitcoin.
        max_leverage: Maximum position size as fraction of capital.
        
    Example:
        >>> from bitcoin_scalper.risk import TargetVolatilitySizer
        >>> 
        >>> # Target 40% annualized volatility
        >>> sizer = TargetVolatilitySizer(target_volatility=0.40)
        >>> 
        >>> # Calculate position with current BTC volatility
        >>> size = sizer.calculate_size(
        ...     capital=10000,
        ...     price=50000,
        ...     asset_volatility=0.80  # 80% annualized
        ... )
        >>> print(f"Position: {size:.4f} BTC")
        
    Notes:
        - Requires accurate volatility estimate (use EWMA or rolling std)
        - Volatility should be annualized (multiply daily vol by sqrt(365))
        - Higher asset volatility → smaller position
        - Works well with mean-reversion strategies
    """
    
    def __init__(
        self,
        target_volatility: float = 0.40,
        max_leverage: float = 1.0,
    ):
        """
        Initialize Target Volatility position sizer.
        
        Args:
            target_volatility: Target annualized portfolio volatility.
                             0.20 = 20% (conservative)
                             0.40 = 40% (moderate, recommended for BTC)
                             0.60 = 60% (aggressive)
            max_leverage: Maximum position size as fraction of capital.
        
        Raises:
            ValueError: If parameters are invalid.
        """
        if target_volatility <= 0:
            raise ValueError(
                f"target_volatility must be > 0, got {target_volatility}"
            )
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
        
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        
        logger.info(
            f"TargetVolatilitySizer initialized with "
            f"target_vol={target_volatility:.2%}, max_leverage={max_leverage}"
        )
    
    def calculate_size(
        self,
        capital: float,
        price: float,
        asset_volatility: float,
        **kwargs
    ) -> float:
        """
        Calculate position size for target volatility.
        
        Args:
            capital: Available capital ($).
            price: Current asset price ($/unit).
            asset_volatility: Annualized asset volatility (e.g., 0.80 = 80%).
                            Calculate from historical returns.
            **kwargs: Additional parameters (ignored).
        
        Returns:
            Position size in units.
            
        Example:
            >>> sizer = TargetVolatilitySizer(target_volatility=0.40)
            >>> 
            >>> # BTC at $40k with 70% annualized vol
            >>> size = sizer.calculate_size(
            ...     capital=10000,
            ...     price=40000,
            ...     asset_volatility=0.70
            ... )
        """
        if capital <= 0:
            logger.warning(f"Invalid capital: {capital}")
            return 0.0
        
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return 0.0
        
        if asset_volatility <= 0:
            logger.warning(f"Invalid asset_volatility: {asset_volatility}")
            return 0.0
        
        # Calculate position value for target volatility
        # portfolio_vol = (position_value / capital) * asset_vol
        # Solve for position_value:
        # position_value = (target_vol / asset_vol) * capital
        position_fraction = self.target_volatility / asset_volatility
        
        # Cap at max leverage
        position_fraction = min(position_fraction, self.max_leverage)
        
        position_value = position_fraction * capital
        
        # Convert to units
        position_size = position_value / price
        
        logger.debug(
            f"TargetVol sizing: asset_vol={asset_volatility:.2%}, "
            f"target_vol={self.target_volatility:.2%}, "
            f"position_fraction={position_fraction:.4f}, "
            f"position_size={position_size:.6f} units"
        )
        
        return position_size
    
    @staticmethod
    def estimate_volatility(
        returns: pd.Series,
        window: int = 20,
        min_periods: int = 10,
        annualization_factor: float = 365.0,
    ) -> float:
        """
        Estimate annualized volatility from returns series.
        
        Uses exponentially weighted moving average (EWMA) for recent volatility.
        
        Args:
            returns: Series of returns (not prices).
            window: Lookback window for volatility calculation.
            min_periods: Minimum periods required for calculation.
            annualization_factor: Factor to annualize volatility.
                                 365 for daily data, 252 for business days.
        
        Returns:
            Annualized volatility estimate.
            
        Example:
            >>> # Calculate daily returns
            >>> returns = prices.pct_change().dropna()
            >>> 
            >>> # Estimate current volatility
            >>> vol = TargetVolatilitySizer.estimate_volatility(
            ...     returns, window=20, annualization_factor=365
            ... )
            >>> print(f"Current volatility: {vol:.2%}")
        """
        if len(returns) < min_periods:
            logger.warning(
                f"Insufficient data for volatility estimate: "
                f"{len(returns)} < {min_periods}"
            )
            return 0.0
        
        # Use EWMA for recent volatility
        ewm_std = returns.ewm(span=window, min_periods=min_periods).std()
        
        # Get most recent volatility
        daily_vol = ewm_std.iloc[-1]
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(annualization_factor)
        
        return annualized_vol
