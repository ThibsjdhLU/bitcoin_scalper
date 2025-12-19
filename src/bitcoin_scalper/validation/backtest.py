"""
Event-Driven Backtesting Engine for Trading Strategies.

This module provides a realistic backtesting framework that:
1. Accepts signals from ML/RL models
2. Integrates position sizing from risk management
3. Simulates realistic execution with slippage and commissions
4. Generates comprehensive performance reports

The event-driven approach ensures no look-ahead bias and models realistic
order execution dynamics.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Bailey, D. H., et al. (2014). The Deflated Sharpe Ratio.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading actions."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    EXIT = 2


@dataclass
class Trade:
    """
    Record of a single trade execution.
    
    Attributes:
        timestamp: Trade execution time.
        signal: Signal type that triggered trade.
        action: Actual action taken (BUY/SELL).
        price: Execution price.
        size: Position size (units).
        commission: Commission paid.
        slippage: Slippage cost.
    """
    timestamp: pd.Timestamp
    signal: SignalType
    action: str  # 'BUY' or 'SELL'
    price: float
    size: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total transaction cost."""
        return self.commission + self.slippage


@dataclass
class Position:
    """
    Current position state.
    
    Attributes:
        size: Current position size (positive=long, negative=short, 0=flat).
        entry_price: Average entry price.
        entry_timestamp: Time of entry.
    """
    size: float = 0.0
    entry_price: float = 0.0
    entry_timestamp: Optional[pd.Timestamp] = None
    
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.size) < 1e-8
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 1e-8
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < -1e-8


@dataclass
class BacktestResult:
    """
    Comprehensive backtest results and performance metrics.
    
    Attributes:
        equity_curve: Time series of portfolio value.
        trades: List of all executed trades.
        returns: Series of period returns.
        metrics: Dictionary of performance metrics.
    """
    equity_curve: pd.Series
    trades: List[Trade]
    returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate metrics after initialization."""
        if not self.metrics:
            self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        if len(self.returns) == 0:
            logger.warning("No returns to calculate metrics")
            return metrics
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        metrics['total_return'] = total_return
        metrics['n_trades'] = len(self.trades)
        
        # Return statistics
        metrics['mean_return'] = self.returns.mean()
        metrics['std_return'] = self.returns.std()
        metrics['median_return'] = self.returns.median()
        
        # Win rate
        winning_trades = sum(1 for r in self.returns if r > 0)
        metrics['win_rate'] = winning_trades / len(self.returns) if len(self.returns) > 0 else 0
        
        # Sharpe Ratio (annualized, assuming daily returns)
        if self.returns.std() > 0:
            sharpe = self.returns.mean() / self.returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino Ratio (only penalize downside volatility)
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = self.returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0.0
        metrics['sortino_ratio'] = sortino
        
        # Maximum Drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Calmar Ratio (return / max_drawdown)
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = 0.0
        metrics['calmar_ratio'] = calmar
        
        # Trade statistics
        if len(self.trades) > 0:
            commissions = sum(t.commission for t in self.trades)
            slippage = sum(t.slippage for t in self.trades)
            metrics['total_commissions'] = commissions
            metrics['total_slippage'] = slippage
            metrics['total_costs'] = commissions + slippage
        
        return metrics
    
    def summary(self) -> str:
        """
        Generate human-readable summary of results.
        
        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "BACKTEST RESULTS SUMMARY",
            "=" * 60,
            "",
            "Performance Metrics:",
            f"  Total Return:      {self.metrics.get('total_return', 0):.2%}",
            f"  Sharpe Ratio:      {self.metrics.get('sharpe_ratio', 0):.3f}",
            f"  Sortino Ratio:     {self.metrics.get('sortino_ratio', 0):.3f}",
            f"  Max Drawdown:      {self.metrics.get('max_drawdown', 0):.2%}",
            f"  Calmar Ratio:      {self.metrics.get('calmar_ratio', 0):.3f}",
            "",
            "Trade Statistics:",
            f"  Number of Trades:  {self.metrics.get('n_trades', 0)}",
            f"  Win Rate:          {self.metrics.get('win_rate', 0):.2%}",
            f"  Mean Return:       {self.metrics.get('mean_return', 0):.4%}",
            f"  Std Return:        {self.metrics.get('std_return', 0):.4%}",
            "",
            "Costs:",
            f"  Commissions:       ${self.metrics.get('total_commissions', 0):.2f}",
            f"  Slippage:          ${self.metrics.get('total_slippage', 0):.2f}",
            f"  Total Costs:       ${self.metrics.get('total_costs', 0):.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Event-driven backtesting engine with realistic execution simulation.
    
    This engine simulates trading a strategy with:
    - Signal generation from ML/RL models
    - Position sizing from risk management
    - Realistic slippage and commission costs
    - Portfolio tracking and metrics
    
    The event-driven approach processes data chronologically, preventing
    look-ahead bias and modeling realistic trading conditions.
    
    Attributes:
        initial_capital: Starting capital ($).
        commission_pct: Commission as percentage of trade value.
        slippage_pct: Slippage as percentage of trade value.
        position_sizer: Position sizing strategy (optional).
        
    Example:
        >>> from bitcoin_scalper.validation import Backtester
        >>> from bitcoin_scalper.risk import KellySizer
        >>> 
        >>> # Setup
        >>> backtester = Backtester(
        ...     initial_capital=10000,
        ...     commission_pct=0.001,  # 0.1%
        ...     slippage_pct=0.0005,   # 0.05%
        ...     position_sizer=KellySizer(kelly_fraction=0.5)
        ... )
        >>> 
        >>> # Run backtest with ML signals
        >>> results = backtester.run(
        ...     prices=price_series,
        ...     signals=ml_signals,
        ...     signal_params={'win_prob': 0.6, 'payoff_ratio': 2.0}
        ... )
        >>> 
        >>> # Analyze results
        >>> print(results.summary())
        >>> print(f"Sharpe: {results.metrics['sharpe_ratio']:.2f}")
        
    Notes:
        - Signals can be from ML models, RL agents, or any strategy
        - Position sizer determines "how much" for each trade
        - Realistic costs prevent overoptimistic backtest results
        - Compatible with purged cross-validation
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_sizer: Optional[Any] = None,
        signal_threshold_long: float = 0.6,
        signal_threshold_short: float = 0.4,
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital ($).
            commission_pct: Commission rate (e.g., 0.001 = 0.1%).
            slippage_pct: Slippage rate (e.g., 0.0005 = 0.05%).
            position_sizer: Position sizing strategy instance.
                          If None, uses full capital for each trade.
            signal_threshold_long: Threshold for converting continuous signals to LONG (default 0.6).
            signal_threshold_short: Threshold for converting continuous signals to SHORT (default 0.4).
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
        if commission_pct < 0:
            raise ValueError(f"commission_pct must be >= 0, got {commission_pct}")
        if slippage_pct < 0:
            raise ValueError(f"slippage_pct must be >= 0, got {slippage_pct}")
        
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.position_sizer = position_sizer
        self.signal_threshold_long = signal_threshold_long
        self.signal_threshold_short = signal_threshold_short
        
        # State
        self.capital = initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_history: List[tuple] = []
        
        logger.info(
            f"Backtester initialized: capital=${initial_capital}, "
            f"commission={commission_pct:.3%}, slippage={slippage_pct:.3%}"
        )
    
    def reset(self) -> None:
        """Reset backtester to initial state."""
        self.capital = self.initial_capital
        self.position = Position()
        self.trades = []
        self.equity_history = []
        logger.info("Backtester reset")
    
    def run(
        self,
        prices: pd.Series,
        signals: Union[pd.Series, np.ndarray],
        signal_params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        Run backtest with given prices and signals.
        
        Args:
            prices: Series of asset prices with DatetimeIndex.
            signals: Series or array of trading signals.
                    Values: 1=LONG, -1=SHORT, 0=NEUTRAL/EXIT.
                    Or can be continuous values (thresholded internally).
            signal_params: Optional parameters for position sizing
                         (e.g., win_prob, payoff_ratio, asset_volatility).
        
        Returns:
            BacktestResult with equity curve, trades, and metrics.
            
        Example:
            >>> # Binary signals (1, 0, -1)
            >>> results = backtester.run(prices, ml_signals)
            >>> 
            >>> # Continuous signals (will be thresholded)
            >>> results = backtester.run(prices, probabilities)
            >>> 
            >>> # With position sizing parameters
            >>> results = backtester.run(
            ...     prices, signals,
            ...     signal_params={'win_prob': 0.6, 'payoff_ratio': 2.0}
            ... )
        """
        self.reset()
        
        if signal_params is None:
            signal_params = {}
        
        # Ensure signals is a Series
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals, index=prices.index)
        
        # Align signals with prices
        signals = signals.loc[prices.index]
        
        logger.info(f"Running backtest on {len(prices)} price points")
        
        # Process each time step
        for timestamp, price in prices.items():
            signal = signals.loc[timestamp]
            
            # Handle continuous signals (convert to discrete)
            if not isinstance(signal, (int, np.integer)):
                if signal > self.signal_threshold_long:  # Strong buy signal
                    signal = SignalType.LONG
                elif signal < self.signal_threshold_short:  # Strong sell signal
                    signal = SignalType.SHORT
                else:
                    signal = SignalType.NEUTRAL
            else:
                # Convert integer to SignalType
                if signal > 0:
                    signal = SignalType.LONG
                elif signal < 0:
                    signal = SignalType.SHORT
                else:
                    signal = SignalType.NEUTRAL
            
            # Process signal
            self._process_signal(timestamp, price, signal, signal_params)
            
            # Record equity
            equity = self._calculate_equity(price)
            self.equity_history.append((timestamp, equity))
        
        # Close any open position at end
        if not self.position.is_flat():
            self._close_position(prices.index[-1], prices.iloc[-1])
        
        # Create result
        result = self._create_result()
        
        logger.info(
            f"Backtest complete: {len(self.trades)} trades, "
            f"final equity=${self.capital:.2f}"
        )
        
        return result
    
    def _process_signal(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: SignalType,
        signal_params: Dict[str, Any],
    ) -> None:
        """Process trading signal at given timestamp."""
        # If neutral signal or exit, close position
        if signal == SignalType.NEUTRAL or signal == SignalType.EXIT:
            if not self.position.is_flat():
                self._close_position(timestamp, price)
            return
        
        # Long signal
        if signal == SignalType.LONG:
            if self.position.is_short():
                # Close short, open long
                self._close_position(timestamp, price)
                self._open_position(timestamp, price, SignalType.LONG, signal_params)
            elif self.position.is_flat():
                # Open long
                self._open_position(timestamp, price, SignalType.LONG, signal_params)
            # If already long, do nothing
        
        # Short signal
        elif signal == SignalType.SHORT:
            if self.position.is_long():
                # Close long, open short
                self._close_position(timestamp, price)
                self._open_position(timestamp, price, SignalType.SHORT, signal_params)
            elif self.position.is_flat():
                # Open short
                self._open_position(timestamp, price, SignalType.SHORT, signal_params)
            # If already short, do nothing
    
    def _open_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: SignalType,
        signal_params: Dict[str, Any],
    ) -> None:
        """Open new position."""
        # Calculate position size
        if self.position_sizer is not None:
            size = self.position_sizer.calculate_size(
                capital=self.capital,
                price=price,
                **signal_params
            )
        else:
            # Use full capital
            size = self.capital / price
        
        if size <= 0:
            logger.debug(f"Position sizer returned size <= 0, no trade")
            return
        
        # Apply slippage (worse execution price)
        if signal == SignalType.LONG:
            execution_price = price * (1 + self.slippage_pct)
            action = 'BUY'
            position_size = size
        else:  # SHORT
            execution_price = price * (1 - self.slippage_pct)
            action = 'SELL'
            position_size = -size
        
        # Calculate costs
        trade_value = size * execution_price
        commission = trade_value * self.commission_pct
        slippage_cost = size * price * self.slippage_pct
        
        # Update capital
        self.capital -= (trade_value + commission)
        
        # Update position
        self.position.size = position_size
        self.position.entry_price = execution_price
        self.position.entry_timestamp = timestamp
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            signal=signal,
            action=action,
            price=execution_price,
            size=size,
            commission=commission,
            slippage=slippage_cost,
        )
        self.trades.append(trade)
        
        logger.debug(
            f"{action} {size:.6f} @ ${execution_price:.2f}, "
            f"costs=${commission + slippage_cost:.2f}"
        )
    
    def _close_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
    ) -> None:
        """Close current position."""
        if self.position.is_flat():
            return
        
        size = abs(self.position.size)
        
        # Apply slippage
        if self.position.is_long():
            execution_price = price * (1 - self.slippage_pct)
            action = 'SELL'
        else:  # SHORT
            execution_price = price * (1 + self.slippage_pct)
            action = 'BUY'
        
        # Calculate costs
        trade_value = size * execution_price
        commission = trade_value * self.commission_pct
        slippage_cost = size * price * self.slippage_pct
        
        # Update capital
        self.capital += (trade_value - commission)
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            signal=SignalType.EXIT,
            action=action,
            price=execution_price,
            size=size,
            commission=commission,
            slippage=slippage_cost,
        )
        self.trades.append(trade)
        
        # Reset position
        self.position = Position()
        
        logger.debug(
            f"CLOSE {action} {size:.6f} @ ${execution_price:.2f}, "
            f"costs=${commission + slippage_cost:.2f}"
        )
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current portfolio equity."""
        # Cash + position value
        if self.position.is_flat():
            return self.capital
        
        position_value = abs(self.position.size) * current_price
        
        if self.position.is_long():
            # Long: gain if price up
            return self.capital + position_value
        else:
            # Short: gain if price down
            # We owe the position, so it's a liability
            short_pnl = abs(self.position.size) * (self.position.entry_price - current_price)
            return self.capital + short_pnl
    
    def _create_result(self) -> BacktestResult:
        """Create BacktestResult from backtest history."""
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_curve = equity_df.set_index('timestamp')['equity']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            returns=returns,
        )
