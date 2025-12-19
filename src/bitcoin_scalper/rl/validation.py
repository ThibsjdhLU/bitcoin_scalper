"""
Validation wrapper for backtesting trained RL agents.

This module provides tools to validate RL agents on unseen data,
ensuring they haven't simply memorized the training set.

Key Features:
- Run trained agents on validation/test data
- Collect performance metrics (returns, Sharpe, Sortino, drawdown)
- Generate trading logs for analysis
- Compare multiple agents

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 11: The Dangers of Backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from .env import TradingEnv
from .rewards import calculate_sharpe_ratio, calculate_sortino_ratio

logger = logging.getLogger(__name__)


class ValidationWrapper:
    """
    Wrapper for validating trained RL agents on unseen data.
    
    This class runs a trained agent through a trading environment with
    test/validation data and collects comprehensive performance metrics.
    
    The goal is to verify that the agent can generalize to new market
    conditions and hasn't simply overfit to the training data.
    
    Attributes:
        env: Trading environment with validation data.
        agent: Trained RL agent (PPO, DQN, or any with predict() method).
        trade_log: DataFrame with step-by-step trading records.
        episode_metrics: Dict with episode-level performance metrics.
    
    Example:
        >>> from bitcoin_scalper.rl import TradingEnv, RLAgentFactory, ValidationWrapper
        >>> 
        >>> # Load trained agent
        >>> factory = RLAgentFactory(env, agent_type='ppo')
        >>> factory.load('models/ppo_trained')
        >>> 
        >>> # Create validation environment with unseen data
        >>> val_env = TradingEnv(val_df, reward_mode='sortino')
        >>> 
        >>> # Validate
        >>> validator = ValidationWrapper(val_env, factory.model)
        >>> metrics = validator.run_validation(n_episodes=10)
        >>> print(f"Sharpe: {metrics['mean_sharpe']:.2f}")
        >>> print(f"Sortino: {metrics['mean_sortino']:.2f}")
        >>> print(f"Win Rate: {metrics['win_rate']:.2%}")
    """
    
    def __init__(
        self,
        env: TradingEnv,
        agent: Any,
        deterministic: bool = True,
    ):
        """
        Initialize validation wrapper.
        
        Args:
            env: Trading environment with validation/test data.
            agent: Trained RL agent with predict() method.
            deterministic: Whether to use deterministic policy (recommended).
        """
        self.env = env
        self.agent = agent
        self.deterministic = deterministic
        
        self.trade_log: Optional[pd.DataFrame] = None
        self.episode_metrics: Dict[str, Any] = {}
        
        logger.info(f"Initialized ValidationWrapper with deterministic={deterministic}")
    
    def run_validation(
        self,
        n_episodes: int = 1,
        render: bool = False,
        log_trades: bool = True,
    ) -> Dict[str, Any]:
        """
        Run validation over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run.
            render: Whether to render environment (prints to console).
            log_trades: Whether to log individual trades.
        
        Returns:
            Dictionary with aggregated metrics across all episodes:
            - mean_return: Average total return
            - mean_sharpe: Average Sharpe ratio
            - mean_sortino: Average Sortino ratio
            - mean_max_drawdown: Average maximum drawdown
            - win_rate: Fraction of profitable episodes
            - std_return: Std dev of returns (consistency measure)
            - all_episodes: List of per-episode metrics
        
        Example:
            >>> metrics = validator.run_validation(n_episodes=10)
            >>> print(f"Mean Return: {metrics['mean_return']:.2%}")
            >>> print(f"Sharpe: {metrics['mean_sharpe']:.2f}")
        """
        logger.info(f"Starting validation for {n_episodes} episodes")
        
        all_episode_metrics = []
        all_trades = []
        
        for episode_idx in range(n_episodes):
            logger.debug(f"Running episode {episode_idx + 1}/{n_episodes}")
            
            # Run single episode
            episode_data = self._run_episode(
                episode_idx=episode_idx,
                render=render,
                log_trades=log_trades
            )
            
            all_episode_metrics.append(episode_data['metrics'])
            
            if log_trades:
                all_trades.extend(episode_data['trades'])
        
        # Aggregate metrics across episodes
        aggregated = self._aggregate_metrics(all_episode_metrics)
        aggregated['all_episodes'] = all_episode_metrics
        
        # Store trade log
        if log_trades and all_trades:
            self.trade_log = pd.DataFrame(all_trades)
        
        self.episode_metrics = aggregated
        
        logger.info(
            f"Validation complete: "
            f"mean_return={aggregated['mean_return']:.2%}, "
            f"mean_sharpe={aggregated['mean_sharpe']:.2f}, "
            f"win_rate={aggregated['win_rate']:.2%}"
        )
        
        return aggregated
    
    def _run_episode(
        self,
        episode_idx: int,
        render: bool,
        log_trades: bool,
    ) -> Dict[str, Any]:
        """
        Run a single validation episode.
        
        Args:
            episode_idx: Episode index (for logging).
            render: Whether to render.
            log_trades: Whether to log trades.
        
        Returns:
            Dict with episode metrics and trade log.
        """
        obs, info = self.env.reset()
        done = False
        truncated = False
        
        episode_trades = []
        episode_steps = 0
        
        while not (done or truncated):
            # Get action from agent
            action, _states = self.agent.predict(obs, deterministic=self.deterministic)
            
            # Take step
            obs, reward, done, truncated, info = self.env.step(action)
            episode_steps += 1
            
            # Log trade if requested
            if log_trades:
                trade_record = {
                    'episode': episode_idx,
                    'step': episode_steps,
                    'action': int(action),
                    'price': info['current_price'],
                    'position': info['position'],
                    'balance': info['balance'],
                    'equity': info['equity'],
                    'reward': reward,
                }
                episode_trades.append(trade_record)
            
            # Render if requested
            if render:
                self.env.render()
        
        # Get episode summary
        episode_summary = info.get('episode', {})
        
        return {
            'metrics': episode_summary,
            'trades': episode_trades,
        }
    
    def _aggregate_metrics(self, episode_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple episodes.
        
        Args:
            episode_metrics: List of per-episode metrics.
        
        Returns:
            Aggregated metrics dictionary.
        """
        if not episode_metrics:
            return {}
        
        # Extract metrics
        returns = [m.get('total_return', 0.0) for m in episode_metrics]
        sharpes = [m.get('sharpe_ratio', 0.0) for m in episode_metrics]
        sortinos = [m.get('sortino_ratio', 0.0) for m in episode_metrics]
        drawdowns = [m.get('max_drawdown', 0.0) for m in episode_metrics]
        
        # Calculate aggregated statistics
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        mean_sharpe = float(np.mean(sharpes))
        mean_sortino = float(np.mean(sortinos))
        mean_max_drawdown = float(np.mean(drawdowns))
        
        # Win rate (fraction of profitable episodes)
        win_rate = float(np.mean([r > 0 for r in returns]))
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_sharpe': mean_sharpe,
            'mean_sortino': mean_sortino,
            'mean_max_drawdown': mean_max_drawdown,
            'win_rate': win_rate,
            'n_episodes': len(episode_metrics),
        }
    
    def get_trade_log(self) -> Optional[pd.DataFrame]:
        """
        Get detailed trade log from validation.
        
        Returns:
            DataFrame with columns:
            - episode: Episode number
            - step: Step within episode
            - action: Action taken (0=Hold, 1=Buy, 2=Sell)
            - price: Execution price
            - position: Current position (-1=Short, 0=Flat, 1=Long)
            - balance: Cash balance
            - equity: Total account value
            - reward: Reward received
        
        Example:
            >>> validator.run_validation(n_episodes=5)
            >>> log = validator.get_trade_log()
            >>> print(log.head())
        """
        return self.trade_log
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated validation metrics.
        
        Returns:
            Dictionary of metrics from last validation run.
        """
        return self.episode_metrics
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save validation results to disk.
        
        Args:
            output_dir: Directory to save results.
        
        Saves:
            - metrics.json: Aggregated metrics
            - trade_log.csv: Detailed trade log (if available)
        
        Example:
            >>> validator.run_validation(n_episodes=10)
            >>> validator.save_results('results/validation')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = output_dir / 'metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            # Convert to JSON-serializable format
            metrics = self.episode_metrics.copy()
            if 'all_episodes' in metrics:
                metrics['all_episodes'] = [
                    {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                     for k, v in ep.items()}
                    for ep in metrics['all_episodes']
                ]
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save trade log as CSV
        if self.trade_log is not None:
            log_path = output_dir / 'trade_log.csv'
            self.trade_log.to_csv(log_path, index=False)
            logger.info(f"Saved trade log to {log_path}")
    
    def compare_with_baseline(
        self,
        baseline_strategy: str = 'buy_and_hold'
    ) -> Dict[str, Any]:
        """
        Compare agent performance with a baseline strategy.
        
        Args:
            baseline_strategy: Baseline to compare against:
                'buy_and_hold': Buy at start, hold until end
                'random': Random actions
        
        Returns:
            Dict with comparison metrics:
            - agent_return: Agent's total return
            - baseline_return: Baseline's total return
            - outperformance: Difference (agent - baseline)
            - sharpe_improvement: Agent Sharpe - Baseline Sharpe
        
        Example:
            >>> comparison = validator.compare_with_baseline('buy_and_hold')
            >>> print(f"Outperformance: {comparison['outperformance']:.2%}")
        
        Notes:
            - Helps assess if RL agent adds value over simple strategies
            - Negative outperformance suggests overfitting or poor generalization
        """
        if not self.episode_metrics:
            raise ValueError("Run validation first using run_validation()")
        
        # Get agent metrics
        agent_return = self.episode_metrics['mean_return']
        agent_sharpe = self.episode_metrics['mean_sharpe']
        
        # Calculate baseline performance
        if baseline_strategy == 'buy_and_hold':
            baseline_return, baseline_sharpe = self._calculate_buy_and_hold()
        elif baseline_strategy == 'random':
            baseline_return, baseline_sharpe = self._calculate_random_strategy()
        else:
            raise ValueError(f"Unknown baseline strategy: {baseline_strategy}")
        
        outperformance = agent_return - baseline_return
        sharpe_improvement = agent_sharpe - baseline_sharpe
        
        logger.info(
            f"Comparison vs {baseline_strategy}: "
            f"outperformance={outperformance:.2%}, "
            f"sharpe_improvement={sharpe_improvement:.2f}"
        )
        
        return {
            'baseline_strategy': baseline_strategy,
            'agent_return': agent_return,
            'baseline_return': baseline_return,
            'outperformance': outperformance,
            'agent_sharpe': agent_sharpe,
            'baseline_sharpe': baseline_sharpe,
            'sharpe_improvement': sharpe_improvement,
        }
    
    def _calculate_buy_and_hold(self) -> tuple:
        """Calculate buy-and-hold baseline performance."""
        # Get price data from environment
        prices = self.env.data[:, self.env.price_col_idx]
        
        # Simple return: (final - initial) / initial
        initial_price = prices[self.env.window_size]
        final_price = prices[-1]
        
        bh_return = (final_price - initial_price) / initial_price
        
        # Calculate returns for Sharpe
        price_returns = np.diff(prices[self.env.window_size:]) / prices[self.env.window_size:-1]
        bh_sharpe = calculate_sharpe_ratio(price_returns, periods_per_year=525600)
        
        return float(bh_return), float(bh_sharpe)
    
    def _calculate_random_strategy(self) -> tuple:
        """
        Calculate random action baseline performance.
        
        Note: This is not fully implemented. A proper implementation would require
        re-running the environment with random actions for each step, which is
        computationally expensive. This method returns estimates as placeholders.
        
        For accurate random strategy comparison, manually run validation with a
        random agent using ValidationWrapper.
        
        Returns:
            Tuple of (return, sharpe) - currently returns (0.0, 0.0) as placeholder
        """
        logger.warning(
            "Random strategy comparison not fully implemented. "
            "For accurate results, run validation with a random agent manually."
        )
        return 0.0, 0.0


def validate_agent(
    agent: Any,
    val_df: pd.DataFrame,
    n_episodes: int = 10,
    reward_mode: str = 'sortino',
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to validate an agent on test data.
    
    Args:
        agent: Trained RL agent.
        val_df: Validation data DataFrame.
        n_episodes: Number of episodes to run.
        reward_mode: Reward mode for validation environment.
        output_dir: Optional directory to save results.
    
    Returns:
        Validation metrics dictionary.
    
    Example:
        >>> from bitcoin_scalper.rl import RLAgentFactory, validate_agent
        >>> 
        >>> # Load trained agent
        >>> factory = RLAgentFactory(None, agent_type='ppo')
        >>> agent = factory.load('models/ppo_trained')
        >>> 
        >>> # Validate on test data
        >>> metrics = validate_agent(
        ...     agent,
        ...     test_df,
        ...     n_episodes=10,
        ...     output_dir='results/test'
        ... )
    """
    # Create validation environment
    val_env = TradingEnv(val_df, reward_mode=reward_mode)
    
    # Create validator
    validator = ValidationWrapper(val_env, agent)
    
    # Run validation
    metrics = validator.run_validation(n_episodes=n_episodes)
    
    # Save results if output directory provided
    if output_dir is not None:
        validator.save_results(output_dir)
    
    return metrics
