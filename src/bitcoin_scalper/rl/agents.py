"""
RL Agent Factory for PPO and DQN agents using Stable-Baselines3.

This module provides a factory class for creating, training, and managing
RL agents optimized for different market conditions:
- PPO (Proximal Policy Optimization): Best for trending/bull markets
- DQN (Deep Q-Network): Best for range-bound/choppy markets

References:
    Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
    Raffin, A., et al. (2021). Stable-Baselines3: Reliable RL Implementations.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable
import numpy as np

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        StopTrainingOnRewardThreshold,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    logger.warning("stable-baselines3 not available. Install it to use RL agents.")


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log custom metrics if episode ended
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep_info = info['episode']
                self.logger.record('episode/return', ep_info['total_return'])
                self.logger.record('episode/sharpe_ratio', ep_info['sharpe_ratio'])
                self.logger.record('episode/sortino_ratio', ep_info['sortino_ratio'])
                self.logger.record('episode/max_drawdown', ep_info['max_drawdown'])
        
        return True


class RLAgentFactory:
    """
    Factory for creating and managing RL agents (PPO, DQN).
    
    This class provides methods to initialize, train, save, and load RL agents
    using Stable-Baselines3. It configures agents with hyperparameters optimized
    for Bitcoin trading based on market regime.
    
    PPO Configuration:
    - On-policy algorithm
    - Best for trending/momentum markets (bull markets)
    - More aggressive, follows trends
    - Policy network: MlpPolicy with custom architecture
    
    DQN Configuration:
    - Off-policy algorithm with experience replay
    - Best for range-bound/choppy markets
    - More conservative, waits for clear signals
    - Double Dueling DQN architecture (reduces overestimation bias)
    
    Attributes:
        env: Trading environment (TradingEnv or vectorized).
        agent_type: Type of agent ('ppo' or 'dqn').
        model: Trained agent model (PPO or DQN instance).
    
    Example:
        >>> from bitcoin_scalper.rl import TradingEnv, RLAgentFactory
        >>> 
        >>> # Create environment
        >>> env = TradingEnv(df, reward_mode='sortino')
        >>> 
        >>> # Create PPO agent for bull market
        >>> factory = RLAgentFactory(env, agent_type='ppo')
        >>> factory.train(total_timesteps=100000, eval_freq=5000)
        >>> 
        >>> # Save trained agent
        >>> factory.save('models/ppo_bull_market')
        >>> 
        >>> # Load and use
        >>> factory.load('models/ppo_bull_market')
        >>> action, _states = factory.predict(obs)
    """
    
    def __init__(
        self,
        env: Any,
        agent_type: str = 'ppo',
        policy: str = 'MlpPolicy',
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize the RL agent factory.
        
        Args:
            env: Trading environment (TradingEnv or compatible gym environment).
            agent_type: Type of agent to create ('ppo' or 'dqn').
            policy: Policy network architecture ('MlpPolicy' for MLP).
            device: Device for training ('auto', 'cpu', 'cuda').
            verbose: Verbosity level (0: no output, 1: info, 2: debug).
            tensorboard_log: Path for TensorBoard logs (None disables).
        
        Raises:
            ImportError: If stable-baselines3 is not installed.
            ValueError: If agent_type is not supported.
        """
        if not _HAS_SB3:
            raise ImportError(
                "stable-baselines3 is required but not installed. "
                "Install it with: pip install stable-baselines3"
            )
        
        if agent_type not in ['ppo', 'dqn']:
            raise ValueError(f"agent_type must be 'ppo' or 'dqn', got '{agent_type}'")
        
        self.env = env
        self.agent_type = agent_type.lower()
        self.policy = policy
        self.device = device
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.model: Optional[Union[PPO, DQN]] = None
        
        logger.info(
            f"Initialized RLAgentFactory: agent_type={agent_type}, "
            f"policy={policy}, device={device}"
        )
    
    def create_agent(
        self,
        learning_rate: float = 3e-4,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[PPO, DQN]:
        """
        Create a new RL agent with specified hyperparameters.
        
        Args:
            learning_rate: Learning rate for optimizer.
            batch_size: Batch size for training.
                       PPO: number of steps per update (default 64)
                       DQN: size of replay buffer samples (default 32)
            **kwargs: Additional agent-specific hyperparameters.
        
        Returns:
            Initialized agent (PPO or DQN instance).
        
        PPO Hyperparameters:
            - n_steps: Steps per update (default 2048)
            - n_epochs: Epochs per update (default 10)
            - gamma: Discount factor (default 0.99)
            - gae_lambda: GAE lambda (default 0.95)
            - clip_range: PPO clipping (default 0.2)
        
        DQN Hyperparameters:
            - buffer_size: Replay buffer size (default 100000)
            - learning_starts: Steps before learning (default 1000)
            - target_update_interval: Target network update freq (default 1000)
            - exploration_fraction: Fraction for epsilon decay (default 0.1)
            - exploration_final_eps: Final epsilon (default 0.05)
        
        Example:
            >>> factory = RLAgentFactory(env, agent_type='ppo')
            >>> agent = factory.create_agent(
            ...     learning_rate=1e-4,
            ...     n_steps=1024,
            ...     gamma=0.99
            ... )
        """
        if self.agent_type == 'ppo':
            # PPO hyperparameters optimized for Bitcoin trading
            ppo_defaults = {
                'n_steps': kwargs.pop('n_steps', 2048),
                'batch_size': batch_size or kwargs.pop('batch_size', 64),
                'n_epochs': kwargs.pop('n_epochs', 10),
                'gamma': kwargs.pop('gamma', 0.99),
                'gae_lambda': kwargs.pop('gae_lambda', 0.95),
                'clip_range': kwargs.pop('clip_range', 0.2),
                'ent_coef': kwargs.pop('ent_coef', 0.01),  # Encourage exploration
                'vf_coef': kwargs.pop('vf_coef', 0.5),
                'max_grad_norm': kwargs.pop('max_grad_norm', 0.5),
                'policy_kwargs': kwargs.pop('policy_kwargs', {
                    'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # 2-layer MLP
                }),
            }
            ppo_defaults.update(kwargs)
            
            self.model = PPO(
                self.policy,
                self.env,
                learning_rate=learning_rate,
                verbose=self.verbose,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **ppo_defaults
            )
            
            logger.info(f"Created PPO agent with learning_rate={learning_rate}")
        
        elif self.agent_type == 'dqn':
            # DQN hyperparameters optimized for Bitcoin trading
            dqn_defaults = {
                'buffer_size': kwargs.pop('buffer_size', 100000),
                'learning_starts': kwargs.pop('learning_starts', 1000),
                'batch_size': batch_size or kwargs.pop('batch_size', 32),
                'tau': kwargs.pop('tau', 0.005),  # Soft update coefficient
                'gamma': kwargs.pop('gamma', 0.99),
                'target_update_interval': kwargs.pop('target_update_interval', 1000),
                'exploration_fraction': kwargs.pop('exploration_fraction', 0.1),
                'exploration_initial_eps': kwargs.pop('exploration_initial_eps', 1.0),
                'exploration_final_eps': kwargs.pop('exploration_final_eps', 0.05),
                'max_grad_norm': kwargs.pop('max_grad_norm', 10),
                'policy_kwargs': kwargs.pop('policy_kwargs', {
                    'net_arch': [256, 256]  # 2-layer MLP
                }),
            }
            dqn_defaults.update(kwargs)
            
            self.model = DQN(
                self.policy,
                self.env,
                learning_rate=learning_rate,
                verbose=self.verbose,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **dqn_defaults
            )
            
            logger.info(f"Created DQN agent with learning_rate={learning_rate}")
        
        return self.model
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
        tb_log_name: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> Union[PPO, DQN]:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of steps to train for.
            eval_env: Optional evaluation environment for periodic testing.
            eval_freq: Frequency of evaluation (in timesteps).
            n_eval_episodes: Number of episodes per evaluation.
            callback: Custom callback for training monitoring.
            log_interval: Frequency of logging (in updates).
            tb_log_name: Name for TensorBoard run.
            reset_num_timesteps: Whether to reset timestep counter.
        
        Returns:
            Trained agent model.
        
        Example:
            >>> # Train PPO for 100k steps with evaluation
            >>> factory = RLAgentFactory(env, agent_type='ppo')
            >>> factory.create_agent()
            >>> factory.train(
            ...     total_timesteps=100000,
            ...     eval_env=eval_env,
            ...     eval_freq=5000
            ... )
        
        Notes:
            - Use eval_env to monitor generalization to validation data
            - Training stops early if reward threshold is reached
            - TensorBoard logs include episode returns, Sharpe, Sortino
        """
        if self.model is None:
            logger.info("No model created yet, creating with default parameters")
            self.create_agent()
        
        # Setup evaluation callback if eval_env provided
        callbacks = []
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose,
            )
            callbacks.append(eval_callback)
        
        # Add tensorboard callback for custom metrics
        tb_callback = TensorboardCallback(verbose=self.verbose)
        callbacks.append(tb_callback)
        
        # Add custom callback if provided
        if callback is not None:
            callbacks.append(callback)
        
        # Train the model
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        if tb_log_name is None:
            tb_log_name = f"{self.agent_type}_agent"
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        )
        
        logger.info("Training completed")
        
        return self.model
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action given observation.
        
        Args:
            observation: Current state observation.
            deterministic: Whether to use deterministic policy (no exploration).
        
        Returns:
            Tuple of (action, internal_states).
        
        Example:
            >>> obs, info = env.reset()
            >>> action, _states = factory.predict(obs)
            >>> obs, reward, done, truncated, info = env.step(action)
        """
        if self.model is None:
            raise ValueError("No model available. Create or load a model first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save trained agent to disk.
        
        Args:
            path: Path to save model (without extension).
        
        Example:
            >>> factory.save('models/ppo_bitcoin_bull')
            # Saves to models/ppo_bitcoin_bull.zip
        """
        if self.model is None:
            raise ValueError("No model to save. Create or load a model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove .zip extension if provided (SB3 adds it automatically)
        if path.suffix == '.zip':
            path = path.with_suffix('')
        
        self.model.save(str(path))
        logger.info(f"Saved {self.agent_type.upper()} model to {path}.zip")
    
    def load(self, path: Union[str, Path]) -> Union[PPO, DQN]:
        """
        Load trained agent from disk.
        
        Args:
            path: Path to load model from (with or without .zip extension).
        
        Returns:
            Loaded agent model.
        
        Example:
            >>> factory = RLAgentFactory(env, agent_type='ppo')
            >>> factory.load('models/ppo_bitcoin_bull')
            >>> action, _states = factory.predict(obs)
        """
        path = Path(path)
        
        # Add .zip extension if not present
        if path.suffix != '.zip':
            path = path.with_suffix('.zip')
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the appropriate model type
        if self.agent_type == 'ppo':
            self.model = PPO.load(str(path), env=self.env, device=self.device)
        elif self.agent_type == 'dqn':
            self.model = DQN.load(str(path), env=self.env, device=self.device)
        
        logger.info(f"Loaded {self.agent_type.upper()} model from {path}")
        
        return self.model
    
    def set_env(self, env: Any) -> None:
        """
        Update environment for the agent.
        
        Args:
            env: New environment.
        
        Example:
            >>> # Switch to evaluation environment
            >>> factory.set_env(eval_env)
        """
        self.env = env
        if self.model is not None:
            self.model.set_env(env)
            logger.debug("Updated environment for model")


def create_ppo_agent(
    env: Any,
    learning_rate: float = 3e-4,
    **kwargs
) -> tuple:
    """
    Convenience function to create PPO agent.
    
    Args:
        env: Trading environment.
        learning_rate: Learning rate.
        **kwargs: Additional PPO hyperparameters.
    
    Returns:
        Tuple of (factory, model).
    
    Example:
        >>> env = TradingEnv(df, reward_mode='sortino')
        >>> factory, ppo = create_ppo_agent(env, learning_rate=1e-4)
        >>> factory.train(total_timesteps=100000)
    """
    factory = RLAgentFactory(env, agent_type='ppo', **kwargs)
    model = factory.create_agent(learning_rate=learning_rate, **kwargs)
    return factory, model


def create_dqn_agent(
    env: Any,
    learning_rate: float = 1e-4,
    **kwargs
) -> tuple:
    """
    Convenience function to create DQN agent.
    
    Args:
        env: Trading environment.
        learning_rate: Learning rate.
        **kwargs: Additional DQN hyperparameters.
    
    Returns:
        Tuple of (factory, model).
    
    Example:
        >>> env = TradingEnv(df, reward_mode='sortino')
        >>> factory, dqn = create_dqn_agent(env, learning_rate=1e-4)
        >>> factory.train(total_timesteps=100000)
    """
    factory = RLAgentFactory(env, agent_type='dqn', **kwargs)
    model = factory.create_agent(learning_rate=learning_rate, **kwargs)
    return factory, model
