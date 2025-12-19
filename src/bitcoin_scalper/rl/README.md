# Deep Reinforcement Learning Module

This module implements **Section 4** of the ML Trading Bitcoin strategy: Deep Reinforcement Learning agents for autonomous Bitcoin trading.

## Overview

The RL module provides a complete framework for training, validating, and deploying reinforcement learning agents that learn to trade through trial and error. Unlike supervised learning models that predict future prices, RL agents learn to maximize risk-adjusted returns directly.

```
rl/
├── __init__.py                  # Module exports
├── env.py                       # TradingEnv: Gymnasium trading environment
├── agents.py                    # RLAgentFactory: PPO & DQN agents
├── rewards.py                   # Risk-adjusted reward functions
└── validation.py                # Validation & backtesting tools
```

## Key Features

### 1. Trading Environment (`TradingEnv`)
A custom Gymnasium environment that simulates realistic Bitcoin trading:
- **Discrete action space:** {0: Hold, 1: Buy, 2: Sell}
- **Rich observations:** Market data window + portfolio state (balance, position, P&L)
- **Realistic simulation:** Transaction fees, bid-ask spreads, position tracking
- **Episode termination:** Account bust detection, end of data
- **Multiple reward modes:** PnL, Sharpe, Sortino, DSR

### 2. Risk-Adjusted Rewards (`rewards.py`)
Advanced reward functions that guide agents toward stable, profitable strategies:
- **Sharpe Ratio:** Standard risk-adjusted returns (penalizes all volatility)
- **Sortino Ratio:** Preferred for crypto (penalizes only downside volatility)
- **Differential Sharpe Ratio (DSR):** Online learning with incremental updates
- **Step penalty:** Discourages indefinite position holding

### 3. RL Agent Factory (`agents.py`)
Factory for creating and managing PPO and DQN agents:
- **PPO (Proximal Policy Optimization):** Best for trending/bull markets
- **DQN (Deep Q-Network):** Best for range-bound/choppy markets
- **Complete lifecycle:** Create, train, save, load, predict
- **TensorBoard integration:** Episode metrics logging
- **Hyperparameter tuning:** Pre-configured for Bitcoin trading

### 4. Validation Wrapper (`validation.py`)
Tools for validating agents on unseen data:
- **Multi-episode validation:** Run multiple evaluation episodes
- **Performance metrics:** Returns, Sharpe, Sortino, drawdown, win rate
- **Trade logging:** Detailed step-by-step trade records
- **Baseline comparison:** Compare vs buy-and-hold or random strategies

## Quick Start

### 1. Create Trading Environment

```python
import pandas as pd
from bitcoin_scalper.rl import TradingEnv

# Load your market data
df = pd.DataFrame({
    'close': [10000, 10050, 10100, ...],  # Price data
    'volume': [1000, 1100, 1050, ...],    # Volume data
    # ... additional features (RSI, MACD, etc.)
})

# Create environment
env = TradingEnv(
    df,
    window_size=30,           # 30 timesteps in observation
    initial_balance=10000.0,  # Starting capital
    reward_mode='sortino',    # Use Sortino ratio (recommended)
    fee=0.0005,              # 0.05% transaction fee
    spread=0.0002            # 0.02% bid-ask spread
)
```

### 2. Train PPO Agent (Bull Markets)

```python
from bitcoin_scalper.rl import RLAgentFactory

# Create agent factory
factory = RLAgentFactory(
    env,
    agent_type='ppo',
    verbose=1,
    tensorboard_log='./logs'
)

# Create and configure agent
model = factory.create_agent(
    learning_rate=3e-4,
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95
)

# Train agent
factory.train(
    total_timesteps=100000,
    eval_freq=5000,
    n_eval_episodes=5
)

# Save trained agent
factory.save('models/ppo_bull_market')
```

### 3. Train DQN Agent (Range Markets)

```python
# Create DQN agent
factory = RLAgentFactory(env, agent_type='dqn', verbose=1)

model = factory.create_agent(
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    exploration_final_eps=0.05
)

# Train
factory.train(total_timesteps=100000)

# Save
factory.save('models/dqn_range_market')
```

### 4. Validate Trained Agent

```python
from bitcoin_scalper.rl import ValidationWrapper

# Load trained agent
factory = RLAgentFactory(val_env, agent_type='ppo')
factory.load('models/ppo_bull_market')

# Create validation wrapper
validator = ValidationWrapper(val_env, factory.model)

# Run validation
metrics = validator.run_validation(
    n_episodes=10,
    log_trades=True
)

print(f"Mean Return: {metrics['mean_return']:.2%}")
print(f"Mean Sharpe: {metrics['mean_sharpe']:.2f}")
print(f"Mean Sortino: {metrics['mean_sortino']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")

# Compare with baseline
comparison = validator.compare_with_baseline('buy_and_hold')
print(f"Outperformance: {comparison['outperformance']:.2%}")

# Save results
validator.save_results('results/validation')
```

### 5. Use Agent in Production

```python
# Load trained agent
factory = RLAgentFactory(env, agent_type='ppo')
factory.load('models/ppo_bull_market')

# Get current observation
obs, info = env.reset()

# Make trading decision
action, _states = factory.predict(obs, deterministic=True)

# Interpret action
if action == 0:
    print("Hold current position")
elif action == 1:
    print("Buy BTC")
elif action == 2:
    print("Sell BTC")
```

## Reward Functions

### Sharpe Ratio
```python
from bitcoin_scalper.rl.rewards import calculate_sharpe_ratio

returns = [0.01, -0.005, 0.02, 0.015, -0.003]
sharpe = calculate_sharpe_ratio(returns, periods_per_year=525600)
print(f"Sharpe Ratio: {sharpe:.2f}")
```

**Characteristics:**
- Penalizes both upside and downside volatility
- Standard metric in traditional finance
- Less suitable for crypto with positive skew

### Sortino Ratio (Recommended)
```python
from bitcoin_scalper.rl.rewards import calculate_sortino_ratio

sortino = calculate_sortino_ratio(returns, periods_per_year=525600)
print(f"Sortino Ratio: {sortino:.2f}")
```

**Characteristics:**
- Penalizes only downside volatility
- Better for asymmetric return distributions (Bitcoin)
- Preferred metric for crypto optimization
- Recommended by López de Prado

### Differential Sharpe Ratio (DSR)
```python
from bitcoin_scalper.rl.rewards import DifferentialSharpeRatio

dsr = DifferentialSharpeRatio(eta=0.001)

for return_t in episode_returns:
    reward = dsr.update(return_t)
    # Use reward for RL training
```

**Characteristics:**
- Enables online learning (no need to wait for episode end)
- Incremental Sharpe ratio computation at each timestep
- Adapts quickly to regime changes
- Ideal for continuous trading

## Agent Selection Guide

### PPO: Proximal Policy Optimization
**Best for:** Trending/Bull Markets

**Characteristics:**
- On-policy algorithm (learns from recent experience)
- Aggressive trend-following behavior
- Direct policy optimization
- More exploratory

**Hyperparameters:**
```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,          # Steps per update
    'batch_size': 64,
    'n_epochs': 10,           # Epochs per update
    'gamma': 0.99,            # Discount factor
    'gae_lambda': 0.95,       # GAE parameter
    'clip_range': 0.2,        # PPO clipping
}
```

### DQN: Deep Q-Network
**Best for:** Range-bound/Choppy Markets

**Characteristics:**
- Off-policy algorithm (learns from replay buffer)
- Conservative, selective trading ("sniper" style)
- Value-based learning
- Better for sideways markets

**Hyperparameters:**
```python
{
    'learning_rate': 1e-4,
    'buffer_size': 100000,       # Replay buffer size
    'learning_starts': 1000,     # Steps before learning
    'batch_size': 32,
    'tau': 0.005,                # Soft update coefficient
    'gamma': 0.99,
    'exploration_final_eps': 0.05,  # Final exploration rate
}
```

## Environment Configuration

### Observation Space
Shape: `(window_size, n_features + 4)`
- Market features: Price, volume, technical indicators
- Portfolio state: `[balance, position, entry_price, unrealized_pnl]`

### Action Space
Discrete(3):
- `0`: Hold current position
- `1`: Buy (open long or close short)
- `2`: Sell (open short or close long)

### Reward Modes
- `'pnl'`: Simple profit/loss (not recommended, too volatile)
- `'sharpe'`: Sharpe ratio (standard)
- `'sortino'`: Sortino ratio (**recommended for crypto**)
- `'dsr'`: Differential Sharpe Ratio (online learning)

### Termination Conditions
1. **Episode complete:** Reached end of data
2. **Account bust:** Balance < bust_threshold * initial_balance

## Best Practices

### 1. Data Preparation
- Use at least 30,000+ timesteps for training
- Include relevant technical indicators (RSI, MACD, ATR, etc.)
- Normalize features to similar scales
- Split data: 70% train, 15% validation, 15% test

### 2. Reward Function Selection
- Start with **Sortino ratio** for crypto trading
- Use DSR for online learning scenarios
- Avoid simple PnL (too volatile for stable learning)
- Add step penalty to discourage inaction

### 3. Training Strategy
- Start with short episodes (1000 steps) for faster iteration
- Gradually increase episode length
- Use evaluation callbacks to monitor progress
- Save checkpoints regularly
- Monitor TensorBoard for training metrics

### 4. Hyperparameter Tuning
- PPO: Tune `n_steps` and `clip_range` first
- DQN: Tune `learning_rate` and `buffer_size` first
- Use lower learning rates for stability (1e-4 to 3e-4)
- Increase `gamma` closer to 1.0 for longer time horizons

### 5. Validation
- Always validate on unseen data (walk-forward)
- Run multiple episodes (10+) for stable metrics
- Compare against buy-and-hold baseline
- Check for consistent performance across market regimes

### 6. Deployment
- Use deterministic policy (`deterministic=True`)
- Implement position size limits
- Add manual circuit breakers
- Monitor live performance continuously
- Retrain periodically with recent data

## Advanced Features

### Custom Reward Functions
```python
class CustomReward:
    def calculate(self, returns, info):
        # Your custom logic
        return reward

# Use in environment
env = TradingEnv(df, reward_function=CustomReward())
```

### Multi-Agent Ensemble
```python
# Train multiple agents
ppo = train_ppo_agent(train_data)
dqn = train_dqn_agent(train_data)

# Use regime detector to select agent
if market_regime == 'trending':
    action = ppo.predict(obs)
else:
    action = dqn.predict(obs)
```

### Curriculum Learning
```python
# Start with simpler markets
easy_env = TradingEnv(easy_data, reward_mode='sortino')
factory.train(total_timesteps=50000)

# Progress to harder markets
factory.set_env(hard_env)
factory.train(total_timesteps=50000)
```

## References

1. **Schulman, J., et al. (2017).** Proximal Policy Optimization Algorithms. arXiv:1707.06347.
2. **Mnih, V., et al. (2015).** Human-level control through deep reinforcement learning. Nature.
3. **Moody, J., & Saffell, M. (2001).** Learning to trade via direct reinforcement. Neural Networks.
4. **Sortino, F., & Price, L. (1994).** Performance measurement in a downside risk framework.
5. **López de Prado, M. (2018).** Advances in Financial Machine Learning. Wiley.

## Troubleshooting

### Agent Not Learning
- Check reward signal (should vary during episode)
- Verify observation normalization
- Increase training timesteps
- Reduce learning rate
- Check for data leakage

### High Variance in Performance
- Increase training episodes
- Use Sortino instead of Sharpe
- Add step penalty
- Tune gamma (discount factor)
- Validate on longer periods

### Overfitting to Training Data
- Use walk-forward validation
- Implement early stopping
- Reduce model complexity
- Increase data diversity
- Use regularization (dropout, weight decay)

## Testing

Run tests with:
```bash
pytest tests/rl/ -v
```

Test coverage:
- **24 tests** for reward functions
- **25 tests** for trading environment
- **16 tests** for agent factory
- **16 tests** for validation wrapper

Total: **81 passing tests**
