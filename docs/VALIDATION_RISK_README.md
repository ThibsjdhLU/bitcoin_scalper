# Validation & Risk Management Modules

This directory contains the **validation** and **risk** management modules for the Bitcoin Scalper trading system, implementing Section 5 (VALIDATION) and Section 6.1 (RISK MANAGEMENT) of the ML Trading Bitcoin strategy.

## Overview

These modules provide:

1. **Scientific Validation** (`validation/`): Tools to ensure backtests are scientifically valid
2. **Risk Management** (`risk/`): Mathematical methods for position sizing and capital allocation

Together, they form the "Lie Detector" (validation) and "Shield" (risk management) for the trading system.

## Validation Module (`src/bitcoin_scalper/validation/`)

### Components

#### 1. Combinatorial Purged Cross-Validation (`cross_val.py`)

Implements the gold standard for financial ML validation:

- **PurgedKFold**: K-fold cross-validation with purging and embargo
  - Removes training samples whose labels overlap with test set (purging)
  - Adds buffer period after test sets (embargo)
  - Prevents look-ahead bias from Triple Barrier labeling
  - Compatible with scikit-learn API

- **CombinatorialPurgedCV**: Generates multiple train-test combinations
  - Creates distribution of performance metrics
  - Tests model across diverse market regimes
  - Provides confidence intervals instead of point estimates

**Example:**
```python
from bitcoin_scalper.validation import PurgedKFold
from sklearn.ensemble import RandomForestClassifier

# Create purged cross-validator
cv = PurgedKFold(n_splits=5, embargo_pct=0.01)

# Use with sklearn
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=cv.split(X, y, t1=event_times))
```

#### 2. Drift Detection (`drift.py`)

Monitors model performance in real-time to detect concept drift:

- **ADWINDetector**: Lightweight implementation of ADWIN algorithm
  - Automatically detects changes in error distribution
  - Memory efficient with adaptive windowing
  - Online learning compatible

- **DriftScanner**: High-level drift monitoring system
  - Wraps ADWIN with production-friendly interface
  - Logs drift events with timestamps
  - Integrates with river library if available

**Example:**
```python
from bitcoin_scalper.validation import DriftScanner

scanner = DriftScanner(delta=0.002)

# Monitor predictions online
for y_true, y_pred in trading_stream:
    error = abs(y_true - y_pred)
    if scanner.scan(error, timestamp):
        print("Drift detected! Stop trading and retrain.")
```

#### 3. Event-Driven Backtesting (`backtest.py`)

Realistic backtesting engine with:

- Event-driven architecture (no look-ahead bias)
- Integration with position sizing from risk module
- Realistic slippage and commission simulation
- Comprehensive performance metrics:
  - Sharpe Ratio
  - Sortino Ratio (downside-only volatility)
  - Maximum Drawdown
  - Calmar Ratio
  - Win Rate
  - Trade statistics

**Example:**
```python
from bitcoin_scalper.validation import Backtester
from bitcoin_scalper.risk import KellySizer

# Setup backtester with position sizer
backtester = Backtester(
    initial_capital=10000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    position_sizer=KellySizer(kelly_fraction=0.5)
)

# Run backtest
results = backtester.run(
    prices=price_series,
    signals=ml_signals,
    signal_params={'win_prob': 0.6, 'payoff_ratio': 2.0}
)

print(results.summary())
```

## Risk Management Module (`src/bitcoin_scalper/risk/`)

### Components

#### 1. Position Sizing (`sizing.py`)

Mathematical methods for determining "how much to trade":

- **KellySizer**: Kelly Criterion with fractional Kelly support
  - Maximizes long-term growth rate
  - Fractional Kelly reduces volatility (recommended: 0.25-0.5)
  - Accounts for win probability and payoff ratio
  - Includes max leverage caps

- **TargetVolatilitySizer**: Target volatility approach
  - Maintains consistent portfolio volatility
  - Automatically adjusts for asset volatility
  - Deleverages in high volatility (protects capital)
  - Leverages in low volatility (maximizes returns)

**Example:**
```python
from bitcoin_scalper.risk import KellySizer, TargetVolatilitySizer

# Kelly Criterion (based on edge)
kelly = KellySizer(kelly_fraction=0.5)
size = kelly.calculate_size(
    capital=10000,
    price=50000,
    win_prob=0.60,
    payoff_ratio=2.0
)

# Target Volatility (based on risk)
vol_sizer = TargetVolatilitySizer(target_volatility=0.40)
size = vol_sizer.calculate_size(
    capital=10000,
    price=50000,
    asset_volatility=0.80  # 80% annual vol
)
```

## Integration with Existing Modules

### With ML Models (Section 3)

```python
from bitcoin_scalper.models import GradientBoostingTrainer
from bitcoin_scalper.validation import PurgedKFold, Backtester
from bitcoin_scalper.risk import KellySizer

# Train with purged CV
cv = PurgedKFold(n_splits=5)
model = GradientBoostingTrainer()
# ... train and validate

# Backtest with position sizing
backtester = Backtester(position_sizer=KellySizer(kelly_fraction=0.5))
results = backtester.run(prices, model.predict(X))
```

### With RL Agents (Section 4)

```python
from bitcoin_scalper.rl import TradingEnv, RLAgentFactory
from bitcoin_scalper.validation import Backtester, DriftScanner
from bitcoin_scalper.risk import TargetVolatilitySizer

# Train RL agent
env = TradingEnv(df, reward_mode='sortino')
factory = RLAgentFactory(env, agent_type='ppo')
factory.train(total_timesteps=100000)

# Backtest RL signals
backtester = Backtester(position_sizer=TargetVolatilitySizer(target_volatility=0.40))
signals = agent.predict(observations)
results = backtester.run(prices, signals)

# Monitor in production
scanner = DriftScanner()
for error in prediction_errors:
    if scanner.scan(error):
        # Retrain agent
```

### With Triple Barrier Labeling (Section 2)

```python
from bitcoin_scalper.labeling import get_events
from bitcoin_scalper.validation import PurgedKFold

# Generate labels with Triple Barrier
events = get_events(
    close=prices,
    timestamps=signal_times,
    pt_sl=0.02,
    max_holding_period=pd.Timedelta('15min')
)

# Use event end times (t1) for purging
t1 = events['t1']

# Cross-validate with purging
cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
for train_idx, test_idx in cv.split(X, y, t1=t1):
    # Train/test without look-ahead bias
    pass
```

## Key Features

### Scientific Rigor
- ✅ Prevents look-ahead bias through purging
- ✅ Eliminates serial correlation with embargo
- ✅ Generates performance distributions (not point estimates)
- ✅ Compatible with scikit-learn ecosystem

### Realistic Backtesting
- ✅ Event-driven architecture
- ✅ Realistic slippage and commissions
- ✅ Position sizing integration
- ✅ Comprehensive performance metrics
- ✅ Works with both ML and RL signals

### Risk Management
- ✅ Kelly Criterion (mathematically optimal)
- ✅ Target volatility (risk-based)
- ✅ Fractional sizing (reduces volatility)
- ✅ Max leverage caps

### Production Monitoring
- ✅ Online drift detection
- ✅ Memory efficient
- ✅ Automatic window adaptation
- ✅ Event logging with timestamps

## Testing

All modules have comprehensive test coverage:

```bash
# Run validation tests
pytest tests/validation/ -v

# Run risk tests
pytest tests/risk/ -v

# All together
pytest tests/validation/ tests/risk/ -v
```

Current test coverage:
- Validation module: 19/19 tests passing
- Risk module: 17/17 tests passing
- Total: **36/36 tests passing** ✅

## Examples

See `examples/validation_risk_integration.py` for complete integration examples:

1. Purged cross-validation with sklearn
2. Position sizing with Kelly Criterion
3. Full backtest with realistic costs
4. Online drift detection

Run examples:
```bash
PYTHONPATH=src python examples/validation_risk_integration.py
```

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
   - Chapter 7: Cross-Validation in Finance
   - Chapter 10: Bet Sizing

2. Kelly, J. L. (1956). "A New Interpretation of Information Rate". Bell System Technical Journal.

3. Bifet, A., & Gavaldà, R. (2007). "Learning from Time-Changing Data with Adaptive Windowing". SDM.

4. Bailey, D. H., et al. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality". Journal of Portfolio Management.

## Status Update to CHECKLIST_ML_TRADING_BITCOIN.md

### Section 5: VALIDATION & BACKTESTING

- ✅ **5.1 Combinatorial Purged Cross-Validation (CPCV)** - Fully implemented
  - ✅ Purging logic to prevent look-ahead bias
  - ✅ Embargo to eliminate serial correlation
  - ✅ Combinatorial splits for performance distributions
  - ✅ Scikit-learn compatible API

- ✅ **5.2 Drift Detection** - Fully implemented
  - ✅ ADWIN algorithm (lightweight implementation)
  - ✅ Online monitoring capabilities
  - ✅ River library integration (optional)
  - ✅ Drift event logging

### Section 6.1: RISK MANAGEMENT - Position Sizing

- ✅ **Kelly Criterion (Fractional)** - Fully implemented
  - ✅ Full and fractional Kelly support
  - ✅ Max leverage caps
  - ✅ Model confidence interface

- ✅ **Target Volatility Sizing** - Fully implemented
  - ✅ Volatility-based position sizing
  - ✅ Automatic volatility estimation
  - ✅ Max leverage caps

- ✅ **Backtesting Engine** - Fully implemented
  - ✅ Event-driven architecture
  - ✅ ML/RL signal integration
  - ✅ Position sizing integration
  - ✅ Realistic slippage and commissions
  - ✅ Comprehensive metrics (Sharpe, Sortino, Max DD)

## Next Steps

These modules complete the validation and risk management infrastructure. Recommended next steps:

1. **Integration Testing**: Test modules with live ML models and RL agents
2. **Performance Optimization**: Profile and optimize for production use
3. **Documentation**: Add more examples and use cases
4. **Advanced Features**:
   - Implement Sortino Ratio as reward function (Section 4.3)
   - Add Meta-Labeling support (Section 2.2)
   - Implement fractional differentiation (Section 1.2.1)

## License

See project LICENSE file.
