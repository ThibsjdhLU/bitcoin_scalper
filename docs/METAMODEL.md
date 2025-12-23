# MetaModel Documentation

## Overview

The `MetaModel` class implements a modern meta-labeling strategy for trading signal generation, adapted from the legacy `MetaLabelingPipeline`. It provides a two-stage prediction system that significantly improves trading performance by filtering false signals.

## Concept

Meta-labeling is a powerful machine learning technique that improves trading strategy performance by combining two models:

1. **Primary Model (Stage 1)**: Predicts trade direction
   - Output: Buy (1), Sell (-1), or Neutral (0)
   - Goal: Identify potential trading opportunities
   - Uses all market features (price, volume, indicators, etc.)

2. **Meta Model (Stage 2)**: Predicts trade success probability
   - Output: Success probability (0 to 1)
   - Goal: Filter false positives from primary model
   - Uses original features + primary model probabilities

The meta model learns which market conditions lead to successful vs. failed primary predictions, effectively creating a "confidence filter" for trading signals.

## Key Benefits

- **Improved Sharpe Ratio**: By filtering low-confidence signals
- **Reduced False Positives**: Meta model learns to recognize when primary model is unreliable
- **Better Risk Management**: Confidence scores can be used for position sizing
- **Adaptability**: Learns market regimes where primary model performs well/poorly

## Architecture

```
Market Data (X)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
Primary Model                    Meta Model Training
(Direction)                      (on failures/successes)
    │                                     │
    ├──── Predictions ────┐              │
    │                     │              │
    └──── Probabilities ──┼──────────────┘
                          │
                          ▼
                   Meta Model Prediction
                   (Success Probability)
                          │
                          ▼
                   Threshold Filter
                   (meta_conf >= threshold)
                          │
                          ▼
                   Final Signal
                (filtered direction)
```

## Installation

The `MetaModel` class is located in `src/bitcoin_scalper/models/meta_model.py` and can be imported as:

```python
from bitcoin_scalper.models.meta_model import MetaModel
```

### Dependencies

- `numpy`: Array operations
- `pandas`: DataFrame handling
- `catboost` (optional but recommended): High-performance gradient boosting
- Any classifier with `.fit()`, `.predict()`, `.predict_proba()` methods

## Usage

### Basic Example

```python
from catboost import CatBoostClassifier
from bitcoin_scalper.models.meta_model import MetaModel
import pandas as pd

# 1. Prepare your data
X_train = pd.DataFrame({...})  # Market features
y_direction = [1, -1, 0, 1, ...]  # Direction labels
y_success = [1, 0, 1, 1, ...]     # Success labels (did trade work?)

# 2. Create models
primary = CatBoostClassifier(iterations=100, depth=6, verbose=False)
meta = CatBoostClassifier(iterations=50, depth=4, verbose=False)

# 3. Create MetaModel
model = MetaModel(
    primary_model=primary,
    meta_model=meta,
    meta_threshold=0.6  # Only take trades with 60%+ confidence
)

# 4. Train
model.train(X_train, y_direction, y_success)

# 5. Predict with filtering
result = model.predict_meta(X_test)

print(result['final_signal'])  # Filtered signals
print(result['meta_conf'])     # Confidence scores
print(result['raw_signal'])    # Original signals (for analysis)
```

### Integration with TradingEngine

```python
# In your engine.py or trading bot

class TradingEngine:
    def __init__(self, use_meta_labeling=True):
        self.use_meta_labeling = use_meta_labeling
        
        if use_meta_labeling:
            # Initialize meta model
            primary = CatBoostClassifier(...)
            meta = CatBoostClassifier(...)
            self.meta_model = MetaModel(primary, meta, meta_threshold=0.6)
            # Load or train
            self.meta_model.train(X, y_direction, y_success)
    
    def get_trading_signal(self, features):
        if self.use_meta_labeling:
            # Use meta-filtered signals
            result = self.meta_model.predict_meta(features)
            signal = result['final_signal'][0]
            confidence = result['meta_conf'][0]
        else:
            # Use regular model
            signal = self.ml_model.predict(features)[0]
            confidence = self.ml_model.predict_proba(features)[0].max()
        
        return signal, confidence
```

## API Reference

### Constructor

```python
MetaModel(primary_model, meta_model, meta_threshold=0.5)
```

**Parameters:**
- `primary_model`: Model for direction prediction (Buy/Sell/Neutral)
- `meta_model`: Model for success prediction (0/1)
- `meta_threshold` (float): Confidence threshold for signal filtering (default: 0.5)
  - Higher values = more conservative (fewer but higher quality trades)
  - Lower values = more aggressive (more trades, potentially lower quality)

### Methods

#### `train()`

```python
train(X, y_direction, y_success, sample_weights=None, eval_set=None, **kwargs)
```

Train both primary and meta models.

**Parameters:**
- `X`: Features (DataFrame or ndarray)
- `y_direction`: Direction labels (-1, 0, 1)
- `y_success`: Success labels (0, 1)
- `sample_weights`: Optional sample weights (from Triple Barrier method)
- `eval_set`: Optional validation set (X_val, y_dir_val, y_succ_val)
- `**kwargs`: Additional parameters passed to model training

**Returns:** `self` (for method chaining)

**Example:**
```python
model.train(
    X_train, 
    y_direction, 
    y_success,
    sample_weights=barrier_weights,
    eval_set=(X_val, y_dir_val, y_succ_val),
    early_stopping_rounds=20  # Passed to CatBoost
)
```

#### `predict_meta()`

```python
predict_meta(X, return_all=False)
```

Make filtered predictions using meta-labeling.

**Parameters:**
- `X`: Features to predict on
- `return_all` (bool): If True, return additional diagnostic info

**Returns:** Dictionary with:
- `final_signal`: Filtered signals (-1, 0, 1) after meta filtering
- `meta_conf`: Meta model confidence scores (0 to 1)
- `raw_signal`: Original primary predictions before filtering
- `primary_proba`: (if return_all=True) Primary probability distributions
- `meta_proba`: (if return_all=True) Meta probability distributions

**Example:**
```python
result = model.predict_meta(X_test, return_all=True)

# Access results
signals = result['final_signal']      # [-1, 0, 1, 0, 1, ...]
confidences = result['meta_conf']     # [0.7, 0.4, 0.8, 0.3, 0.9, ...]
raw_signals = result['raw_signal']    # [-1, 1, 1, -1, 1, ...]

# Count filtered trades
n_filtered = (raw_signals != 0).sum() - (signals != 0).sum()
print(f"Filtered {n_filtered} low-confidence trades")
```

#### `save()` and `load()`

```python
save(primary_path, meta_path)
load(primary_path, meta_path)
```

Save/load both models to/from disk.

**Example:**
```python
# Save
model.save('models/primary.cbm', 'models/meta.cbm')

# Load
new_model = MetaModel(CatBoostClassifier(), CatBoostClassifier())
new_model.load('models/primary.cbm', 'models/meta.cbm')
```

## Advanced Topics

### Choosing the Meta Threshold

The `meta_threshold` parameter controls trade filtering aggressiveness:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3 - 0.4 | Aggressive | High-frequency trading, bull markets |
| 0.5 - 0.6 | Balanced | General trading (recommended default) |
| 0.7 - 0.8 | Conservative | Risk-averse, volatile markets |
| 0.9+ | Very conservative | Only extremely confident trades |

**Optimization Strategy:**
1. Start with 0.5 (balanced)
2. Backtest with different thresholds
3. Choose based on your Sharpe ratio and drawdown tolerance
4. Consider adaptive thresholds based on market regime

### Label Generation for Training

#### Direction Labels (y_direction)

Generate using Triple Barrier method or similar:

```python
from bitcoin_scalper.labeling.barrier import get_events

# Generate barrier events
events = get_events(
    prices,
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=-0.01,  # 1% stop loss
    t_expiry=pd.Timedelta('1h')  # 1 hour max holding
)

y_direction = events['side']  # 1 (Buy), -1 (Sell), 0 (Neutral)
```

#### Success Labels (y_success)

Label trades as successful (1) or failed (0):

```python
# Option 1: Binary based on profit/loss
y_success = (events['return'] > 0).astype(int)

# Option 2: Threshold-based
y_success = (events['return'] > 0.005).astype(int)  # Only >0.5% is success

# Option 3: Risk-adjusted
y_success = (events['return'] / events['volatility'] > 1.0).astype(int)
```

### Sample Weights

Use Triple Barrier sample weights for better results:

```python
# Compute sample weights (inversely proportional to holding period)
sample_weights = 1.0 / events['holding_period'].dt.total_seconds()

# Train with weights
model.train(X, y_direction, y_success, sample_weights=sample_weights)
```

### Feature Engineering

The meta model benefits from rich features:

```python
# Multi-timeframe features (as in engine.py)
features = pd.DataFrame({
    # 1-minute timeframe
    '1min_close': ...,
    '1min_volume': ...,
    '1min_rsi': ...,
    '1min_macd': ...,
    
    # 5-minute timeframe
    '5min_close': ...,
    '5min_volume': ...,
    '5min_rsi': ...,
    '5min_macd': ...,
    
    # Market microstructure
    'spread': ...,
    'order_flow': ...,
    'volatility': ...,
})
```

## Performance Tips

1. **Use CatBoost**: Best performance for tabular trading data
2. **Proper Validation**: Use walk-forward or time-series cross-validation
3. **Feature Importance**: Check which features meta model uses most
4. **Monitor Drift**: Retrain when market regime changes
5. **Threshold Tuning**: Optimize on validation set, not training

## Comparison with Legacy

| Feature | Legacy MetaLabelingPipeline | New MetaModel |
|---------|----------------------------|---------------|
| CatBoost Support | Partial | Full native support |
| Error Handling | Basic | Comprehensive |
| Type Hints | None | Complete |
| Logging | Minimal | Detailed |
| Flexibility | BaseModel only | Any sklearn-compatible |
| Documentation | Limited | Extensive |
| Testing | Basic | Comprehensive (17 tests) |
| Integration | Complex | Simple with engine.py |

## Examples

See `examples/meta_model_integration.py` for a complete working example demonstrating:
- Data generation
- Model creation and training
- Prediction and analysis
- Integration patterns

Run with:
```bash
cd /home/runner/work/bitcoin_scalper/bitcoin_scalper
PYTHONPATH=src:$PYTHONPATH python examples/meta_model_integration.py
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 3: Meta-Labeling
- Chan, E. (2017). *Machine Trading*
- CatBoost documentation: https://catboost.ai/

## Support

For questions or issues, please refer to:
- Main project documentation
- Test suite in `tests/models/test_meta_model.py`
- Integration example in `examples/meta_model_integration.py`
