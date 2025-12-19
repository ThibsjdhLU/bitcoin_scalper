# Labeling Module

## Overview

The `labeling` module implements advanced labeling techniques for financial machine learning, specifically the **Triple Barrier Method** as described in López de Prado's "Advances in Financial Machine Learning" (2018).

This implementation addresses **Section 2: LABELS & TARGETS** of the ML Trading Bitcoin checklist.

## Why Triple Barrier Method?

Traditional labeling approaches (e.g., "did price go up in N minutes?") have significant limitations:

- **No risk management**: Doesn't consider stop-loss or take-profit levels
- **Fixed time horizons**: Ignores that profitable trades may materialize at different speeds
- **Unrealistic**: Doesn't reflect how actual trading works

The Triple Barrier Method solves these by defining three exit conditions:

1. **Upper Barrier (Take Profit)**: Price increases by threshold → Label: +1 (Long win)
2. **Lower Barrier (Stop Loss)**: Price decreases by threshold → Label: -1 (Short win)
3. **Vertical Barrier (Time Limit)**: Max holding period expires → Label: 0 (Neutral)

## Module Structure

```
labeling/
├── __init__.py          # Module exports
├── volatility.py        # Dynamic volatility estimation (EWMA)
├── barriers.py          # Triple Barrier Method implementation
└── labels.py            # Primary and meta-labeling functions
```

## Key Components

### 1. Volatility Estimation (`volatility.py`)

Dynamic volatility is used to set adaptive barriers that respond to market conditions.

```python
from bitcoin_scalper.labeling import estimate_ewma_volatility

# Estimate volatility using EWMA
volatility = estimate_ewma_volatility(
    prices,           # Price series or DataFrame
    span=100,         # EWMA span (higher = smoother)
    price_col='close' # Column name if DataFrame
)
```

**Features:**
- Exponential Weighted Moving Average (EWMA) for recent data emphasis
- Adaptive span recommendations based on market regime
- Handles both Series and DataFrame inputs

### 2. Triple Barrier Method (`barriers.py`)

Core labeling logic that determines which barrier is touched first.

```python
from bitcoin_scalper.labeling import get_events

# Apply Triple Barrier Method
events = get_events(
    close=prices,                          # Price series
    timestamps=signal_times,               # Event timestamps (e.g., from CUSUM filter)
    pt_sl=0.02,                           # 2% barriers (or dynamic from volatility)
    max_holding_period=pd.Timedelta('15min'), # Time limit
    side=None                              # Optional: {-1, 1} for short/long
)
```

**Output DataFrame contains:**
- `type`: Which barrier hit first (-1: SL, 0: Time, 1: TP)
- `t1`: Timestamp of barrier touch
- `return`: Actual return achieved
- `pt`: Profit target price level
- `sl`: Stop loss price level

**Key Functions:**
- `get_events()`: High-level interface (recommended)
- `apply_triple_barrier()`: Low-level implementation
- `get_vertical_barriers()`: Helper for time-based barriers

### 3. Label Generation (`labels.py`)

Convert barrier results into labels suitable for ML models.

#### Primary Labels

```python
from bitcoin_scalper.labeling import get_labels

# Generate labels from events
labels = get_labels(
    events,
    prices,
    label_type='fixed'  # 'fixed', 'sign', 'threshold', 'binary'
)
```

**Label Types:**
- `fixed`: Use barrier type directly {-1, 0, 1}
- `sign`: Use sign of actual return
- `threshold`: Require minimum return magnitude
- `binary`: Remove neutral labels (only {-1, 1})

#### Meta-Labeling

Meta-labeling trains a secondary model to filter false positives from a primary model.

```python
from bitcoin_scalper.labeling import get_meta_labels

# Generate meta-labels
meta_labels = get_meta_labels(
    events,
    prices,
    primary_model_predictions=primary_pred,  # {-1, 1}
    side_from_predictions=True
)
```

**Output:** Binary labels {0, 1}
- `1`: Primary model signal would be profitable → **TAKE TRADE**
- `0`: Primary model signal would be unprofitable → **FILTER OUT**

**Benefits:**
- Improves Sharpe ratio by filtering bad bets
- Reduces drawdowns
- Keeps winning bets from primary model

## Quick Start

### Basic Usage

```python
import pandas as pd
from bitcoin_scalper.labeling import (
    estimate_ewma_volatility,
    get_events,
    get_labels
)

# 1. Load your price data
prices = pd.Series([...], index=pd.DatetimeIndex([...]))

# 2. Calculate dynamic volatility
volatility = estimate_ewma_volatility(prices, span=100)

# 3. Define event timestamps (e.g., every 50 bars or from signals)
event_times = prices.index[::50]

# 4. Set dynamic barriers (2-sigma)
pt_sl = 2.0 * volatility.loc[event_times]

# 5. Apply Triple Barrier Method
events = get_events(
    close=prices,
    timestamps=event_times,
    pt_sl=pt_sl,
    max_holding_period=pd.Timedelta('15min')
)

# 6. Generate labels
labels = get_labels(events, prices, label_type='fixed')

# Now you can use these labels for supervised learning!
```

### With Existing Features

```python
# Assuming you have a DataFrame with features from Section 1
df = pd.DataFrame({
    'close': [...],
    'sma_20': [...],
    'rsi': [...],
    # ... other features
})

# Add volatility as a feature
df['volatility'] = estimate_ewma_volatility(df, price_col='close')

# Generate signals (example: SMA crossover)
df['signal'] = (df['close'] > df['sma_20']).astype(int)
signal_times = df[df['signal'] == 1].index

# Label the signals
events = get_events(
    close=df['close'],
    timestamps=signal_times,
    pt_sl=2.0 * df.loc[signal_times, 'volatility'],
    max_holding_period=pd.Timedelta('20min')
)

labels = get_labels(events, df['close'])

# Add labels to DataFrame
df['label'] = pd.NA
df.loc[labels.index, 'label'] = labels

# Now ready for model training with X = features, y = labels
```

## Integration with Existing Pipeline

The labeling module is designed to integrate seamlessly with the existing bitcoin_scalper codebase:

1. **Data Format Compatibility**: Works with DataFrame format from `data/preprocessing.py`
2. **Feature Engineering**: Volatility can be used as additional feature in `core/feature_engineering.py`
3. **Existing Labeling**: Complements (not replaces) `core/labeling.py` - provides more sophisticated alternative
4. **Model Training**: Labels work directly with `core/modeling.py` XGBoost/CatBoost training

## Testing

Comprehensive unit tests are provided in `tests/labeling/`:

```bash
# Run all labeling tests
pytest tests/labeling/ -v

# Run specific test files
pytest tests/labeling/test_barriers.py -v
pytest tests/labeling/test_volatility.py -v
```

**Test Coverage:**
- ✅ Volatility estimation (15 tests)
- ✅ Triple Barrier Method (13 tests)
- ✅ Label generation (8 tests)
- ✅ Edge cases (5 tests)
- ✅ Total: 41 tests, all passing

## Examples

See `examples/labeling_integration_example.py` for complete working examples:

```bash
python examples/labeling_integration_example.py
```

This demonstrates:
1. Basic Triple Barrier usage
2. Meta-labeling workflow
3. Integration with feature pipeline

## Configuration Recommendations

### Barrier Sizing

| Market Condition | Volatility Span | Barrier Multiplier | Holding Period |
|-----------------|-----------------|-------------------|----------------|
| High Volatility | 50              | 2.5-3.0 σ         | 10-15 min      |
| Normal          | 100             | 2.0 σ             | 15-20 min      |
| Low Volatility  | 200             | 1.5-2.0 σ         | 20-30 min      |

### Label Type Selection

| Use Case                    | Label Type  | Notes                           |
|-----------------------------|-------------|---------------------------------|
| General classification      | `fixed`     | Simplest, aligns with barriers  |
| Directional prediction only | `binary`    | Removes neutrals                |
| High-confidence signals     | `threshold` | Filters small moves             |
| Return prediction           | `sign`      | Uses actual returns             |

### Meta-Labeling Strategy

1. **Train primary model** on standard labels
2. **Generate predictions** on validation set
3. **Create meta-labels** based on profitability
4. **Train meta-model** (can use same features + primary probability)
5. **Production**: Only trade when both agree

## Performance Notes

- Vectorized operations using pandas/numpy for speed
- Efficient barrier detection with early stopping
- Memory-efficient for large datasets
- Typical performance: ~1000 events/second

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 3: Labeling.
2. RiskMetrics Technical Document (1996). J.P. Morgan/Reuters. (EWMA methodology)

## Limitations

- Requires clean, quality price data
- Vertical barrier needs sufficient future data
- Very small barriers may have numerical precision issues
- Not suitable for tick-level data without aggregation

## Future Enhancements

Potential extensions (not yet implemented):

- [ ] Asymmetric barriers (different PT/SL levels)
- [ ] Dynamic vertical barriers based on volatility
- [ ] Trend-following bias in barrier sizing
- [ ] Multi-timeframe labeling
- [ ] Sample weighting based on uniqueness

## License

Part of the bitcoin_scalper project. See main repository LICENSE.

---

**Status**: ✅ Implemented and tested  
**Checklist Section**: Section 2: LABELS & TARGETS  
**Last Updated**: 2024-12-19
