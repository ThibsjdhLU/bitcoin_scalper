# Meta Threshold Parameter Flow - Technical Documentation

## Overview

This document explains how the `meta_threshold` parameter flows through the trading system and how the refactoring ensures **engine_config.yaml is the single source of truth**.

## Problem Statement (Before Refactoring)

Previously, the `meta_threshold` parameter could be silently ignored:

1. **engine_config.yaml** specified `meta_threshold: 0.53`
2. **TradingConfig** had a default of `meta_threshold: 0.6`
3. **TradingEngine** had a default of `meta_threshold: 0.6`
4. **MetaModel** (when loaded from .pkl) used its saved threshold (e.g., `0.5`)

**The Issue**: When loading a pre-trained MetaModel from a `.pkl` file, the system used the threshold value that was pickled with the model, completely ignoring the user's configuration in `engine_config.yaml`.

## Solution (After Refactoring)

The refactoring ensures that `engine_config.yaml` is the **absolute source of truth** for `meta_threshold`, overriding:
- Default values in code
- Values stored in pickled model files

## Data Flow

### 1. Configuration Loading (YAML → TradingConfig)

```yaml
# config/engine_config.yaml
trading:
  mode: ml
  model_type: catboost
  model_path: models/meta_model_production.pkl
  symbol: BTC/USDT
  timeframe: 1m
  meta_threshold: 0.53  # ← USER'S VALUE
```

```python
# Load configuration
config = TradingConfig.from_yaml('config/engine_config.yaml')
print(config.meta_threshold)  # → 0.53
```

**Key Code**: `src/bitcoin_scalper/core/config.py`
```python
@dataclass
class TradingConfig:
    meta_threshold: float = 0.6  # Default (used only if not in YAML)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingConfig':
        flat = {}
        if 'trading' in data:
            flat.update(data['trading'])  # Includes meta_threshold: 0.53
        # ...
        return cls(**filtered)
```

### 2. Engine Initialization (TradingConfig → TradingEngine)

```python
# engine_main.py - run_live_mode() or run_paper_mode()

engine = TradingEngine(
    connector=connector,
    mode=mode,
    symbol=config.symbol,
    timeframe=config.timeframe,
    # ... other params ...
    meta_threshold=config.meta_threshold  # ← Pass config value (0.53)
)
```

**Key Change**: The engine now receives the threshold from config, not using its default.

**Key Code**: `src/bitcoin_scalper/core/engine.py`
```python
def __init__(
    self,
    connector,
    # ... other params ...
    meta_threshold: float = 0.6,  # Default (only used if not passed)
):
    # ...
    self.meta_threshold = meta_threshold  # Store config value
    self.logger.info(f"Meta-labeling threshold: {meta_threshold:.2f}")
```

### 3. Model Loading (TradingEngine → MetaModel)

```python
# engine_main.py - run_live_mode() or run_paper_mode()

if mode == TradingMode.ML:
    success = engine.load_ml_model(
        config.model_path,
        meta_threshold=config.meta_threshold  # ← Pass config value (0.53)
    )
```

**Key Change**: The `load_ml_model()` now accepts `meta_threshold` parameter.

**Key Code**: `src/bitcoin_scalper/core/engine.py`
```python
def load_ml_model(
    self,
    model_path: str,
    features_list: Optional[List[str]] = None,
    meta_threshold: Optional[float] = None  # ← NEW PARAMETER
):
    """Load ML model with optional meta_threshold override."""
    
    # Load the model from disk
    loaded_model = joblib.load(model_path)
    
    if isinstance(loaded_model, MetaModel):
        # CRITICAL: Override threshold from config
        threshold_to_use = meta_threshold if meta_threshold is not None else self.meta_threshold
        original_threshold = loaded_model.meta_threshold
        
        if threshold_to_use != original_threshold:
            self.logger.warning(
                f"⚠️  Overriding MetaModel threshold: "
                f"{original_threshold:.2f} → {threshold_to_use:.2f} "
                f"(from engine_config.yaml)"
            )
            loaded_model.meta_threshold = threshold_to_use  # ← OVERRIDE HERE
        
        self.ml_model = loaded_model
        # ...
```

### 4. Prediction (MetaModel uses overridden threshold)

```python
# During trading, when making predictions:
result = meta_model.predict_meta(X)

# Inside MetaModel.predict_meta():
def predict_meta(self, X):
    # ...
    # Step 4: Filter signals based on meta confidence
    final_signal = np.where(
        meta_conf >= self.meta_threshold,  # ← Uses overridden value (0.53)
        raw_signal,  # Keep signal if confident
        0            # Set to neutral if not confident
    )
    # ...
```

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ engine_config.yaml                                              │
│   trading:                                                       │
│     meta_threshold: 0.53  ← SINGLE SOURCE OF TRUTH             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TradingConfig.from_yaml()                                       │
│   config.meta_threshold = 0.53                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TradingEngine(meta_threshold=config.meta_threshold)             │
│   engine.meta_threshold = 0.53                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ engine.load_ml_model(meta_threshold=config.meta_threshold)      │
│   1. Load model from .pkl (threshold might be 0.5)              │
│   2. Override: model.meta_threshold = 0.53  ← CRITICAL STEP    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ model.predict_meta(X)                                           │
│   Uses self.meta_threshold = 0.53  ✅                          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Changes Summary

| File | Change | Purpose |
|------|--------|---------|
| `engine.py` | Added `meta_threshold` param to `load_ml_model()` | Accept threshold from config |
| `engine.py` | Override `loaded_model.meta_threshold` after loading | Force config value on loaded model |
| `engine_main.py` | Pass `config.meta_threshold` to `load_ml_model()` | Propagate config value |
| `engine_main.py` | Pass `config.meta_threshold` to `TradingEngine()` | Ensure engine has correct value |

## Testing

Run the test suite to verify the behavior:

```bash
cd /home/runner/work/bitcoin_scalper/bitcoin_scalper
python tests/core/test_meta_threshold_override.py
```

Expected output:
```
✅ TEST PASSED: meta_threshold override from config
✅ TEST PASSED: Complete meta_threshold flow from YAML
```

## Usage Example

### Before (Incorrect - Threshold Ignored)

```yaml
# engine_config.yaml
trading:
  meta_threshold: 0.53
```

```python
# Model was trained and saved with threshold=0.5
model = MetaModel(..., meta_threshold=0.5)
joblib.dump(model, "model.pkl")

# Later, when loading...
engine.load_ml_model("model.pkl")
# ❌ PROBLEM: Model uses 0.5, ignoring config's 0.53
```

### After (Correct - Config is Source of Truth)

```yaml
# engine_config.yaml
trading:
  meta_threshold: 0.53
```

```python
# Model was trained and saved with threshold=0.5
model = MetaModel(..., meta_threshold=0.5)
joblib.dump(model, "model.pkl")

# Later, when loading...
config = TradingConfig.from_yaml("engine_config.yaml")
engine = TradingEngine(..., meta_threshold=config.meta_threshold)
engine.load_ml_model("model.pkl", meta_threshold=config.meta_threshold)
# ✅ SUCCESS: Model now uses 0.53 from config
# Log: "⚠️  Overriding MetaModel threshold: 0.50 → 0.53 (from engine_config.yaml)"
```

## Benefits

1. **Predictability**: Users can change `meta_threshold` in config without retraining models
2. **Transparency**: Warning logs show when threshold is overridden
3. **Flexibility**: Easy to A/B test different thresholds in production
4. **No Silent Failures**: System explicitly overrides and logs the change

## Notes for Future Development

- If you add new parameters that should be configurable, follow this same pattern
- Always pass config values through the chain: YAML → TradingConfig → TradingEngine → Model
- For loaded models, explicitly override parameters after loading
- Add warning logs when overriding values from saved models
- Write tests to verify the override behavior

## Related Files

- `config/engine_config.yaml` - User configuration
- `src/bitcoin_scalper/core/config.py` - Configuration loading
- `src/bitcoin_scalper/core/engine.py` - Engine and model loading
- `src/bitcoin_scalper/engine_main.py` - Main entry points
- `src/bitcoin_scalper/models/meta_model.py` - MetaModel implementation
- `tests/core/test_meta_threshold_override.py` - Test suite
