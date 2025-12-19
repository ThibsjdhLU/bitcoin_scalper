# ML Models Module

This module provides a unified interface for machine learning models used in the Bitcoin Scalper trading system.

## Architecture Overview

The module follows the **Model Factory** pattern with a consistent interface across all model types:

```
models/
├── __init__.py                  # Module exports
├── base.py                      # BaseModel abstract interface
├── gradient_boosting.py         # XGBoost & CatBoost implementations
├── pipeline.py                  # Training utilities & meta-labeling
├── deep_learning/              
    ├── __init__.py
    ├── torch_wrapper.py         # PyTorch model wrapper
    ├── lstm.py                  # LSTM implementation
    └── transformer.py           # Transformer placeholder
```

## Key Features

### 1. Unified Interface (`BaseModel`)
All models implement the same interface:
- `train(X, y, sample_weights, eval_set)` - Train with Triple Barrier weights
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get class probabilities (classification)
- `save(path)` / `load(path)` - Model persistence
- `get_feature_importance()` - Feature importance (if available)

### 2. Robust Data Handling
- Automatic NaN detection and handling
- Infinite value detection and clipping
- Feature count validation
- Sample weight support from Triple Barrier method

### 3. GPU Acceleration
- Automatic GPU detection for XGBoost (`tree_method='gpu_hist'`)
- CUDA support for PyTorch models
- Graceful fallback to CPU if GPU unavailable

### 4. Early Stopping
- All models support early stopping via `eval_set`
- Prevents overfitting automatically
- Monitors validation loss/accuracy

### 5. Meta-Labeling Pipeline
- Two-stage prediction (Side + Size)
- Significantly improves Sharpe ratio
- Filters false positives from primary model

## Quick Start

### Basic XGBoost Training

```python
from src.bitcoin_scalper.models import XGBoostClassifier, Trainer
from src.bitcoin_scalper.labeling import get_events

# Generate labels using Triple Barrier method
events = get_events(
    close=price_series,
    timestamps=signal_times,
    pt_sl=0.02,  # 2% barriers
    max_holding_period=pd.Timedelta('15min')
)

# Extract labels and weights
y = events['type']  # -1, 0, 1
weights = 1.0 / (events['t1'] - events.index).dt.total_seconds()  # Weight by holding time

# Create and train model
model = XGBoostClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_gpu=True
)

trainer = Trainer(model, handle_nans='fill', handle_infs='clip')
trainer.train(
    X_train, y,
    sample_weights=weights,
    eval_set=(X_val, y_val),
    early_stopping_rounds=20
)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Feature importance
importance = model.get_feature_importance()
print(importance.sort_values(ascending=False).head(10))
```

### Meta-Labeling Pipeline

```python
from src.bitcoin_scalper.models import XGBoostClassifier, MetaLabelingPipeline

# Create two models
primary = XGBoostClassifier(n_estimators=100)    # Predicts side (Long/Short)
secondary = XGBoostClassifier(n_estimators=50)   # Predicts bet/pass

# Create pipeline
pipeline = MetaLabelingPipeline(primary, secondary)

# Prepare labels
y_side = events['type']  # -1, 0, 1
y_meta = (events['return'] > 0).astype(int)  # 0 (fail), 1 (success)

# Train
pipeline.train(
    X_train, y_side, y_meta,
    sample_weights=weights,
    eval_set=(X_val, y_side_val, y_meta_val)
)

# Predict
side, bet = pipeline.predict(X_test)
# side: -1 (Short), 0 (Neutral), 1 (Long)
# bet: 0 (Pass), 1 (Bet)

# Combined prediction
final_signal = pipeline.predict_combined(X_test)
# -1: Short position, 0: No position, 1: Long position
```

### LSTM Training (PyTorch)

```python
from src.bitcoin_scalper.models.deep_learning import LSTMModel, TorchModelWrapper

# Create LSTM
lstm = LSTMModel(
    input_size=50,      # Number of features
    hidden_size=128,    # LSTM hidden units
    num_layers=2,       # Number of LSTM layers
    output_size=3,      # Number of classes (Long/Neutral/Short)
    dropout=0.2
)

# Wrap it
model = TorchModelWrapper(
    model=lstm,
    task_type='classification',
    learning_rate=0.001,
    use_gpu=True
)

# Train
model.train(
    X_train, y_train,
    eval_set=(X_val, y_val),
    epochs=50,
    batch_size=32,
    early_stopping_patience=10
)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Model Comparison

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| **XGBoost** | Tabular data, production | Fast, accurate, feature importance | No sequence modeling |
| **CatBoost** | Categorical features | Handles categories natively | Slower than XGBoost |
| **LSTM** | Sequential patterns | Temporal dependencies | Slow training, needs more data |
| **Transformer** | Long-range patterns | Parallel training, attention | Complex, resource-intensive |

## Integration with Triple Barrier Method

All models are designed to work with labels from the Triple Barrier method:

```python
from src.bitcoin_scalper.labeling import get_events

# Step 1: Generate barrier events
events = get_events(
    close=prices,
    timestamps=signal_times,
    pt_sl=volatility * 2,  # Dynamic barriers based on volatility
    max_holding_period=pd.Timedelta('15min')
)

# Step 2: Extract labels and weights
y = events['type']  # Which barrier was hit first

# Weight by inverse holding time (faster exits = higher weight)
weights = 1.0 / (events['t1'] - events.index).dt.total_seconds()
weights = weights / weights.sum()  # Normalize

# Step 3: Train model
model.train(X, y, sample_weights=weights, eval_set=(X_val, y_val))
```

## Hyperparameter Tuning with Optuna

The models are designed to work with Optuna for hyperparameter tuning:

```python
import optuna
from src.bitcoin_scalper.models import XGBoostClassifier

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    # Create and train model
    model = XGBoostClassifier(**params)
    model.train(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate
    predictions = model.predict(X_val)
    accuracy = (predictions == y_val).mean()
    
    return accuracy

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

## Model Persistence

```python
# Save model
model.save('models/xgboost_primary.json')

# Load model
model = XGBoostClassifier()
model.load('models/xgboost_primary.json')

# Meta-labeling pipeline
pipeline.save('models/primary.json', 'models/secondary.json')
pipeline.load('models/primary.json', 'models/secondary.json')
```

## Future Enhancements

### Coming Soon
- **Transformer Model**: Full implementation with positional encoding
- **Transformer-XGBoost Hybrid**: Extract embeddings from Transformer, feed to XGBoost
- **State Space Models (Mamba)**: For very long sequences
- **Ensemble Methods**: Voting, stacking, blending

### Planned Features
- Automatic feature selection (RFE, SHAP)
- Model monitoring and drift detection
- A/B testing framework
- Model versioning with MLflow

## Best Practices

### 1. Always Use Sample Weights
```python
# Good: Weight samples by importance
weights = 1.0 / holding_times
model.train(X, y, sample_weights=weights)

# Bad: Ignore sample weights
model.train(X, y)  # All samples treated equally
```

### 2. Use Early Stopping
```python
# Good: Prevent overfitting
model.train(X, y, eval_set=(X_val, y_val), early_stopping_rounds=20)

# Bad: No validation, risk overfitting
model.train(X, y)
```

### 3. Handle Data Quality
```python
# Good: Robust to bad data
trainer = Trainer(model, handle_nans='fill', handle_infs='clip')
trainer.train(X, y)

# Bad: Crash on bad data
model.train(X, y)  # May raise error on NaN/inf
```

### 4. Use Meta-Labeling for Production
```python
# Good: Two-stage filtering
pipeline = MetaLabelingPipeline(primary, secondary)
side, bet = pipeline.predict(X)
final_signal = side * bet  # Only trade when both agree

# Bad: Single model, more false signals
signal = primary.predict(X)
```

### 5. Monitor Feature Importance
```python
# Good: Understand model behavior
importance = model.get_feature_importance()
print(importance.sort_values(ascending=False).head(20))

# If importance looks wrong, debug features
if importance['random_noise'] > 0.1:
    print("Warning: Model overfitting to noise!")
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 3: Labeling (Triple Barrier Method)
  - Chapter 6: Ensemble Methods (Meta-Labeling)
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD 2016.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural computation.
- Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS 2017.

## Support

For issues or questions:
1. Check the docstrings in the code
2. Review the examples in `examples/` directory
3. See CHECKLIST_ML_TRADING_BITCOIN.md for implementation status
