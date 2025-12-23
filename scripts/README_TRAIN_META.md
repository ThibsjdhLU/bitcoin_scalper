# MetaModel Training Script

## Overview

The `train_meta_model.py` script provides a complete production-ready pipeline for training a meta-labeling model for Bitcoin trading.

## Architecture

**Two-Stage Meta-Labeling:**
1. **Primary Model (CatBoost)**: Predicts direction (Buy=1, Sell=-1, Neutral=0)
2. **Meta Model (CatBoost)**: Predicts success probability (Take=1, Pass=0)

The meta model learns which market conditions produce reliable primary predictions, filtering ~40-60% of low-confidence trades to improve Sharpe ratio.

## Usage

### Basic Usage
```bash
python scripts/train_meta_model.py
```

### Custom Configuration
```bash
python scripts/train_meta_model.py \
    --csv data/raw/custom_data.csv \
    --threshold 0.7 \
    --primary_iterations 300 \
    --meta_iterations 150 \
    --output models/meta_custom.pkl
```

### Help
```bash
python scripts/train_meta_model.py --help
```

## Command Line Arguments

### Data Arguments
- `--csv`: Path to OHLCV CSV file (default: `data/raw/BTCUSD_M1_202301010000_202512011647.csv`)
- `--test_size`: Test set size (default: 0.2)

### Primary Model Hyperparameters
- `--primary_iterations`: Number of iterations (default: 200)
- `--primary_depth`: Tree depth (default: 6)
- `--primary_lr`: Learning rate (default: 0.05)

### Meta Model Hyperparameters
- `--meta_iterations`: Number of iterations (default: 100)
- `--meta_depth`: Tree depth (default: 4)
- `--meta_lr`: Learning rate (default: 0.05)

### Configuration
- `--threshold`: Meta confidence threshold (default: 0.6)
  - Higher = more conservative (fewer but better trades)
  - Lower = more aggressive (more trades, potentially lower quality)

### Labeling Arguments
- `--horizon`: Prediction horizon in minutes (default: 15)
- `--label_k`: Label threshold multiplier (default: 0.5)

### Output Arguments
- `--output`: Output path for trained model (default: `models/meta_model_production.pkl`)
- `--verbose`: Enable verbose output

## Pipeline Stages

### Stage 1: Data Loading
- Loads OHLCV data from CSV
- Validates data integrity
- Fills missing values

### Stage 2: Feature Engineering
- Computes log returns
- Generates technical indicators (RSI, MACD, BB, ATR, etc.)
- Creates derived features
- Drops NaN values

### Stage 3: Label Generation
- **Direction labels**: Based on future price movement
  - BUY (1): Price expected to rise
  - SELL (-1): Price expected to fall
  - NEUTRAL (0): No clear direction
- **Success labels**: Whether trade would be profitable
  - SUCCESS (1): Trade direction matches future movement
  - FAILED (0): Trade direction opposes future movement

### Stage 4: Train/Test Split
- Chronological split (respects temporal order)
- 80/20 split by default
- Ensures no data leakage

### Stage 5: Model Training
- Creates CatBoost classifiers with optimized hyperparameters
- Trains primary model on direction labels
- Generates primary predictions for training set
- Augments features with primary probabilities
- Trains meta model on success labels
- Uses early stopping to prevent overfitting

### Stage 6: Model Evaluation
Comprehensive evaluation report including:
- Primary model accuracy
- Meta filtering statistics (# trades filtered)
- Filtered model accuracy
- Success rate analysis (before/after filtering)
- Confidence distribution
- Performance improvement metrics

### Stage 7: Model Saving
- Saves complete MetaModel object as pickle
- Saves metadata (JSON)
- Ready for deployment in TradingEngine

## Output

### Model File
The trained model is saved as a joblib pickle file:
```
models/meta_model_production.pkl
```

### Metadata File
A JSON file with model metadata:
```
models/meta_model_production.json
```

Contains:
- Model type
- Meta threshold
- Number of features
- Feature names
- Training date
- Model architectures

### Log File
Training logs are saved to:
```
logs/train_meta_YYYYMMDD_HHMMSS.log
```

## Example Output

```
======================================================================
🚀 MetaModel Training Pipeline
======================================================================

📊 STEP 1: DATA LOADING
======================================================================
Loading data from: data/raw/BTCUSD_M1_202301010000_202512011647.csv
✅ Loaded 524,288 rows
   Date range: 2023-01-01 00:00:00 to 2025-12-01 16:47:00

🔧 STEP 2: FEATURE ENGINEERING
======================================================================
Computing log returns...
Computing technical indicators...
Computing derived features...
✅ Generated features for 524,200 rows (dropped 88 NaN rows)
   Total features: 127

🏷️ STEP 3: LABEL GENERATION
======================================================================
Generating direction labels (horizon=15, k=0.5)...
Direction label distribution:
   SELL     (-1):  98,234 (18.8%)
   NEUTRAL  ( 0): 328,150 (62.6%)
   BUY      ( 1):  97,816 (18.6%)

Success label distribution:
   FAILED   (0):  85,432 (16.3%)
   SUCCESS  (1): 438,768 (83.7%)

✂️ STEP 4: TRAIN/TEST SPLIT
======================================================================
Train set: 419,360 samples (80%)
Test set:  104,840 samples (20%)
Features: 127

🤖 STEP 5: MODEL TRAINING
======================================================================
Creating Primary Model (Direction Prediction)...
Creating Meta Model (Success Prediction)...
Creating MetaModel (threshold=0.60)...
🚀 Starting training...
   Stage 1: Primary model (Direction)
   Stage 2: Meta model (Success)
✅ Training completed successfully!

📊 STEP 6: MODEL EVALUATION
======================================================================

PRIMARY MODEL PERFORMANCE (Direction Prediction)
----------------------------------------------------------------------
Accuracy: 0.6523 (65.23%)

META MODEL FILTERING
----------------------------------------------------------------------
Signal Statistics:
   Raw signals (Primary):
      BUY:     19,863 (18.9%)
      SELL:    19,425 (18.5%)
      NEUTRAL: 65,552 (62.5%)
      Total trades: 39,288

   Final signals (After Meta Filter):
      BUY:     11,234 (10.7%)
      SELL:    10,986 (10.5%)
      NEUTRAL: 82,620 (78.8%)
      Total trades: 22,220

   📉 Filtered: 17,068 trades (43.4%)

META MODEL PERFORMANCE (On Filtered Trades)
----------------------------------------------------------------------
Primary accuracy on trades: 0.6523 (65.23%)
Filtered accuracy on trades: 0.7834 (78.34%)
🎯 Improvement: +13.11 percentage points

SUCCESS RATE ANALYSIS
----------------------------------------------------------------------
Success Rate by Signal (Raw vs Filtered):
   SELL:
      Raw:       62.3%
      Filtered:  81.7% (+19.4pp)
   BUY:
      Raw:       63.1%
      Filtered:  82.5% (+19.4pp)

   Overall success rate (Raw trades):      62.7%
   Overall success rate (Filtered trades): 82.1%
   🎯 Improvement: +19.4 percentage points

💾 STEP 7: MODEL SAVING
======================================================================
Saving MetaModel to: models/meta_model_production.pkl
✅ Model saved successfully!

======================================================================
✅ TRAINING COMPLETED SUCCESSFULLY
======================================================================
   Model saved to: models/meta_model_production.pkl
   Primary accuracy: 0.6523
   Trades filtered: 17,068 (43.4%)
   Final accuracy: 0.7834
======================================================================
```

## Integration with TradingEngine

After training, load the model in the engine:

```python
from bitcoin_scalper.core.engine import TradingEngine

# Initialize engine with meta threshold
engine = TradingEngine(
    connector=my_connector,
    mode=TradingMode.ML,
    meta_threshold=0.6
)

# Load trained MetaModel
engine.load_ml_model('models/meta_model_production.pkl')
# Output: ✅ Loaded MetaModel successfully (meta-labeling enabled)

# Predictions happen automatically with enhanced logging
```

## Performance Tips

1. **Threshold Tuning**: Start with 0.6, adjust based on backtest results
2. **Hyperparameter Tuning**: Use Optuna for automated optimization
3. **Feature Engineering**: Add domain-specific features for better performance
4. **Data Quality**: Ensure clean, complete OHLCV data
5. **Horizon Selection**: Match to your trading timeframe (15min = scalping)

## Troubleshooting

### "CSV file not found"
- Check that the CSV file exists at the specified path
- Use absolute or relative path from project root

### "Module not found"
- Ensure you're running from project root
- Check that all dependencies are installed: `pip install -r requirements.txt`

### "CatBoost not installed"
- Install CatBoost: `pip install catboost`

### Memory Issues
- Reduce iterations or depth
- Use smaller dataset for testing
- Increase available RAM

## References

- [MetaModel Documentation](../docs/METAMODEL.md)
- [Engine Integration](../ENGINE_INTEGRATION_SUMMARY.md)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 3: Meta-Labeling
