# Data Requirements for Feature Engineering

## Overview

The Bitcoin Scalper trading system uses multi-timeframe feature engineering that requires a minimum amount of historical data to function properly. This document explains why these requirements exist and how to ensure you have sufficient data.

## The Problem

**Error**: `index 6 is out of bounds for axis 0 with size 1`

This error occurs when the feature engineering pipeline doesn't have enough historical data to:
1. Calculate technical indicators that require lookback periods (e.g., SMA-200 needs 200 candles)
2. Resample 1-minute data to 5-minute timeframe
3. Remove NaN values from warm-up periods
4. Maintain sufficient data after all transformations

## Minimum Data Requirements

### Individual Indicator Windows

| Indicator | Lookback Period | Description |
|-----------|----------------|-------------|
| **EMA/SMA** | **200** | Longest moving averages (21, 50, 200) |
| Z-scores | 100 | Rolling statistics (5, 20, 50, 100) |
| Rolling High/Low | 100 | Distance calculations |
| Ichimoku | 52 | Ichimoku Cloud (9, 26, 52) |
| Bollinger Bands | 50 | BB windows (20, 50) |
| MACD | 34 | MACD (12, 26, 9) |
| FracDiff | 23 | Fractional differentiation (d=0.4) |
| RSI | 21 | RSI windows (7, 14, 21) |
| ATR | 21 | ATR windows (14, 21) |
| Volume SMA | 20 | Volume moving average |
| Volatility | 20 | Rolling volatility |
| SuperTrend | 7 | SuperTrend indicator |

### Multi-Timeframe Calculation

The system generates features for **both** 1-minute and 5-minute timeframes:

1. **1-minute features**: Direct calculation on 1-minute data
2. **5-minute features**: Resample 1-minute to 5-minute, then calculate indicators
3. **Merge**: 5-minute features are forward-filled to align with 1-minute data

**Critical**: To get 200 valid 5-minute bars (for SMA-200), you need:
- 200 5-minute bars × 5 minutes/bar = **1000 1-minute bars minimum**

### NaN Removal Impact

After calculating all indicators:
- **FracDiff warm-up**: ~23 rows with NaN
- **SMA/EMA warm-up**: ~200 rows with NaN  
- **Other indicators**: Various NaN values
- **Total rows dropped**: ~200-300 (can be up to ~281 rows)

### Final Recommendation

**Minimum Required**: 1500 1-minute candles

This provides:
- 1000+ candles for 5-minute feature calculation
- 300+ valid rows after NaN removal (system minimum)
- Safety buffer for edge cases

## Implementation

### Default Behavior

All connectors now default to fetching **1500 candles**:

```python
from bitcoin_scalper.connectors.binance_connector import BinanceConnector

connector = BinanceConnector(api_key="...", api_secret="...")

# Default: 1500 candles
df = connector.fetch_ohlcv("BTC/USDT", "1m")

# Explicit override (not recommended unless you know what you're doing)
df = connector.fetch_ohlcv("BTC/USDT", "1m", limit=2000)
```

### Configuration

The `config/engine_config.yaml` includes a note about data requirements:

```yaml
# Data Requirements
# ⚠️ IMPORTANT: Feature engineering requires at least 1500 historical candles
# for multi-timeframe analysis (1min + 5min). The system will automatically
# fetch this amount, but you can override in code if needed.
```

### Validation

The system validates data at two stages:

**1. Pre-processing validation** (feature_engineering.py):
```python
# Warns if input data < 1500 rows
⚠️ Input data has only 500 rows, recommended minimum is 1500 rows.
   Some indicators may not have enough historical data.
```

**2. Post-processing validation** (feature_engineering.py):
```python
# Fails if output data < 300 rows
❌ Insufficient data after NaN removal: 150 rows (minimum: 300)
   Original rows: 500, dropped: 350 (70.0%)
   💡 SOLUTION: Increase fetch limit to at least 1500 candles.
```

### Error Handling

The engine (engine.py) checks for empty DataFrames after feature engineering:

```python
# Check after 1-minute feature engineering
if df.empty:
    return {
        'error': 'Insufficient data for 1-minute feature engineering',
        'reason': 'Need at least 1500 candles for proper indicator calculation'
    }

# Check after 5-minute feature engineering  
if df_5m.empty:
    return {
        'error': 'Insufficient data for 5-minute feature engineering',
        'reason': 'Need at least 1500 1-minute candles for proper calculation'
    }
```

## Troubleshooting

### Error: "Insufficient data after NaN removal"

**Cause**: Not enough historical data fetched initially

**Solution**:
```python
# Increase limit in connector fetch
df = connector.fetch_ohlcv(symbol, "1m", limit=2000)
```

### Error: "Feature engineering returned empty DataFrame"

**Cause**: Data validation failed, check logs for details

**Solution**:
1. Check connector is working properly
2. Verify exchange/broker has sufficient historical data
3. Increase fetch limit if needed
4. Check logs for specific validation failures

### Demo/Paper Trading Mode

Paper trading client generates synthetic data. Ensure it initializes with at least 1500 historical candles:

```python
paper_client = PaperMT5Client(
    initial_balance=10000.0,
    initial_history_size=1500  # Ensure enough history
)
```

### Dashboard Mode

The dashboard automatically fetches 1500 candles when launching the engine. If you see data errors:

1. Check the dashboard logs for fetch errors
2. Verify the connector configuration
3. Ensure the exchange API is accessible

## Code References

- **Constants**: `src/bitcoin_scalper/core/data_requirements.py`
- **Validation**: `src/bitcoin_scalper/core/feature_engineering.py` (add_indicators method)
- **Error Handling**: `src/bitcoin_scalper/core/engine.py` (process_tick method)
- **Connectors**:
  - `src/bitcoin_scalper/connectors/binance_connector.py`
  - `src/bitcoin_scalper/connectors/mt5_rest_client.py`
  - `src/bitcoin_scalper/connectors/paper.py`

## Best Practices

1. **Always use default limits**: Don't override unless you have a specific reason
2. **Monitor logs**: Check for validation warnings during startup
3. **Test with sufficient data**: When testing, ensure test data has 1500+ rows
4. **Production deployment**: Verify data availability before going live
5. **Paper trading**: Initialize with realistic history size

## Performance Considerations

Fetching 1500 candles instead of 100:
- **API calls**: No increase (single request)
- **Memory**: Minimal (~150KB for OHLCV data)
- **Processing time**: Slight increase (~100-200ms)
- **Reliability**: Significantly improved (no more data errors)

The trade-off is strongly in favor of fetching more data upfront.

## Future Improvements

Potential enhancements for data management:
1. **Caching**: Store historical data locally to reduce API calls
2. **Incremental updates**: Fetch only new candles after initial load
3. **Adaptive limits**: Automatically adjust based on indicator requirements
4. **Streaming mode**: Maintain rolling window of historical data

## Summary

- **Minimum required**: 1500 1-minute candles
- **Default behavior**: All connectors fetch 1500 candles
- **Validation**: Automatic checks at pre/post-processing stages
- **Error handling**: Clear messages with actionable solutions
- **Configuration**: Documented in engine_config.yaml
- **Backwards compatible**: Explicit limit parameter still works

For questions or issues, refer to the code references above or check the logs for detailed error messages.
