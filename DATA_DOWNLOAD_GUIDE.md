# Binance Historical Data Download Tool

## Overview

This tool enables downloading historical OHLCV data from Binance for training ML models. It uses the public API (no authentication required) and handles pagination automatically.

## Components

### 1. BinancePublicClient

**File:** `src/bitcoin_scalper/connectors/binance_public.py`

Public API client for fetching historical data:
- No API keys required
- Automatic pagination for large date ranges
- Returns standardized DataFrame format

```python
from bitcoin_scalper.connectors.binance_public import BinancePublicClient

client = BinancePublicClient()

# Fetch 1 year of hourly data
df = client.fetch_history(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(df.head())
```

### 2. Download Script

**File:** `src/bitcoin_scalper/scripts/download_data.py`

Command-line tool for downloading data:

```bash
# Download 1 year of BTC/USDT 1-minute data (default)
python src/bitcoin_scalper/scripts/download_data.py

# Download 30 days of ETH/USDT 5-minute data
python src/bitcoin_scalper/scripts/download_data.py --symbol ETH/USDT --timeframe 5m --days 30

# Download specific date range
python src/bitcoin_scalper/scripts/download_data.py --start-date 2024-01-01 --end-date 2024-12-31

# Custom output location
python src/bitcoin_scalper/scripts/download_data.py --output custom_data.csv
```

## Output Format

CSV with standardized columns:
```csv
date,open,high,low,close,volume
2024-01-01 00:00:00,45000.0,45100.0,44900.0,45050.0,100.0
2024-01-01 01:00:00,45010.0,45110.0,44910.0,45060.0,101.0
```

**Default save location:** `data/raw/BINANCE_{symbol}_{timeframe}.csv`

## Features

✅ **No Authentication Required** - Uses public endpoints  
✅ **Automatic Pagination** - Handles Binance's 1000-candle limit  
✅ **Rate Limiting** - Respects exchange rate limits  
✅ **Progress Logging** - Shows download progress  
✅ **Statistics Display** - Shows date range, total rows, file size  
✅ **Standardized Format** - Lowercase columns matching feature engineering  

## Example Output

```
============================================================
BINANCE HISTORICAL DATA DOWNLOADER
============================================================
Symbol:       BTC/USDT
Timeframe:    1h
Days:         7 (from 2024-01-01 to 2024-01-08)
Output:       data/raw/BINANCE_BTCUSDT_1h.csv
============================================================

[2024-01-08 12:00:00][INFO] Initializing Binance public API client...
[2024-01-08 12:00:00][INFO] Successfully connected to Binance public API
[2024-01-08 12:00:00][INFO] Fetching historical data...
[2024-01-08 12:00:01][INFO] Progress: 10 requests, 1000 candles fetched
[2024-01-08 12:00:02][INFO] Completed paginated fetch: 2 requests, 168 candles
[2024-01-08 12:00:02][INFO] Saving to data/raw/BINANCE_BTCUSDT_1h.csv...

============================================================
DOWNLOAD COMPLETE
============================================================
Symbol:       BTC/USDT
Timeframe:    1h
Start Date:   2024-01-01 00:00:00
End Date:     2024-01-07 23:00:00
Total Rows:   168
Saved to:     data/raw/BINANCE_BTCUSDT_1h.csv
File Size:    9.55 KB
============================================================
```

## Integration with Training

The downloaded CSV format is ready for use with the training pipeline:

```python
import pandas as pd
from bitcoin_scalper.core.feature_engineering import FeatureEngineering

# Load downloaded data
df = pd.read_csv('data/raw/BINANCE_BTCUSDT_1h.csv', index_col='date', parse_dates=True)

# Apply feature engineering (columns already match!)
fe = FeatureEngineering()
df = fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='volume')
df = fe.add_features(df, price_col='close', volume_col='volume')

# Ready for training
print(f"Features ready: {len(df.columns)} columns, {len(df)} rows")
```

## Benefits for Training

1. **Better Data Quality** - Direct from Binance, more reliable than MT5
2. **Easy Access** - No MT5 setup or credentials needed
3. **Flexible Timeframes** - 1m, 5m, 15m, 1h, 4h, 1d, etc.
4. **Historical Depth** - Can fetch years of data
5. **Multiple Symbols** - Any trading pair on Binance
6. **Standardized Format** - Consistent with live trading data format

## Notes

- The public API has rate limits - the script respects these with delays
- For very large date ranges, the download may take several minutes
- Network connection required (obviously)
- Data is fetched from Binance spot market by default
