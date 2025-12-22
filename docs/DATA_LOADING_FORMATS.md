# Data Loading Format Support

## Overview

The `load_minute_csv` function in `src/bitcoin_scalper/core/data_loading.py` now supports **two CSV formats**:

1. **Legacy MT5 Format** (original format with `<TAGS>`)
2. **Binance/Standard Format** (new format with lowercase columns)

This dual-format support enables seamless migration from MT5 data sources to Binance data sources without breaking the existing ML pipeline.

## Supported Formats

### 1. Legacy MT5 Format

**Columns:**
- `<DATE>` - Date in format `YYYY.MM.DD`
- `<TIME>` - Time in format `HH:MM:SS`
- `<OPEN>` - Open price
- `<HIGH>` - High price
- `<LOW>` - Low price
- `<CLOSE>` - Close price
- `<TICKVOL>` - Tick volume
- `<VOL>` - Volume (removed during loading)
- `<SPREAD>` - Spread (removed during loading)

**Example:**
```csv
<DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
2023.01.01	00:00:00	16512.74	16514.97	16511.86	16514.97	38	0	3305
2023.01.01	00:01:00	16514.97	16514.97	16511.47	16511.47	26	0	3305
```

### 2. Binance/Standard Format

**Columns:**
- `date` or `timestamp` - Datetime with timezone (e.g., `2023-01-01 00:00:00+00:00`)
- `open` - Open price
- `high` - High price
- `low` - Low price
- `close` - Close price
- `volume` - Volume

**Example:**
```csv
date,open,high,low,close,volume
2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5
2023-01-01 00:01:00+00:00,16514.97,16514.97,16511.47,16511.47,26.3
```

## How It Works

### Automatic Format Detection

The function automatically detects the format by checking for specific column names:

- If `date` or `timestamp` column is present → **Binance/Standard Format**
- If `<DATE>` and `<TIME>` columns are present → **Legacy MT5 Format**

### Column Normalization

Both formats are normalized to the internal pipeline format with `<TAGS>` columns:

| Input Format | Output Format |
|--------------|---------------|
| `date` or `timestamp` | DatetimeIndex |
| `open` | `<OPEN>` |
| `high` | `<HIGH>` |
| `low` | `<LOW>` |
| `close` | `<CLOSE>` |
| `volume` | `<TICKVOL>` |
| `<DATE>` + `<TIME>` | DatetimeIndex |
| `<OPEN>` | `<OPEN>` (unchanged) |
| `<HIGH>` | `<HIGH>` (unchanged) |
| `<LOW>` | `<LOW>` (unchanged) |
| `<CLOSE>` | `<CLOSE>` (unchanged) |
| `<TICKVOL>` | `<TICKVOL>` (unchanged) |

### Output Format

Regardless of input format, the function always returns a DataFrame with:
- **Index:** DatetimeIndex (UTC timezone)
- **Columns:** `['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']`
- **Data types:** All prices/volumes as `float32`

## Usage

### Loading Legacy MT5 CSV

```python
from bitcoin_scalper.core.data_loading import load_minute_csv

# Load MT5 format CSV
df = load_minute_csv('data/raw/BTCUSD_M1_202301010000_202512011647.csv')

print(df.columns)
# Output: Index(['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>'], dtype='object')
```

### Loading Binance CSV

```python
from bitcoin_scalper.core.data_loading import load_minute_csv

# Load Binance format CSV (downloaded from Binance via download_data.py)
df = load_minute_csv('data/raw/BINANCE_BTCUSDT_1m.csv')

print(df.columns)
# Output: Index(['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>'], dtype='object')
```

### Using with train.py

The `train.py` script works with both formats without any changes:

```bash
# Using Legacy MT5 CSV
python scripts/train.py --csv data/raw/BTCUSD_M1_202301010000_202512011647.csv

# Using Binance CSV
python scripts/train.py --csv data/raw/BINANCE_BTCUSDT_1m.csv
```

## Downloading Binance Data

Use the `download_data.py` script to fetch historical data from Binance:

```bash
# Download 1 year of BTC/USDT 1-minute data
python src/bitcoin_scalper/scripts/download_data.py

# Download 30 days of ETH/USDT 5-minute data
python src/bitcoin_scalper/scripts/download_data.py --symbol ETH/USDT --timeframe 5m --days 30

# Download to custom location
python src/bitcoin_scalper/scripts/download_data.py --output data/raw/my_data.csv
```

The downloaded CSV will be in Binance/Standard format and can be used directly with `train.py`.

## Benefits

1. **Backward Compatibility:** Existing MT5 CSVs continue to work without changes
2. **Forward Compatibility:** New Binance CSVs work seamlessly with the pipeline
3. **Transparent Migration:** No changes needed in feature engineering or modeling code
4. **Consistent Output:** Both formats produce identical output structure

## Testing

The dual-format support is thoroughly tested in `tests/core/test_data_loading.py`:

```bash
# Run all data loading tests
cd /home/runner/work/bitcoin_scalper/bitcoin_scalper
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/core/test_data_loading.py -v

# Expected: 16 tests passing
# - Legacy MT5 format tests
# - Binance format tests
# - Format detection tests
# - Output consistency tests
# - Data validation tests
# - Fill method tests
```

## Migration Guide

### From MT5 to Binance

1. **Download new data:**
   ```bash
   python src/bitcoin_scalper/scripts/download_data.py --symbol BTC/USDT --days 365
   ```

2. **Use the new CSV with train.py:**
   ```bash
   python scripts/train.py --csv data/raw/BINANCE_BTCUSDT_1m.csv --fill_missing
   ```

3. **That's it!** The pipeline automatically detects and adapts to the Binance format.

### Keeping Both Formats

You can use both formats interchangeably:

```bash
# Train on MT5 data
python scripts/train.py --csv data/raw/BTCUSD_M1_old.csv

# Train on Binance data
python scripts/train.py --csv data/raw/BINANCE_BTCUSDT_1m.csv
```

The internal pipeline sees no difference - both produce the same `<TAGS>` format columns.

## Troubleshooting

### "Format CSV non reconnu" Error

**Cause:** The CSV doesn't match either expected format.

**Solution:** Ensure your CSV has either:
- Legacy format: `<DATE>` and `<TIME>` columns
- Binance format: `date` or `timestamp` column

### Missing Columns After Loading

**Cause:** Input CSV is missing required OHLCV columns.

**Solution:** Verify your CSV has:
- Legacy: `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`
- Binance: `open`, `high`, `low`, `close`, `volume`

### Data Type Errors

**Cause:** Non-numeric values in price/volume columns.

**Solution:** The function uses `pd.to_numeric` with `errors='coerce'` to handle this. Check for NaN values after loading:

```python
df = load_minute_csv('data.csv')
print(df.isna().sum())  # Check for NaN values
```

## Technical Details

### Implementation

The format detection and adaptation logic is implemented in `load_minute_csv()`:

1. **Detection Phase:**
   - Checks for `date`/`timestamp` columns → Binance format
   - Checks for `<DATE>`/`<TIME>` columns → Legacy MT5 format
   - Raises error if neither found

2. **Adaptation Phase:**
   - **Binance format:**
     - Parse `date`/`timestamp` to datetime
     - Set as index
     - Rename columns: `open` → `<OPEN>`, `volume` → `<TICKVOL>`, etc.
   - **Legacy MT5 format:**
     - Combine `<DATE>` + `<TIME>` into datetime
     - Set as index
     - Remove `<VOL>` and `<SPREAD>` columns

3. **Validation Phase:**
   - Convert all columns to `float32`
   - Remove duplicates
   - Handle NaN values (drop or fill based on `fill_method`)
   - Verify final DataFrame has required columns and DatetimeIndex

### Performance

- Format detection: O(1) - just checks column names
- Column renaming: O(n) where n = number of rows
- No performance difference between formats

## See Also

- [BINANCE_MIGRATION.md](../BINANCE_MIGRATION.md) - Complete Binance migration guide
- [DATA_DOWNLOAD_GUIDE.md](../DATA_DOWNLOAD_GUIDE.md) - Data download instructions
- [download_data.py](../src/bitcoin_scalper/scripts/download_data.py) - Binance data downloader
