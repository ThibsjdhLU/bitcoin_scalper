# Binance Migration Guide

## Overview

This guide documents the successful migration from MetaTrader 5 (MT5) to Binance exchange using the CCXT library. The migration maintains full compatibility with the existing ML pipeline while providing a modern, standardized interface for cryptocurrency trading.

## Key Changes

### 1. New Binance Connector

**File:** `src/bitcoin_scalper/connectors/binance_connector.py`

A new exchange connector built on CCXT that provides:
- Market data fetching via `fetch_ohlcv()`
- Order execution via `execute_order()`
- Balance retrieval via `get_balance()`

**Key Features:**
- Returns pandas DataFrame with standardized lowercase columns
- Supports both testnet and mainnet
- Built-in rate limiting
- Comprehensive error handling

### 2. Engine Refactoring

**File:** `src/bitcoin_scalper/core/engine.py`

The trading engine is now exchange-agnostic:
- Accepts any connector (MT5RestClient, BinanceConnector, PaperMT5Client)
- Removed hardcoded MT5 dependencies
- Works with standardized data format

**Changes:**
```python
# Before
def __init__(self, mt5_client: MT5RestClient, ...):
    self.mt5_client = mt5_client

# After
def __init__(self, connector, ...):
    self.connector = connector
```

### 3. Data Format Standardization

**Column Names:**
- Binance format: `['date', 'open', 'high', 'low', 'close', 'volume']`
- All lowercase
- 'date' column as datetime index

**Feature Engineering Compatibility:**
The feature engineering module already defaults to lowercase column names, so no changes were needed:
```python
fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='volume')
```

### 4. Configuration Updates

**File:** `config/engine_config.yaml`

New configuration options:
```yaml
# Exchange Configuration
exchange: binance  # Options: binance, mt5, paper
api_key: "YOUR_BINANCE_API_KEY"
api_secret: "YOUR_BINANCE_API_SECRET"
testnet: true  # Use testnet for testing

trading:
  symbol: BTC/USDT  # Binance format with /
  timeframe: 1m     # Binance format: 1m, 5m, 1h, etc.
```

**File:** `src/bitcoin_scalper/core/config.py`

New configuration fields:
- `exchange`: Exchange selection (binance, mt5, paper)
- `binance_api_key`: Binance API key (from env var)
- `binance_api_secret`: Binance API secret (from env var)
- `binance_testnet`: Use testnet mode (from env var)

## Usage

### 1. Installation

Install CCXT library:
```bash
pip install ccxt
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration

#### Option A: Environment Variables (Recommended)
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export BINANCE_TESTNET="true"  # Use testnet for safety
export EXCHANGE="binance"
```

#### Option B: Configuration File
Edit `config/engine_config.yaml`:
```yaml
exchange: binance
api_key: "your_api_key"  # Or use env var
api_secret: "your_api_secret"  # Or use env var
testnet: true

trading:
  symbol: BTC/USDT
  timeframe: 1m
```

### 3. Running the Engine

#### Paper Trading Mode (Recommended for Testing)
```bash
python src/bitcoin_scalper/engine_main.py --mode paper --config config/engine_config.yaml
```

#### Live Trading Mode (Real Money)
```bash
# Make sure to set testnet: false in config
python src/bitcoin_scalper/engine_main.py --mode live --config config/engine_config.yaml
```

### 4. Using the Connector Directly

```python
from bitcoin_scalper.connectors.binance_connector import BinanceConnector

# Initialize connector
connector = BinanceConnector(
    api_key="your_key",
    api_secret="your_secret",
    testnet=True  # Use testnet for testing
)

# Fetch market data
df = connector.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=100)
print(df.head())

# Get balance
balance = connector.get_balance("USDT")
print(f"Free USDT: {balance}")

# Execute order
result = connector.execute_order("BTC/USDT", "buy", 0.001)
print(f"Order ID: {result['id']}")
```

## Symbol Format Differences

### Binance Format
- Use slash separator: `BTC/USDT`, `ETH/USDT`
- Timeframes: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`

### MT5 Format (Legacy)
- No separator: `BTCUSD`, `ETHUSD`
- Timeframes: `M1`, `M5`, `M15`, `H1`, `H4`, `D1`

## Testing

### Unit Tests
```bash
cd /home/runner/work/bitcoin_scalper/bitcoin_scalper
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/connectors/test_binance_connector.py -v
```

**Test Coverage:**
- ✅ Connector initialization
- ✅ OHLCV data fetching with correct columns
- ✅ DataFrame datetime index
- ✅ Order execution
- ✅ Balance retrieval
- ✅ Empty data handling

### Integration Test
```bash
PYTHONPATH=src:$PYTHONPATH python examples/binance_demo.py
```

**Demonstrates:**
- Basic connector usage
- Engine integration
- Column name compatibility
- Feature engineering

## Migration Checklist

- [x] Create Binance connector with CCXT
- [x] Implement fetch_ohlcv with lowercase columns
- [x] Implement execute_order for market orders
- [x] Implement get_balance for USDT
- [x] Refactor engine to be connector-agnostic
- [x] Update configuration files
- [x] Add CCXT to requirements.txt
- [x] Create comprehensive tests
- [x] Verify feature engineering compatibility
- [x] Create documentation
- [x] Create demo script

## Benefits of the Migration

1. **Modern Exchange Support**: Direct integration with Binance, one of the largest crypto exchanges
2. **Testnet Support**: Safe testing environment before live trading
3. **Standardized Data Format**: Lowercase columns align with Python conventions
4. **Exchange Flexibility**: Easy to add support for other exchanges via CCXT
5. **Native macOS Support**: No Windows/MT5 dependencies
6. **Better API**: CCXT provides robust, well-documented exchange APIs
7. **Maintained**: ML pipeline unchanged, maintaining all existing strategies

## Troubleshooting

### API Connection Issues
- Verify API credentials are correct
- Check if testnet mode is enabled (set `testnet: true`)
- Ensure network connectivity
- Check Binance API status: https://www.binance.com/en/support/announcement

### Rate Limiting
- CCXT has built-in rate limiting enabled by default
- Avoid making too many requests in short time
- Consider increasing delays between API calls

### Symbol Format Errors
- Use Binance format: `BTC/USDT` (with slash)
- Not MT5 format: `BTCUSD` (no slash)

### Balance Issues
- Ensure you have funds in your testnet account
- For testnet, request test funds from Binance testnet faucet
- Check the currency code matches (e.g., "USDT", not "USD")

## Security Recommendations

1. **Use Testnet First**: Always test with testnet before using real funds
2. **Environment Variables**: Store credentials in environment variables, not in code
3. **API Key Permissions**: Only grant necessary permissions (read, trade)
4. **IP Whitelist**: Configure API key to only work from specific IPs
5. **Withdraw Restrictions**: Disable withdraw permission on API key
6. **Monitor Activity**: Regularly check your account activity

## Next Steps

1. Test in paper trading mode thoroughly
2. Verify all signals and order execution work as expected
3. When ready, switch to testnet with real API (but test funds)
4. Only after extensive testing, consider mainnet with small amounts
5. Gradually increase position sizes as confidence grows

## Support

For issues or questions:
- Check CCXT documentation: https://docs.ccxt.com/
- Review Binance API docs: https://binance-docs.github.io/apidocs/
- Test with demo script: `python examples/binance_demo.py`

## Conclusion

The migration to Binance via CCXT successfully maintains full compatibility with the existing ML pipeline while providing a modern, flexible foundation for cryptocurrency trading. The standardized data format and exchange-agnostic design make it easy to adapt to other exchanges in the future.
