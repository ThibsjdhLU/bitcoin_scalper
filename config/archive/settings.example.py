# Configuration example for Bitcoin Scalper
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Trading settings
TRADING_SETTINGS = {
    "default_strategy": "ema_crossover",
    "risk_per_trade": 0.02,  # 2% risk per trade
    "max_open_trades": 3,
}

# Exchange settings
EXCHANGE_SETTINGS = {
    "name": "binance",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
}

# Logging settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
