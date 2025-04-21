import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration MT5
MT5_CONFIG = {
    'LOGIN': int(os.getenv('MT5_LOGIN')),
    'PASSWORD': os.getenv('MT5_PASSWORD'),
    'SERVER': os.getenv('MT5_SERVER'),
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1
}

# Configuration du trading
TRADING_CONFIG = {
    'MAX_POSITIONS': int(os.getenv('MAX_POSITIONS', 3)),
    'RISK_PER_TRADE': float(os.getenv('RISK_PER_TRADE', 0.02)),
    'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 1.0)),
    'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 2.0)),
    'SYMBOLS': ['BTCUSD', 'ETHUSD'],
    'TIMEFRAMES': ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
}

# Configuration des logs
LOG_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'FILE': 'logs/scalper_bot.log'
}

# Configuration des indicateurs
INDICATOR_CONFIG = {
    'RSI_PERIOD': 14,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    'EMA_FAST': 9,
    'EMA_SLOW': 21,
    'BB_PERIOD': 20,
    'BB_STD': 2
}

# Configuration du backtest
BACKTEST_CONFIG = {
    'START_DATE': '2024-01-01',
    'END_DATE': '2024-04-17',
    'INITIAL_BALANCE': 10000,
    'COMMISSION': 0.001
} 