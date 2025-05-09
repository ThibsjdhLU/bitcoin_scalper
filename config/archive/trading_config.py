"""
Configuration par défaut du bot de scalping
"""

DEFAULT_CONFIG = {
    "platform": "mt5",
    "credentials": {
        "login": "101490774",
        "password": "MatLB356&",
        "server": "Ava-Demo 1-MT5",
    },
    "trading": {
        "symbol": "BTCUSD",
        "timeframe": "M1",
        "volume": 0.01,
        "max_spread_pips": 15,
        "stop_loss_pips": 75,
        "take_profit_pips": 150,
        "max_positions": 2,
        "risk_per_trade": 0.015,
        "max_daily_trades": 8,
    },
    "backtest_mode": False,
    "log_level": "INFO",
    "log_file": "logs/scalper_bot.log",
    "report_dir": "reports",
    "symbols": ["BTCUSD"],
    "indicators": {
        "ema_short_period": 8,
        "ema_long_period": 21,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth": 3,
        "atr_period": 14,
    },
    "strategy": {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "volume_threshold": 1.5,
        "min_volatility": 0.00008,
        "trend_confirmation": True,
        "signal_strength_threshold": 1.2,
        "consecutive_loss_max": 2,
        "auto_adjust": True,
        "atr_multiplier": 1.8,
        "take_profit_atr": 2.5,
        "stop_loss_atr": 1.2,
    },
    "risk_management": {
        "max_risk_per_trade": 0.015,
        "max_daily_risk": 0.04,
        "max_open_positions": 2,
        "max_drawdown": 0.08,
        "trailing_stop_activation": 0.015,
        "trailing_stop_distance": 1.8,
        "disable_on_volatility": True,
        "high_volatility_threshold": 1.8,
        "min_risk_reward": 1.8,
    },
}

# Configuration en cas d'erreur
ERROR_CONFIG = {
    "symbol": "BTCUSD",
    "volume": 0.01,
    "stop_loss_pips": 100,  # Stop loss plus large en cas d'erreur
    "take_profit_pips": 50,
    "max_spread_pips": 20,
    "timeframe": "M5",  # Timeframe plus large en cas d'erreur
    "max_positions": 1,
    "risk_per_trade": 0.01,  # 1% du capital en cas d'erreur
    "max_daily_trades": 5,
    "trading_hours": {"start": "10:00", "end": "16:00"},
}

# Configuration des logs
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "bitcoin_scalper.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["console", "file"], "level": "INFO", "propagate": True}
    },
}

# Création du fichier __init__.py
import os

with open(os.path.join(os.path.dirname(__file__), "__init__.py"), "w") as f:
    f.write("from .scalper_config import DEFAULT_CONFIG\n")
