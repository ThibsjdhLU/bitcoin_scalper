"""
Configuration par défaut pour la stratégie de scalping Bitcoin.
"""

# Configuration par défaut
DEFAULT_CONFIG = {
    "symbol": "BTCUSD",
    "volume": 0.01,
    "stop_loss_pips": 50,
    "take_profit_pips": 30,
    "max_spread_pips": 10,
    "timeframe": "M1",
    "max_positions": 3,
    "risk_per_trade": 0.02,  # 2% du capital
    "max_daily_trades": 10,
    "trading_hours": {
        "start": "09:00",
        "end": "17:00"
    }
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
    "trading_hours": {
        "start": "10:00",
        "end": "16:00"
    }
}

# Configuration des logs
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "bitcoin_scalper.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
} 