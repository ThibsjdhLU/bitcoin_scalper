#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration unifiée pour le Bitcoin Scalper
Ce fichier centralise toutes les configurations du système en un seul endroit
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class UnifiedConfig:
    """Classe de gestion unifiée de la configuration"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnifiedConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("UnifiedConfig")
        self.config_path = Path("config/unified_config.json")
        self.config = self._initialize_default_config()
        self._load_config()
        self._initialized = True
    
    def _initialize_default_config(self) -> Dict[str, Any]:
        """
        Initialise la configuration par défaut

        Returns:
            Dict: Configuration par défaut
        """
        return {
            "broker": {
                "mt5": {
                    "server": "Ava-Demo 1-MT5",
                    "login": "101490774",
                    "password": "MatLB356&",
                    "symbols": ["BTCUSD"],
                    "timeframe": "M5",
                    "initial_balance": 1000,
                    "leverage": 10,
                    "volume": 0.001
                }
            },
            "api": {
                "key": "votre_token_secret_ici",
                "host": "0.0.0.0",
                "port": 8000
            },
            "risk": {
                "max_position_size": 0.001,
                "max_risk_per_trade": 0.02,
                "max_daily_drawdown": 0.05,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 2.0,
                "max_daily_trades": 5,
                "max_daily_loss": 100,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02,
                "symbols": {
                    "BTCUSD": {
                        "max_position_size": 0.25,
                        "min_position_size": 0.02,
                        "max_daily_trades": 15
                    },
                    "ETHUSD": {
                        "max_position_size": 0.20,
                        "min_position_size": 0.02,
                        "max_daily_trades": 12
                    },
                    "BNBUSD": {
                        "max_position_size": 0.15,
                        "min_position_size": 0.01,
                        "max_daily_trades": 10
                    }
                }
            },
            "strategies": {
                "ema": {
                    "enabled": True,
                    "fast_period": 9,
                    "slow_period": 21,
                    "min_crossover_strength": 0.0001,
                    "trend_ema_period": 200,
                    "volume": 0.001
                },
                "macd": {
                    "enabled": True,
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "trend_ema_period": 20,
                    "min_histogram_change": 0.0001,
                    "divergence_lookback": 3,
                    "atr_period": 14,
                    "take_profit_atr_multiplier": 2.0,
                    "volume": 0.001
                },
                "rsi": {
                    "enabled": True,
                    "rsi_period": 14,
                    "overbought_threshold": 70,
                    "oversold_threshold": 30,
                    "trend_ema_period": 200,
                    "exit_rsi_threshold": 5,
                    "min_bounce_strength": 0.001,
                    "volume": 0.001
                },
                "bb": {
                    "enabled": True,
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "min_reversal_pct": 0.5,
                    "volume": 0.001
                }
            },
            "trading": {
                "risk_management": {
                    "max_daily_loss": 50,
                    "max_loss_per_trade": 20,
                    "max_drawdown": 10,
                    "position_sizing": {
                        "type": "fixed",
                        "value": 0.001
                    }
                },
                "trade_amount": 100.0,
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "demo_mode": False,
                "initial_capital": 10000.0,
                "risk_per_trade": 1.0,
                "strategy": "EMA Crossover",
                "trailing_stop": False,
                "time_frame": "15m",
                "max_trades": 3,
                "leverage": 1.0,
                "symbols": ["BTCUSD", "ETHUSD", "XRPUSD"],
                "default_symbol": "BTCUSD",
                "risk_percent": 1.0,
                "volume_min": 0.01,
                "volume_max": 1.0,
                "max_daily_trades": 5,
                "max_daily_loss": 5.0,
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            "logging": {
                "level": "INFO",
                "file": "logs/execution.log",
                "max_size": 10485760,
                "backup_count": 5,
                "config": {
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
            },
            "backtest": {
                "start_date": "2024-01-01",
                "end_date": "2024-05-01",
                "initial_capital": 10000.0,
                "commission": 0.1,
                "data": {
                    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    "symbols": ["BTCUSD", "ETHUSD", "BNBUSD"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                },
                "metrics": {
                    "calculate_sharpe": True,
                    "calculate_sortino": True,
                    "calculate_calmar": True,
                    "calculate_max_drawdown": True,
                    "calculate_win_rate": True,
                    "calculate_profit_factor": True
                },
                "output": {
                    "save_trades": True,
                    "save_equity_curve": True,
                    "save_metrics": True,
                    "plot_results": True,
                    "output_dir": "backtest_results"
                }
            },
            "optimization": {
                "enabled": True,
                "interval_hours": 1,
                "methods": [
                    "grid_search",
                    "random_search",
                    "differential_evolution"
                ],
                "metrics": [
                    "sharpe",
                    "sortino",
                    "profit"
                ],
                "cv_splits": 5,
                "grid_search": {
                    "enabled": True,
                    "param_grid": {
                        "ema_crossover": {
                            "fast_period": [5, 9, 12, 15],
                            "slow_period": [20, 26, 30, 35],
                            "min_crossover_strength": [0.0001, 0.0005, 0.001]
                        },
                        "rsi": {
                            "rsi_period": [9, 14, 21],
                            "oversold_threshold": [20, 25, 30],
                            "overbought_threshold": [70, 75, 80],
                            "trend_ema_period": [100, 200, 300]
                        }
                    }
                },
                "random_search": {
                    "enabled": True,
                    "n_iter": 100,
                    "param_ranges": {
                        "ema_crossover": {
                            "fast_period": [5, 20],
                            "slow_period": [20, 50],
                            "min_crossover_strength": [0.0001, 0.01]
                        },
                        "rsi": {
                            "rsi_period": [5, 30],
                            "oversold_threshold": [20, 40],
                            "overbought_threshold": [60, 80],
                            "trend_ema_period": [50, 500]
                        }
                    }
                },
                "differential_evolution": {
                    "enabled": True,
                    "max_iter": 100,
                    "popsize": 15,
                    "mutation": 0.8,
                    "recombination": 0.7
                }
            },
            "indicators": {
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                "bollinger": {
                    "period": 20,
                    "std_dev": 2.0
                },
                "ema_short_period": 8,
                "ema_long_period": 21,
                "stoch_k": 14,
                "stoch_d": 3,
                "stoch_smooth": 3,
                "atr_period": 14
            },
            "ml_strategy": {
                "enabled": True,
                "models": {
                    "random_forest": {
                        "enabled": True,
                        "params": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "min_samples_split": 5,
                            "min_samples_leaf": 2
                        }
                    },
                    "xgboost": {
                        "enabled": True,
                        "params": {
                            "n_estimators": 100,
                            "max_depth": 6,
                            "learning_rate": 0.1,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8
                        }
                    }
                },
                "features": {
                    "price": ["returns", "log_returns"],
                    "moving_averages": [5, 10, 20, 50],
                    "volatility": [5, 10, 20],
                    "volume": [5, 20],
                    "momentum": ["rsi", "macd"]
                },
                "prediction": {
                    "feature_window": 20,
                    "prediction_window": 5,
                    "signal_threshold": 0.6
                },
                "training": {
                    "train_size": 0.8,
                    "shuffle": False,
                    "cv_splits": 5
                }
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipient_email": ""
                },
                "telegram": {
                    "enabled": False,
                    "token": "",
                    "chat_id": ""
                },
                "alerts": {
                    "signal": {
                        "enabled": True,
                        "min_strength": 0.5
                    },
                    "order": {
                        "enabled": True,
                        "min_volume": 0.01
                    },
                    "risk": {
                        "enabled": True,
                        "drawdown_threshold": 10.0,
                        "daily_loss_threshold": 5.0
                    }
                }
            },
            "exchange": {
                "login": "101490774",
                "password": "MatLB356&",
                "server": "Ava-Demo 1-MT5"
            },
            "interface": {
                "refresh_interval": 60,
                "theme": "dark",
                "chart_height": 600,
                "log_level": "info",
                "max_log_entries": 100
            }
        }
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier JSON"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Mettre à jour la configuration avec les valeurs chargées
                    self._update_dict(self.config, loaded_config)
                    self.logger.info(f"Configuration chargée depuis {self.config_path}")
            else:
                # Si le fichier n'existe pas, on le crée avec la configuration par défaut
                self.save()
                self.logger.info(f"Fichier de configuration créé: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """
        Met à jour un dictionnaire de manière récursive
        
        Args:
            target: Dictionnaire cible
            source: Dictionnaire source
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration
        
        Args:
            key_path: Chemin de la clé (ex: "trading.risk_per_trade")
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Any: Valeur de la configuration
        """
        try:
            keys = key_path.split(".")
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Définit une valeur de configuration
        
        Args:
            key_path: Chemin de la clé (ex: "trading.risk_per_trade")
            value: Valeur à définir
        """
        keys = key_path.split(".")
        
        current = self.config
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save(self) -> bool:
        """
        Sauvegarde la configuration dans le fichier JSON
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Créer le répertoire parent s'il n'existe pas
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.logger.info(f"Configuration sauvegardée dans {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False
    
    def load_env(self, env_path: str = ".env") -> None:
        """
        Charge les variables d'environnement depuis un fichier .env
        
        Args:
            env_path: Chemin du fichier .env
        """
        try:
            # Charger les variables d'environnement
            load_dotenv(env_path)
            
            # Mettre à jour la configuration avec les variables d'environnement
            # Exemple pour MT5
            if os.getenv("AVATRADE_LOGIN"):
                self.set("exchange.login", os.getenv("AVATRADE_LOGIN"))
            if os.getenv("AVATRADE_PASSWORD"):
                self.set("exchange.password", os.getenv("AVATRADE_PASSWORD"))
            if os.getenv("AVATRADE_SERVER"):
                self.set("exchange.server", os.getenv("AVATRADE_SERVER"))
            
            self.logger.info(f"Variables d'environnement chargées depuis {env_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des variables d'environnement: {str(e)}")
    
    def reset_to_default(self) -> None:
        """Réinitialise la configuration aux valeurs par défaut"""
        self.config = self._initialize_default_config()
        self.save()
        self.logger.info("Configuration réinitialisée aux valeurs par défaut")
    
    def merge_from_file(self, file_path: str) -> bool:
        """
        Fusionne la configuration avec un fichier externe
        
        Args:
            file_path: Chemin du fichier JSON à fusionner
            
        Returns:
            bool: True si la fusion a réussi
        """
        try:
            with open(file_path, 'r') as f:
                external_config = json.load(f)
            
            self._update_dict(self.config, external_config)
            self.logger.info(f"Configuration fusionnée avec {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion de la configuration: {str(e)}")
            return False

# Exporter une instance unique
config = UnifiedConfig() 