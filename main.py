"""
Point d'entrée principal du bot de trading crypto.
Orchestre l'initialisation et l'exécution des différents composants.
"""
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from utils.logger import setup_logger, get_logger, format_boolean
from utils.notifier import Notifier
from utils.indicators import calculate_ema, calculate_macd
from utils.optimizer import StrategyOptimizer, MLStrategy
from core.mt5_connector import MT5Connector, TimeFrame
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.order_executor import OrderExecutor
from core.crash_handler import CrashHandler
from core.data_fetcher import DataFetcher
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_bands_reversal import BollingerBandsReversalStrategy

# Paramètres d'optimisation pour chaque stratégie
STRATEGY_PARAMS = {
    'ema': {
        'param_ranges': {
            'fast_period': (5, 20),
            'slow_period': (20, 50),
            'min_crossover_strength': (0.0001, 0.001)
        },
        'param_grid': {
            'fast_period': [9, 12, 15],
            'slow_period': [21, 26, 30],
            'min_crossover_strength': [0.0001, 0.0005, 0.001]
        }
    },
    'macd': {
        'param_ranges': {
            'fast_period': (8, 16),
            'slow_period': (20, 40),
            'signal_period': (5, 15),
            'trend_ema_period': (100, 300),
            'min_histogram_change': (0.0001, 0.001),
            'divergence_lookback': (5, 15),
            'atr_period': (10, 20),
            'take_profit_atr_multiplier': (1.5, 3.0)
        },
        'param_grid': {
            'fast_period': [12, 14, 16],
            'slow_period': [26, 30, 34],
            'signal_period': [9, 11, 13],
            'trend_ema_period': [100, 200, 300],
            'min_histogram_change': [0.0001, 0.0005, 0.001],
            'divergence_lookback': [5, 10, 15],
            'atr_period': [10, 14, 20],
            'take_profit_atr_multiplier': [1.5, 2.0, 3.0]
        }
    },
    'rsi': {
        'param_ranges': {
            'rsi_period': (9, 21),
            'overbought_threshold': (70, 80),
            'oversold_threshold': (20, 30),
            'trend_ema_period': (100, 300),
            'exit_rsi_threshold': (3, 7),
            'min_bounce_strength': (0.0005, 0.002)
        },
        'param_grid': {
            'rsi_period': [9, 14, 21],
            'overbought_threshold': [70, 75, 80],
            'oversold_threshold': [20, 25, 30],
            'trend_ema_period': [100, 200, 300],
            'exit_rsi_threshold': [3, 5, 7],
            'min_bounce_strength': [0.0005, 0.001, 0.002]
        }
    },
    'bb': {
        'param_ranges': {
            'bb_period': (15, 25),
            'bb_std': (1.5, 2.5),
            'rsi_period': (9, 21),
            'min_reversal_pct': (0.3, 0.7)
        },
        'param_grid': {
            'bb_period': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'rsi_period': [9, 14, 21],
            'min_reversal_pct': [0.3, 0.5, 0.7]
        }
    },
    'ml': {
        'param_ranges': {
            'model_type': ['rf', 'xgb', 'lgb', 'svm'],
            'feature_window': (10, 30),
            'prediction_window': (3, 10)
        },
        'param_grid': {
            'model_type': ['rf', 'xgb', 'lgb', 'svm'],
            'feature_window': [10, 20, 30],
            'prediction_window': [3, 5, 10]
        }
    }
}

def load_config() -> dict:
    """Charge la configuration depuis le fichier config.json."""
    config_path = root_dir / "config" / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """
    Fonction principale du bot de trading.
    Initialise les composants et lance le cycle de trading.
    """
    # Configurer le logger
    setup_logger()
    logger = get_logger()
    
    logger.info("Démarrage du bot de trading crypto")
    
    # Initialiser le crash handler
    with CrashHandler() as crash_handler:
        try:
            # Charger la configuration
            config = load_config()
            config_path = str(root_dir / "config" / "config.json")
            
            # Initialiser les composants
            mt5 = MT5Connector(config_path="config/config.json")
            risk_manager = RiskManager(config=config)
            position_manager = PositionManager(connector=mt5, risk_manager=risk_manager)
            order_executor = OrderExecutor(mt5_connector=mt5, risk_manager=risk_manager)
            data_fetcher = DataFetcher(mt5)
            notifier = Notifier(config)
            
            # Définir les stratégies disponibles
            strategies = {
                'ema': EMACrossoverStrategy,
                'macd': MACDStrategy,
                'rsi': RSIStrategy,
                'bb': BollingerBandsReversalStrategy,
                'ml': MLStrategy
            }
            
            # Configuration de l'optimisation
            optimization_interval = timedelta(hours=24)  # Optimisation quotidienne
            last_optimization = None
            
            # Initialiser la stratégie ML
            ml_strategy = MLStrategy(
                model_type='rf',
                feature_window=20,
                prediction_window=5
            )
            
            # Connexion à MT5
            if not mt5.connect():
                raise ConnectionError("Échec de la connexion à MT5")
            
            logger.info("Initialisation terminée avec succès")
            notifier.send_notification("Bot démarré avec succès")
            
            # Boucle principale de trading
            running = True
            while running:
                try:
                    # Vérifier la connexion MT5
                    if not mt5.ensure_connection():
                        logger.error("Problème de connexion MT5")
                        notifier.send_notification("Problème de connexion MT5")
                        time.sleep(5)
                        continue
                    
                    # Trading sur BTCUSD uniquement
                    symbol = "BTCUSD"
                    timeframe = TimeFrame.M15
                    
                    # Récupérer les données à chaque itération
                    rates = mt5.get_rates(symbol, timeframe, 0, 100)
                    
                    # Attendre 1 seconde avant la prochaine itération
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Arrêt demandé par l'utilisateur")
                    running = False
                except Exception as e:
                    logger.error(f"Erreur dans la boucle principale: {str(e)}")
                    time.sleep(5)
                    continue
            
            logger.info("Arrêt du bot")
            notifier.send_notification("Bot arrêté")
            
        except Exception as e:
            logger.error(f"Erreur critique: {str(e)}")
            notifier.send_notification(f"Erreur critique: {str(e)}")
            raise

if __name__ == "__main__":
    main() 