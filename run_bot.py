#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import time
from dotenv import load_dotenv
from typing import List, Dict
import threading
import queue
import tkinter as tk
from tkinter import messagebox

# Ajout du répertoire src au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.exchange.avatrader_mt5 import AvatraderMT5
from src.strategies.regime_detection import HMMRegimeDetector
from src.models.ensemble import MetaLabeler, ModelStacker
from src.analysis.fractal_analysis import FractalAnalyzer, ElliottWaveAnalyzer
from src.portfolio.risk_management import DynamicCorrelation, KellyCriterion, PortfolioRebalancer
from src.optimization.bayesian_optimizer import BayesianOptimizer
from src.ui import TradingUI
from src.bot import TradingBot

# Chargement des variables d'environnement
load_dotenv('config/.env')

# Variable globale pour contrôler l'exécution du bot
bot_running = False
bot_thread = None

def setup_logging():
    """Configure le système de logging."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.getenv('LOG_FILE', 'logs/bitcoin_scalper.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def initialize_components():
    """Initialise tous les composants du bot de trading."""
    logger = logging.getLogger(__name__)
    
    # Initialisation de la connexion à l'exchange
    exchange = AvatraderMT5(
        login=os.getenv('AVATRADE_LOGIN'),
        password=os.getenv('AVATRADE_PASSWORD'),
        server=os.getenv('AVATRADE_SERVER')
    )
    
    # Initialisation des outils d'analyse technique
    regime_detector = HMMRegimeDetector(
        n_regimes=3
    )
    
    fractal_analyzer = FractalAnalyzer(
        window=int(os.getenv('FRACTAL_WINDOW', '20'))
    )
    
    elliott_analyzer = ElliottWaveAnalyzer(
        min_wave_length=int(os.getenv('MIN_WAVE_LENGTH', '10'))
    )
    
    # Initialisation des stratégies de gestion des risques
    correlation_manager = DynamicCorrelation(
        window=int(os.getenv('CORRELATION_WINDOW', '60'))
    )
    
    kelly_criterion = KellyCriterion(
        risk_free_rate=float(os.getenv('RISK_FREE_RATE', '0.02'))
    )
    
    portfolio_rebalancer = PortfolioRebalancer(
        target_volatility=float(os.getenv('TARGET_VOLATILITY', '0.15')),
        rebalance_threshold=float(os.getenv('REBALANCE_THRESHOLD', '0.1'))
    )
    
    # Initialisation des méthodes d'optimisation
    optimizer = BayesianOptimizer(
        param_space={'learning_rate': (0.001, 0.1)},
        n_trials=int(os.getenv('OPTIMIZATION_TRIALS', '100'))
    )
    
    return {
        'exchange': exchange,
        'regime_detector': regime_detector,
        'fractal_analyzer': fractal_analyzer,
        'elliott_analyzer': elliott_analyzer,
        'correlation_manager': correlation_manager,
        'kelly_criterion': kelly_criterion,
        'portfolio_rebalancer': portfolio_rebalancer,
        'optimizer': optimizer
    }

def analyze_trading_signals(market_regime, fractal_patterns, elliott_waves):
    """
    Analyse les signaux de trading en fonction des différents indicateurs.
    
    Args:
        market_regime (np.ndarray): Régimes de marché prédits
        fractal_patterns (dict): Motifs fractals détectés
        elliott_waves (dict): Analyse des vagues d'Elliott
        
    Returns:
        list: Liste des signaux de trading
    """
    signals = []
    
    # Utiliser le dernier régime prédit (le plus récent)
    current_regime = market_regime[-1] if len(market_regime) > 0 else None
    
    # Analyse basée sur le régime de marché
    if current_regime == 0:  # Régime haussier
        if fractal_patterns.get('support_level') and elliott_waves.get('wave_count') >= 3:
            signals.append({
                'type': 'LONG',
                'entry': fractal_patterns['support_level'],
                'stop_loss': fractal_patterns['support_level'] * 0.98,
                'take_profit': fractal_patterns['support_level'] * 1.05,
                'score': 0.8  # Score initial
            })
    elif current_regime == 1:  # Régime baissier
        if fractal_patterns.get('resistance_level') and elliott_waves.get('wave_count') >= 3:
            signals.append({
                'type': 'SHORT',
                'entry': fractal_patterns['resistance_level'],
                'stop_loss': fractal_patterns['resistance_level'] * 1.02,
                'take_profit': fractal_patterns['resistance_level'] * 0.95,
                'score': 0.8  # Score initial
            })
    
    return signals

def validate_trade_signal(signal, components):
    """
    Valide un signal de trading en fonction des conditions du marché.
    
    Args:
        signal (dict): Signal de trading à valider
        components (dict): Composants du bot
        
    Returns:
        bool: True si le signal est valide, False sinon
    """
    # Vérification du risque de corrélation
    if components['correlation_manager'].check_correlations():
        return False
    
    # Calcul de la taille de position optimale
    kelly_fraction = components['kelly_criterion'].calculate_position_size(
        win_rate=0.5,  # À ajuster en fonction des performances historiques
        win_loss_ratio=2.0
    )
    
    # Si la fraction de Kelly est trop faible, on ne prend pas le trade
    if kelly_fraction < 0.1:
        return False
    
    return True

def execute_trade(signal, components):
    """
    Exécute un trade en fonction du signal.
    
    Args:
        signal (dict): Signal de trading à exécuter
        components (dict): Composants du bot
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Calcul de la taille de la position
        position_size = calculate_position_size(signal, components)
        
        # Exécution de l'ordre
        if signal['type'] == 'LONG':
            order = components['exchange'].create_market_buy_order(
                symbol='BTCUSD',
                amount=position_size,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
        else:  # SHORT
            order = components['exchange'].create_market_sell_order(
                symbol='BTCUSD',
                amount=position_size,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
        
        logger.info(f"Trade exécuté: {order}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du trade: {str(e)}")

def calculate_position_size(signal, components):
    """
    Calcule la taille de position optimale.
    
    Args:
        signal (dict): Signal de trading
        components (dict): Composants du bot
        
    Returns:
        float: Taille de la position
    """
    # Récupération du capital disponible
    available_capital = components['exchange'].get_balance()
    
    # Calcul du risque par trade (1% du capital)
    risk_per_trade = available_capital * 0.01
    
    # Calcul du stop loss en points
    stop_loss_points = abs(signal['entry'] - signal['stop_loss'])
    
    # Calcul de la taille de position
    position_size = risk_per_trade / stop_loss_points
    
    return position_size

def apply_optimized_parameters(signals: List[Dict], optimal_params: Dict) -> List[Dict]:
    """
    Applique les paramètres optimisés aux signaux de trading.
    
    Args:
        signals (List[Dict]): Liste des signaux de trading
        optimal_params (Dict): Paramètres optimisés
        
    Returns:
        List[Dict]: Signaux de trading mis à jour
    """
    # Ajustement des signaux avec les paramètres optimisés
    for signal in signals:
        # Ajustement du stop loss et take profit en fonction du learning rate
        learning_rate = optimal_params.get('learning_rate', 0.01)
        
        if signal['type'] == 'LONG':
            signal['stop_loss'] = signal['entry'] * (1 - learning_rate)
            signal['take_profit'] = signal['entry'] * (1 + learning_rate * 2)
        else:  # SHORT
            signal['stop_loss'] = signal['entry'] * (1 + learning_rate)
            signal['take_profit'] = signal['entry'] * (1 - learning_rate * 2)
    
    return signals

def run_bot(ui_queue):
    """
    Fonction principale du bot de trading.
    
    Args:
        ui_queue: Queue pour communiquer avec l'interface utilisateur
    """
    logger = logging.getLogger(__name__)
    global bot_running
    
    try:
        # Initialisation des composants
        components = initialize_components()
        
        # Vérification de la connexion à l'exchange
        if not components['exchange'].connected:
            ui_queue.put({
                "type": "log",
                "content": "Erreur: Impossible de se connecter à l'exchange"
            })
            return
        
        # Boucle principale de trading
        while bot_running:
            try:
                # Récupération des données récentes
                recent_data = components['exchange'].get_historical_data(
                    symbol='BTCUSD',
                    timeframe='1h',
                    limit=100
                )
                
                if not recent_data.empty:
                    # Mise à jour des données de marché dans l'interface
                    ui_queue.put({
                        "type": "market_data",
                        "data": recent_data
                    })
                    
                    # Analyse du marché
                    market_regime = components['regime_detector'].predict(recent_data)
                    fractal_patterns = components['fractal_analyzer'].analyze(recent_data)
                    elliott_waves = components['elliott_analyzer'].analyze(recent_data)
                    
                    # Envoi des informations de régime à l'interface
                    ui_queue.put({
                        "type": "regime",
                        "data": {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "regime": market_regime[-1] if len(market_regime) > 0 else None
                        }
                    })
                    
                    # Vérification des positions ouvertes
                    open_positions = components['exchange'].get_open_positions()
                    
                    # Mise à jour des positions à l'interface
                    ui_queue.put({
                        "type": "positions",
                        "data": open_positions
                    })
                    
                    # Mise à jour du statut de connexion MT5
                    ui_queue.put({
                        "type": "connection_status",
                        "connected": components['exchange'].connected
                    })
                    
                    # Mise à jour des informations du compte
                    account_info = {
                        'balance': components['exchange'].get_balance(),
                        'equity': components['exchange'].get_balance(),  # Pour simplifier, on utilise le même solde
                        'profit': 0.0  # À calculer si nécessaire
                    }
                    ui_queue.put({
                        "type": "account_info",
                        "data": account_info
                    })
                    
                    # Analyse des signaux de trading
                    signals = analyze_trading_signals(market_regime, fractal_patterns, elliott_waves)
                    
                    # Optimisation des paramètres si des signaux sont détectés
                    if signals:
                        optimal_params = components['optimizer'].optimize(
                            objective_function=lambda params: evaluate_signals(params),
                            n_trials=10
                        )
                        
                        signals = apply_optimized_parameters(signals, optimal_params)
                        
                        # Mise à jour des signaux optimisés dans l'interface
                        ui_queue.put({
                            "type": "signals",
                            "data": signals
                        })
                    
                    # Exécution des trades
                    for signal in signals:
                        if validate_trade_signal(signal, components):
                            execute_trade(signal, components)
                            ui_queue.put({
                                "type": "log",
                                "content": f"Trade exécuté: {signal['type']} à {signal['entry']}"
                            })
                    
                    # Mise à jour de la performance
                    equity = components['exchange'].get_balance()
                    ui_queue.put({
                        "type": "performance",
                        "data": {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "equity": equity
                        }
                    })
                else:
                    ui_queue.put({
                        "type": "log",
                        "content": "Aucune donnée de marché disponible"
                    })
                
                # Attente avant la prochaine itération
                time.sleep(60)  # Attente d'une minute
                
            except Exception as e:
                ui_queue.put({
                    "type": "log",
                    "content": f"Erreur dans la boucle principale: {str(e)}"
                })
                time.sleep(300)  # Attente de 5 minutes en cas d'erreur
                
    except Exception as e:
        ui_queue.put({
            "type": "log",
            "content": f"Erreur critique: {str(e)}"
        })
        ui_queue.put({
            "type": "status",
            "content": "Erreur critique"
        })

def start_bot_thread(ui_queue):
    """Démarre le thread du bot de trading"""
    global bot_running, bot_thread
    
    if not bot_running:
        bot_running = True
        bot_thread = threading.Thread(target=run_bot, args=(ui_queue,))
        bot_thread.daemon = True
        bot_thread.start()
        return True
    return False

def stop_bot_thread():
    """Arrête le thread du bot de trading"""
    global bot_running
    
    if bot_running:
        bot_running = False
        if bot_thread:
            bot_thread.join(timeout=5)
        return True
    return False

def main():
    """Fonction principale qui démarre l'interface utilisateur"""
    # Création de la fenêtre principale
    root = tk.Tk()
    
    # Création de l'interface utilisateur
    ui = TradingUI(root)
    
    # Création du bot
    bot = TradingBot()
    bot.set_ui_queue(ui.message_queue)
    
    # Configuration des callbacks
    ui.start_bot_callback = lambda: start_bot_thread(ui.message_queue)
    ui.stop_bot_callback = stop_bot_thread
    
    # Démarrage de la boucle principale de Tkinter
    root.mainloop()

if __name__ == "__main__":
    main() 