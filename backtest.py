#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de backtest pour le bot de scalping
Permet de tester la stratégie sur des données historiques
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from strategy import Strategy, Signal, SignalType

# Import des modules du bot
from indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    side: str
    size: float
    stop_loss: float
    take_profit: float
    pnl: Optional[float]
    fees: float
    metadata: Dict

class Backtest:
    """
    Système de backtest pour tester les stratégies
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le système de backtest
        
        Args:
            config (dict): Configuration du backtest
        """
        self.config = config
        
        # Paramètres du backtest
        self.initial_balance = config.get('initial_balance', 10000)
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage = config.get('slippage', 0.0001)  # 0.01%
        
        # État du backtest
        self.balance = self.initial_balance
        self.positions: Dict[str, Trade] = {}
        self.trades_history: List[Trade] = []
        self.equity_curve: List[float] = [self.initial_balance]
        
        # Initialisation de la stratégie
        self.strategy = Strategy(config.get('strategy', {}))
        
        logger.info("Système de backtest initialisé")
    
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Exécute le backtest
        
        Args:
            data (Dict[str, pd.DataFrame]): Données historiques par symbole
            
        Returns:
            Dict: Résultats du backtest
        """
        try:
            # Initialisation des résultats
            results = {
                'trades': [],
                'equity_curve': [],
                'metrics': {}
            }
            
            # Pour chaque symbole
            for symbol, df in data.items():
                # Calcul des indicateurs
                df = self.strategy.calculate_indicators(df)
                
                # Pour chaque période
                for i in range(len(df)):
                    current_data = df.iloc[:i+1]
                    current_price = current_data['close'].iloc[-1]
                    current_time = current_data.index[-1]
                    
                    # Mise à jour des positions ouvertes
                    self._update_positions(symbol, current_price, current_time)
                    
                    # Génération du signal
                    signal = self.strategy.generate_signal(symbol, current_data)
                    
                    # Traitement du signal
                    if signal is not None and signal.type != SignalType.NONE:
                        self._handle_signal(signal, current_price, current_time)
            
            # Calcul des métriques
            results['trades'] = self.trades_history
            results['equity_curve'] = self.equity_curve
            results['metrics'] = self._calculate_metrics()
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest: {str(e)}")
            return {
                'trades': [],
                'equity_curve': [self.initial_balance],
                'metrics': {}
            }
    
    def _update_positions(self, symbol: str, current_price: float,
                         current_time: datetime) -> None:
        """
        Met à jour les positions ouvertes
        
        Args:
            symbol (str): Symbole de trading
            current_price (float): Prix actuel
            current_time (datetime): Temps actuel
        """
        # Récupération des positions ouvertes pour le symbole
        open_positions = [p for p in self.positions.values()
                        if p.symbol == symbol and p.exit_time is None]
        
        for position in open_positions:
            # Vérification du stop loss
            if position.side == 'long' and current_price <= position.stop_loss:
                self._close_position(position, current_price, current_time,
                                   'stop_loss')
            elif position.side == 'short' and current_price >= position.stop_loss:
                self._close_position(position, current_price, current_time,
                                   'stop_loss')
            
            # Vérification du take profit
            elif position.side == 'long' and current_price >= position.take_profit:
                self._close_position(position, current_price, current_time,
                                   'take_profit')
            elif position.side == 'short' and current_price <= position.take_profit:
                self._close_position(position, current_price, current_time,
                                   'take_profit')
    
    def _handle_signal(self, signal: Signal, current_price: float,
                      current_time: datetime) -> None:
        """
        Traite un signal de trading
        
        Args:
            signal (Signal): Signal de trading
            current_price (float): Prix actuel
            current_time (datetime): Temps actuel
        """
        try:
            # Vérification des positions existantes
            open_positions = [p for p in self.positions.values()
                            if p.symbol == signal.symbol and p.exit_time is None]
            
            # Fermeture des positions opposées
            for position in open_positions:
                if (signal.type == SignalType.BUY and position.side == 'short') or \
                   (signal.type == SignalType.SELL and position.side == 'long'):
                    self._close_position(position, current_price, current_time,
                                       'signal')
            
            # Ouverture d'une nouvelle position
            if signal.type in [SignalType.BUY, SignalType.SELL]:
                # Calcul de la taille de position
                position_size = self.strategy.calculate_position_size(
                    signal.symbol, signal, self.balance)
                
                # Calcul des niveaux
                stop_loss = self.strategy.calculate_stop_loss(signal, position_size)
                take_profit = self.strategy.calculate_take_profit(
                    signal, stop_loss)
                
                # Création de la position
                position = Trade(
                    symbol=signal.symbol,
                    entry_time=current_time,
                    exit_time=None,
                    entry_price=current_price,
                    exit_price=None,
                    side='long' if signal.type == SignalType.BUY else 'short',
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=None,
                    fees=position_size * current_price * self.commission_rate,
                    metadata={'signal_strength': signal.strength}
                )
                
                # Ajout de la position
                position_id = f"{signal.symbol}_{current_time.timestamp()}"
                self.positions[position_id] = position
                
                # Mise à jour du solde
                self.balance -= position.fees
            
            # Enregistrer le trade
            self.trades_history.append(position)
            
            # Mise à jour de la courbe d'équité
            self.equity_curve.append(self.balance)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du signal: {str(e)}")
    
    def _close_position(self, position: Trade, current_price: float,
                       current_time: datetime, reason: str) -> None:
        """
        Ferme une position
        
        Args:
            position (Trade): Position à fermer
            current_price (float): Prix actuel
            current_time (datetime): Temps actuel
            reason (str): Raison de la fermeture
        """
        # Calcul du PnL
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size
        
        # Calcul des frais
        fees = position.size * current_price * self.commission_rate
        
        # Mise à jour de la position
        position.exit_time = current_time
        position.exit_price = current_price
        position.pnl = pnl - position.fees - fees
        position.metadata['exit_reason'] = reason
        
        # Mise à jour du solde
        self.balance += position.pnl
        
        # Ajout à l'historique
        self.trades_history.append(position)
        
        # Mise à jour de la courbe d'équité
        self.equity_curve.append(self.balance)
    
    def _calculate_metrics(self) -> Dict:
        """
        Calcule les métriques de performance
        
        Returns:
            Dict: Métriques calculées
        """
        if not self.trades_history:
            return {}
        
        # Calcul des métriques de base
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t.pnl > 0])
        losing_trades = len([t for t in self.trades_history if t.pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calcul du PnL
        total_pnl = sum(t.pnl for t in self.trades_history)
        total_fees = sum(t.fees for t in self.trades_history)
        
        # Calcul du drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calcul du ratio de Sharpe
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() \
            if len(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'return': (self.balance - self.initial_balance) / self.initial_balance
        }

    def plot_results(self):
        """Affiche les graphiques des résultats du backtest"""
        try:
            import matplotlib.pyplot as plt
            
            # Création de la figure avec 3 sous-graphiques
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # Graphique 1: Prix et équité
            ax1.plot(self.equity_curve, label='Équité', color='blue')
            ax1.set_title('Courbe d\'Équité')
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('Équité')
            ax1.grid(True)
            ax1.legend()
            
            # Graphique 2: Drawdown
            drawdown = self._calculate_drawdown()
            ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Temps')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True)
            
            # Graphique 3: Distribution des rendements
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            ax3.hist(returns, bins=50, color='green', alpha=0.6)
            ax3.set_title('Distribution des Rendements')
            ax3.set_xlabel('Rendement')
            ax3.set_ylabel('Fréquence')
            ax3.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'affichage des graphiques: {str(e)}")
            raise

    def _calculate_drawdown(self):
        """Calcule le drawdown de la courbe d'équité"""
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        return drawdown

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest du bot de scalping')
    parser.add_argument('--data', type=str, required=True,
                      help='Chemin vers le fichier de données historiques')
    parser.add_argument('--output', type=str, required=True,
                      help='Chemin vers le fichier de sortie des résultats')
    parser.add_argument('--plot', action='store_true',
                      help='Afficher les graphiques des résultats')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Chemin vers le fichier de configuration')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse les arguments
    args = parse_args()
    
    # Configure le logger
    setup_logger()
    
    # Charge la configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        sys.exit(1)
    
    # Charge les données
    try:
        data = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
        data = {os.path.basename(args.data).split('_')[0]: data}
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        sys.exit(1)
    
    # Exécute le backtest
    backtest = Backtest(config)
    results = backtest.run(data)
    
    # Sauvegarde les résultats
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Résultats sauvegardés dans {args.output}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        sys.exit(1)
    
    # Affiche les graphiques si demandé
    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title('Courbe de capital')
        plt.xlabel('Trades')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.show()