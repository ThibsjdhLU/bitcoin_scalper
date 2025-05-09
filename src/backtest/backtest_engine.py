"""
Module de backtesting pour tester les stratégies de trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import matplotlib.pyplot as plt

from utils.logger import setup_logger
from utils.indicators import calculate_atr

logger = setup_logger(__name__)

class BacktestEngine:
    """Classe pour exécuter des backtests de stratégies de trading."""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001):
        """
        Initialise le moteur de backtest.
        
        Args:
            data: Données historiques (OHLCV)
            initial_capital: Capital initial
            commission: Commission par trade (en %)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def run(self, strategy_func: Callable, **strategy_params) -> Dict:
        """
        Exécute le backtest avec une stratégie donnée.
        
        Args:
            strategy_func: Fonction de stratégie à tester
            strategy_params: Paramètres de la stratégie
            
        Returns:
            Dict: Résultats du backtest
        """
        try:
            # Réinitialiser les résultats
            self.positions = []
            self.trades = []
            self.equity_curve = []
            
            # Calculer les signaux de la stratégie
            signals = strategy_func(self.data, **strategy_params)
            
            # Exécuter le backtest
            capital = self.initial_capital
            position = None
            
            for i in range(len(self.data)):
                current_price = self.data.iloc[i]
                signal = signals.iloc[i]
                
                # Gérer les positions existantes
                if position is not None:
                    # Vérifier le stop loss
                    if position['type'] == 'BUY' and current_price['low'] <= position['stop_loss']:
                        self._close_position(position, current_price, i, capital)
                        position = None
                    elif position['type'] == 'SELL' and current_price['high'] >= position['stop_loss']:
                        self._close_position(position, current_price, i, capital)
                        position = None
                    # Vérifier le take profit
                    elif position['type'] == 'BUY' and current_price['high'] >= position['take_profit']:
                        self._close_position(position, current_price, i, capital)
                        position = None
                    elif position['type'] == 'SELL' and current_price['low'] <= position['take_profit']:
                        self._close_position(position, current_price, i, capital)
                        position = None
                        
                # Ouvrir de nouvelles positions
                if position is None and signal != 0:
                    position = self._open_position(signal, current_price, i, capital)
                    if position:
                        capital -= position['cost']
                        
                # Mettre à jour la courbe d'équité
                self.equity_curve.append(capital)
                
            # Calculer les métriques de performance
            metrics = self._calculate_metrics()
            
            logger.info("Backtest terminé avec succès")
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {e}")
            return {}
            
    def _open_position(self, signal: int, price: pd.Series,
                      index: int, capital: float) -> Optional[Dict]:
        """
        Ouvre une nouvelle position.
        
        Args:
            signal: Signal de trading (1 pour BUY, -1 pour SELL)
            price: Prix actuel
            index: Index actuel
            capital: Capital disponible
            
        Returns:
            Dict: Informations sur la position ouverte
        """
        try:
            # Calculer la taille de position
            position_size = self._calculate_position_size(capital, price)
            
            # Calculer le stop loss et take profit
            atr = calculate_atr(self.data['high'], self.data['low'],
                              self.data['close']).iloc[index]
            
            if signal == 1:  # BUY
                stop_loss = price['close'] - (atr * 2)
                take_profit = price['close'] + (atr * 4)
            else:  # SELL
                stop_loss = price['close'] + (atr * 2)
                take_profit = price['close'] - (atr * 4)
                
            # Créer la position
            position = {
                'type': 'BUY' if signal == 1 else 'SELL',
                'entry_price': price['close'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': self.data.index[index],
                'cost': position_size * price['close'] * (1 + self.commission)
            }
            
            self.positions.append(position)
            return position
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture de la position: {e}")
            return None
            
    def _close_position(self, position: Dict, price: pd.Series,
                       index: int, capital: float):
        """
        Ferme une position existante.
        
        Args:
            position: Position à fermer
            price: Prix actuel
            index: Index actuel
            capital: Capital disponible
        """
        try:
            # Calculer le P&L
            if position['type'] == 'BUY':
                pnl = (price['close'] - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - price['close']) * position['size']
                
            # Soustraire la commission
            pnl -= position['size'] * price['close'] * self.commission
            
            # Enregistrer le trade
            trade = {
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': price['close'],
                'size': position['size'],
                'entry_time': position['entry_time'],
                'exit_time': self.data.index[index],
                'pnl': pnl,
                'return': pnl / position['cost']
            }
            
            self.trades.append(trade)
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la position: {e}")
            
    def _calculate_position_size(self, capital: float,
                               price: pd.Series) -> float:
        """
        Calcule la taille de position.
        
        Args:
            capital: Capital disponible
            price: Prix actuel
            
        Returns:
            float: Taille de position
        """
        try:
            # Risquer 2% du capital par trade
            risk_amount = capital * 0.02
            position_size = risk_amount / price['close']
            
            # Arrondir à 2 décimales
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.0
            
    def _calculate_metrics(self) -> Dict:
        """
        Calcule les métriques de performance.
        
        Returns:
            Dict: Métriques de performance
        """
        try:
            if not self.trades:
                return {}
                
            # Convertir les trades en DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Calculer les métriques de base
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculer les métriques de rendement
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            max_pnl = trades_df['pnl'].max()
            min_pnl = trades_df['pnl'].min()
            
            # Calculer le drawdown
            equity_curve = pd.Series(self.equity_curve)
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculer le ratio de Sharpe
            returns = trades_df['return']
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'max_pnl': max_pnl,
                'min_pnl': min_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_capital': self.equity_curve[-1],
                'return': (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {e}")
            return {}
            
    def plot_results(self, save_path: Optional[str] = None):
        """
        Trace les résultats du backtest.
        
        Args:
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        try:
            if not self.trades:
                logger.error("Aucun trade à tracer")
                return
                
            # Créer la figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Tracer le prix et les trades
            ax1.plot(self.data.index, self.data['close'], label='Prix')
            
            # Tracer les entrées et sorties
            trades_df = pd.DataFrame(self.trades)
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            
            ax1.scatter(buy_trades['entry_time'], buy_trades['entry_price'],
                       marker='^', color='g', label='Entrée BUY')
            ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'],
                       marker='v', color='r', label='Sortie BUY')
            ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'],
                       marker='v', color='r', label='Entrée SELL')
            ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'],
                       marker='^', color='g', label='Sortie SELL')
            
            ax1.set_title('Résultats du Backtest')
            ax1.set_ylabel('Prix')
            ax1.legend()
            ax1.grid(True)
            
            # Tracer la courbe d'équité
            ax2.plot(self.data.index, self.equity_curve, label='Équité')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Capital')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Graphique sauvegardé à {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Erreur lors du tracé des résultats: {e}")
            
    def get_trade_history(self) -> pd.DataFrame:
        """
        Récupère l'historique des trades.
        
        Returns:
            pd.DataFrame: Historique des trades
        """
        try:
            if not self.trades:
                return pd.DataFrame()
                
            return pd.DataFrame(self.trades)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return pd.DataFrame() 