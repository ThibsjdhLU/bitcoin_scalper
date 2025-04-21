import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from ..risk_management.risk_manager import RiskManager

class Backtester:
    """
    Module de backtesting pour tester les stratégies de trading
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_manager = RiskManager(config)
        self.trades = []
        self.equity_curve = pd.Series()
        self.metrics = {}
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy,
                    initial_capital: float = 10000.0) -> Dict:
        """
        Exécute le backtest d'une stratégie
        
        Args:
            data: Données historiques
            strategy: Stratégie à tester
            initial_capital: Capital initial
            
        Returns:
            Dict: Résultats du backtest
        """
        try:
            self.logger.info("Démarrage du backtest...")
            
            # Initialisation
            self.trades = []
            self.equity_curve = pd.Series(index=data.index, dtype=float)
            self.equity_curve.iloc[0] = initial_capital
            current_capital = initial_capital
            position = None
            
            # Parcours des données
            for i in range(1, len(data)):
                current_data = data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Génération du signal
                signal = strategy.generate_signal(current_data)
                
                # Gestion de la position
                if position is None and signal != 0:
                    # Calcul de la taille de position
                    stop_loss_pips = self._calculate_stop_loss_pips(current_price, signal)
                    position_size = self.risk_manager.calculate_position_size(
                        current_capital,
                        self.config.get('risk_per_trade', 1.0),
                        stop_loss_pips,
                        current_price
                    )
                    
                    # Vérification des limites de risque
                    risk_ok, risk_message = self.risk_manager.check_risk_limits(position_size)
                    if not risk_ok:
                        self.logger.warning(f"Limite de risque dépassée: {risk_message}")
                        continue
                        
                    # Calcul des niveaux
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        current_price,
                        'long' if signal > 0 else 'short',
                        current_data['atr'].iloc[-1]
                    )
                    take_profit = self.risk_manager.calculate_take_profit(
                        current_price,
                        stop_loss,
                        'long' if signal > 0 else 'short'
                    )
                    
                    # Ouverture de position
                    position = {
                        'entry_time': current_data.index[-1],
                        'entry_price': current_price,
                        'direction': 'long' if signal > 0 else 'short',
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                elif position is not None:
                    # Vérification des conditions de sortie
                    exit_price = None
                    exit_reason = None
                    
                    if position['direction'] == 'long':
                        if current_price <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif current_price >= position['take_profit']:
                            exit_price = position['take_profit']
                            exit_reason = 'take_profit'
                    else:
                        if current_price >= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif current_price <= position['take_profit']:
                            exit_price = position['take_profit']
                            exit_reason = 'take_profit'
                            
                    if exit_price is not None:
                        # Calcul du profit/perte
                        if position['direction'] == 'long':
                            profit = (exit_price - position['entry_price']) * position['size']
                        else:
                            profit = (position['entry_price'] - exit_price) * position['size']
                            
                        # Mise à jour du capital
                        current_capital += profit
                        
                        # Enregistrement du trade
                        trade = {
                            'entry_time': position['entry_time'],
                            'exit_time': current_data.index[-1],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'direction': position['direction'],
                            'size': position['size'],
                            'profit': profit,
                            'exit_reason': exit_reason
                        }
                        self.trades.append(trade)
                        
                        # Mise à jour de la courbe d'équité
                        self.equity_curve.iloc[i] = current_capital
                        
                        # Réinitialisation de la position
                        position = None
                        
                else:
                    # Pas de position, mise à jour de la courbe d'équité
                    self.equity_curve.iloc[i] = current_capital
                    
            # Calcul des métriques finales
            self.metrics = self._calculate_metrics()
            
            self.logger.info("Backtest terminé")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du backtest: {str(e)}")
            raise
            
    def _calculate_stop_loss_pips(self, price: float, signal: int) -> float:
        """
        Calcule la distance du stop loss en pips
        
        Args:
            price: Prix actuel
            signal: Signal de trading (1 ou -1)
            
        Returns:
            float: Distance en pips
        """
        # Conversion du prix en pips (1 pip = 0.0001 pour BTC/USD)
        pip_value = 0.0001
        atr_pips = self.config.get('atr_period', 14)
        
        return atr_pips * pip_value
        
    def _calculate_metrics(self) -> Dict:
        """
        Calcule les métriques de performance
        
        Returns:
            Dict: Métriques calculées
        """
        if not self.trades:
            return {}
            
        metrics = {}
        
        # Métriques de base
        metrics['total_trades'] = len(self.trades)
        metrics['winning_trades'] = sum(1 for trade in self.trades if trade['profit'] > 0)
        metrics['losing_trades'] = sum(1 for trade in self.trades if trade['profit'] < 0)
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        
        # Métriques de profit
        metrics['total_profit'] = sum(trade['profit'] for trade in self.trades)
        metrics['average_profit'] = metrics['total_profit'] / metrics['total_trades']
        metrics['profit_factor'] = (
            sum(trade['profit'] for trade in self.trades if trade['profit'] > 0) /
            abs(sum(trade['profit'] for trade in self.trades if trade['profit'] < 0))
            if sum(trade['profit'] for trade in self.trades if trade['profit'] < 0) != 0
            else float('inf')
        )
        
        # Métriques de risque
        returns = self.equity_curve.pct_change().dropna()
        metrics['sharpe_ratio'] = (
            np.sqrt(252) * returns.mean() / returns.std()
            if returns.std() != 0
            else 0.0
        )
        metrics['max_drawdown'] = (
            (self.equity_curve - self.equity_curve.expanding().max()) /
            self.equity_curve.expanding().max()
        ).min()
        
        # Métriques de temps
        metrics['average_trade_duration'] = np.mean([
            (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
            for trade in self.trades
        ])
        
        return metrics
        
    def plot_results(self) -> None:
        """
        Affiche les résultats du backtest
        """
        try:
            import matplotlib.pyplot as plt
            
            # Création de la figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Courbe d'équité
            self.equity_curve.plot(ax=ax1, label='Equity')
            ax1.set_title('Courbe d\'équité')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Capital')
            ax1.grid(True)
            
            # Distribution des profits
            profits = [trade['profit'] for trade in self.trades]
            ax2.hist(profits, bins=50, label='Distribution des profits')
            ax2.set_title('Distribution des profits')
            ax2.set_xlabel('Profit')
            ax2.set_ylabel('Fréquence')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des résultats: {str(e)}")
            raise 