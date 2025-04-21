import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

class RiskManager:
    """
    Gestionnaire de risques pour le trading
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.positions = []
        self.trades_history = []
        self.risk_metrics = {}
        
    def calculate_position_size(self, 
                              capital: float,
                              risk_per_trade: float,
                              stop_loss_pips: float,
                              current_price: float) -> float:
        """
        Calcule la taille de position optimale
        
        Args:
            capital: Capital disponible
            risk_per_trade: Risque maximum par trade (%)
            stop_loss_pips: Stop loss en pips
            current_price: Prix actuel
            
        Returns:
            float: Taille de position en lots
        """
        try:
            # Calcul du risque monétaire
            risk_amount = capital * (risk_per_trade / 100)
            
            # Calcul de la valeur du pip
            pip_value = 0.0001  # Pour BTC/USD
            risk_in_pips = stop_loss_pips * pip_value
            
            # Calcul de la taille de position
            position_size = risk_amount / risk_in_pips
            
            # Conversion en lots (1 lot = 100,000 unités)
            lots = position_size / 100000
            
            # Arrondi à 2 décimales
            lots = round(lots, 2)
            
            # Vérification des limites
            min_lots = self.config.get('min_lots', 0.01)
            max_lots = self.config.get('max_lots', 10.0)
            
            lots = max(min_lots, min(lots, max_lots))
            
            return lots
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    def calculate_stop_loss(self,
                          entry_price: float,
                          direction: str,
                          atr: float) -> float:
        """
        Calcule le niveau de stop loss optimal
        
        Args:
            entry_price: Prix d'entrée
            direction: Direction du trade ('long' ou 'short')
            atr: Average True Range
            
        Returns:
            float: Niveau de stop loss
        """
        try:
            # Multiplicateur ATR pour le stop loss
            atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
            
            # Calcul du stop loss
            stop_distance = atr * atr_multiplier
            
            if direction == 'long':
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
                
            return round(stop_loss, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            raise
            
    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss: float,
                            direction: str) -> float:
        """
        Calcule le niveau de take profit optimal
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Niveau de stop loss
            direction: Direction du trade ('long' ou 'short')
            
        Returns:
            float: Niveau de take profit
        """
        try:
            # Ratio risque/récompense
            risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
            
            # Calcul de la distance de stop loss
            stop_distance = abs(entry_price - stop_loss)
            
            # Calcul du take profit
            if direction == 'long':
                take_profit = entry_price + (stop_distance * risk_reward_ratio)
            else:
                take_profit = entry_price - (stop_distance * risk_reward_ratio)
                
            return round(take_profit, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            raise
            
    def update_risk_metrics(self, positions: List[Dict]) -> Dict:
        """
        Met à jour les métriques de risque
        
        Args:
            positions: Liste des positions ouvertes
            
        Returns:
            Dict: Métriques de risque mises à jour
        """
        try:
            metrics = {
                'total_exposure': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0
            }
            
            if not positions:
                return metrics
                
            # Calcul de l'exposition totale
            metrics['total_exposure'] = sum(pos['size'] for pos in positions)
            
            # Calcul du drawdown maximum
            if self.trades_history:
                equity_curve = self._calculate_equity_curve()
                metrics['max_drawdown'] = self._calculate_max_drawdown(equity_curve)
                
            # Calcul du taux de réussite
            if self.trades_history:
                winning_trades = sum(1 for trade in self.trades_history if trade['profit'] > 0)
                metrics['win_rate'] = winning_trades / len(self.trades_history)
                
            # Calcul du facteur de profit
            if self.trades_history:
                gross_profit = sum(trade['profit'] for trade in self.trades_history if trade['profit'] > 0)
                gross_loss = abs(sum(trade['profit'] for trade in self.trades_history if trade['profit'] < 0))
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
                
            # Calcul du ratio de Sharpe
            if self.trades_history:
                returns = [trade['profit'] for trade in self.trades_history]
                metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
                
            self.risk_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
            raise
            
    def _calculate_equity_curve(self) -> pd.Series:
        """
        Calcule la courbe d'équité
        
        Returns:
            Series: Courbe d'équité
        """
        if not self.trades_history:
            return pd.Series()
            
        # Tri des trades par date
        sorted_trades = sorted(self.trades_history, key=lambda x: x['close_time'])
        
        # Calcul de l'équité cumulative
        equity = []
        current_equity = self.config.get('initial_capital', 10000)
        
        for trade in sorted_trades:
            current_equity += trade['profit']
            equity.append(current_equity)
            
        return pd.Series(equity, index=[trade['close_time'] for trade in sorted_trades])
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calcule le drawdown maximum
        
        Args:
            equity_curve: Courbe d'équité
            
        Returns:
            float: Drawdown maximum
        """
        if equity_curve.empty:
            return 0.0
            
        # Calcul des maximums cumulatifs
        rolling_max = equity_curve.expanding().max()
        
        # Calcul des drawdowns
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        return abs(drawdowns.min())
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calcule le ratio de Sharpe
        
        Args:
            returns: Liste des rendements
            
        Returns:
            float: Ratio de Sharpe
        """
        if not returns:
            return 0.0
            
        returns = np.array(returns)
        
        # Calcul du rendement moyen
        mean_return = np.mean(returns)
        
        # Calcul de la volatilité
        volatility = np.std(returns)
        
        # Ratio de Sharpe (taux sans risque = 0)
        sharpe_ratio = mean_return / volatility if volatility != 0 else 0.0
        
        return sharpe_ratio
        
    def check_risk_limits(self, position_size: float) -> Tuple[bool, str]:
        """
        Vérifie si la taille de position respecte les limites de risque
        
        Args:
            position_size: Taille de position proposée
            
        Returns:
            Tuple[bool, str]: (Respect des limites, Message)
        """
        try:
            # Vérification de l'exposition totale
            total_exposure = self.risk_metrics.get('total_exposure', 0.0)
            max_exposure = self.config.get('max_total_exposure', 100.0)
            
            if total_exposure + position_size > max_exposure:
                return False, f"Exposition totale ({total_exposure + position_size}) dépasse la limite ({max_exposure})"
                
            # Vérification du drawdown
            max_drawdown = self.risk_metrics.get('max_drawdown', 0.0)
            max_allowed_drawdown = self.config.get('max_drawdown', 0.2)
            
            if max_drawdown > max_allowed_drawdown:
                return False, f"Drawdown maximum ({max_drawdown:.2%}) dépasse la limite ({max_allowed_drawdown:.2%})"
                
            # Vérification du ratio de Sharpe
            sharpe_ratio = self.risk_metrics.get('sharpe_ratio', 0.0)
            min_sharpe = self.config.get('min_sharpe_ratio', 1.0)
            
            if sharpe_ratio < min_sharpe:
                return False, f"Ratio de Sharpe ({sharpe_ratio:.2f}) inférieur au minimum requis ({min_sharpe})"
                
            return True, "Limites de risque respectées"
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            raise 