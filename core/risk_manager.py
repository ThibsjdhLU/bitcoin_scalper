"""
Gestionnaire de risques avancé pour le bot de trading.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

@dataclass
class RiskLimits:
    """Limites de risque configurables."""
    max_drawdown: float  # Drawdown maximum autorisé (en %)
    daily_loss_limit: float  # Perte journalière maximale (en %)
    daily_profit_limit: float  # Profit journalier maximal (en %)
    max_position_size: float  # Taille maximale de position (en % du capital)
    min_position_size: float  # Taille minimale de position (en % du capital)
    max_open_positions: int  # Nombre maximum de positions ouvertes
    max_trades_per_day: int  # Nombre maximum de trades par jour
    max_correlation: float  # Corrélation maximale entre positions

@dataclass
class StrategyRiskLimits:
    """Limites de risque spécifiques à une stratégie."""
    max_daily_trades: int  # Nombre maximum de trades par jour
    max_daily_loss: float  # Perte journalière maximale (en %)
    max_daily_profit: float  # Profit journalier maximal (en %)
    max_position_size: float  # Taille maximale de position (en % du capital)
    min_position_size: float  # Taille minimale de position (en % du capital)

class RiskManager:
    """
    Gestionnaire de risques avancé.
    
    Gère :
    - Protection contre le drawdown maximal
    - Limites journalières de perte et gain
    - Taille de position dynamique selon le risque
    - Restrictions par stratégie et par actif
    """
    
    def __init__(self, config: dict):
        """
        Initialise le gestionnaire de risques.
        
        Args:
            config: Configuration du gestionnaire de risques
        """
        self.config = config
        self.risk_limits = RiskLimits(**config['general'])
        self.strategy_limits = {
            name: StrategyRiskLimits(**limits)
            for name, limits in config['strategies'].items()
        }
        
        # État du gestionnaire
        self.initial_capital = config['general']['initial_capital']
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_stats = {
            'trades': 0,
            'pnl': 0.0,
            'start_capital': self.initial_capital
        }
        self.open_positions: Dict[str, dict] = {}
        self.strategy_stats: Dict[str, dict] = {}
        
        # Initialiser les stats par stratégie
        for strategy in self.strategy_limits.keys():
            self.strategy_stats[strategy] = {
                'trades': 0,
                'pnl': 0.0,
                'start_capital': self.initial_capital
            }
            
    def reset_daily_stats(self):
        """Réinitialise les statistiques journalières."""
        self.daily_stats = {
            'trades': 0,
            'pnl': 0.0,
            'start_capital': self.current_capital
        }
        
        for strategy in self.strategy_stats:
            self.strategy_stats[strategy] = {
                'trades': 0,
                'pnl': 0.0,
                'start_capital': self.current_capital
            }
            
    def update_capital(self, pnl: float):
        """
        Met à jour le capital et les statistiques.
        
        Args:
            pnl: Profit/Perte à ajouter
        """
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.daily_stats['pnl'] += pnl
        
    def check_drawdown(self) -> bool:
        """
        Vérifie si le drawdown actuel est acceptable.
        
        Returns:
            bool: True si le drawdown est acceptable, False sinon
        """
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return current_drawdown <= self.risk_limits.max_drawdown
        
    def check_daily_limits(self) -> bool:
        """
        Vérifie si les limites journalières sont respectées.
        
        Returns:
            bool: True si les limites sont respectées, False sinon
        """
        daily_pnl_pct = self.daily_stats['pnl'] / self.daily_stats['start_capital']
        
        # Vérifier la limite de perte
        if daily_pnl_pct < -self.risk_limits.daily_loss_limit:
            logger.warning(f"Limite de perte journalière atteinte: {daily_pnl_pct:.2%}")
            return False
            
        # Vérifier la limite de profit
        if daily_pnl_pct > self.risk_limits.daily_profit_limit:
            logger.warning(f"Limite de profit journalier atteinte: {daily_pnl_pct:.2%}")
            return False
            
        # Vérifier le nombre de trades
        if self.daily_stats['trades'] >= self.risk_limits.max_trades_per_day:
            logger.warning(f"Nombre maximum de trades journaliers atteint: {self.daily_stats['trades']}")
            return False
            
        return True
        
    def check_strategy_limits(self, strategy: str) -> bool:
        """
        Vérifie si les limites de la stratégie sont respectées.
        
        Args:
            strategy: Nom de la stratégie
            
        Returns:
            bool: True si les limites sont respectées, False sinon
        """
        if strategy not in self.strategy_stats:
            return True
            
        stats = self.strategy_stats[strategy]
        limits = self.strategy_limits[strategy]
        
        # Vérifier le nombre de trades
        if stats['trades'] >= limits.max_daily_trades:
            logger.warning(f"Nombre maximum de trades atteint pour {strategy}: {stats['trades']}")
            return False
            
        # Vérifier la perte journalière
        daily_pnl_pct = stats['pnl'] / stats['start_capital']
        if daily_pnl_pct < -limits.max_daily_loss:
            logger.warning(f"Limite de perte journalière atteinte pour {strategy}: {daily_pnl_pct:.2%}")
            return False
            
        # Vérifier le profit journalier
        if daily_pnl_pct > limits.max_daily_profit:
            logger.warning(f"Limite de profit journalier atteinte pour {strategy}: {daily_pnl_pct:.2%}")
            return False
            
        return True
        
    def calculate_position_size(
        self,
        strategy: str,
        symbol: str,
        price: float,
        stop_loss: float
    ) -> float:
        """
        Calcule la taille de position optimale.
        
        Args:
            strategy: Nom de la stratégie
            symbol: Symbole de trading
            price: Prix actuel
            stop_loss: Prix de stop loss
            
        Returns:
            float: Taille de position en unités
        """
        # Récupérer les limites de la stratégie
        limits = self.strategy_limits.get(strategy, self.risk_limits)
        
        # Calculer le risque par unité
        risk_per_unit = abs(price - stop_loss)
        if risk_per_unit == 0:
            return 0
            
        # Calculer le risque maximum en capital
        max_risk = self.current_capital * self.config['general']['risk_per_trade']
        
        # Calculer la taille de position
        position_size = max_risk / risk_per_unit
        
        # Appliquer les limites de taille
        max_size = self.current_capital * limits.max_position_size / price
        min_size = self.current_capital * limits.min_position_size / price
        
        position_size = min(position_size, max_size)
        position_size = max(position_size, min_size)
        
        return position_size
        
    def can_open_position(
        self,
        strategy: str,
        symbol: str,
        side: str,
        price: float,
        stop_loss: float
    ) -> bool:
        """
        Vérifie si une nouvelle position peut être ouverte.
        
        Args:
            strategy: Nom de la stratégie
            symbol: Symbole de trading
            side: Direction de la position ('long' ou 'short')
            price: Prix d'entrée
            stop_loss: Prix de stop loss
            
        Returns:
            bool: True si la position peut être ouverte, False sinon
        """
        # Vérifier le drawdown
        if not self.check_drawdown():
            logger.warning("Drawdown maximum atteint")
            return False
            
        # Vérifier les limites journalières
        if not self.check_daily_limits():
            return False
            
        # Vérifier les limites de la stratégie
        if not self.check_strategy_limits(strategy):
            return False
            
        # Vérifier le nombre de positions ouvertes
        if len(self.open_positions) >= self.risk_limits.max_open_positions:
            logger.warning("Nombre maximum de positions ouvertes atteint")
            return False
            
        # Vérifier la corrélation avec les positions existantes
        if not self._check_correlation(symbol, side):
            logger.warning("Corrélation trop élevée avec les positions existantes")
            return False
            
        return True
        
    def _check_correlation(self, symbol: str, side: str) -> bool:
        """
        Vérifie la corrélation avec les positions existantes.
        
        Args:
            symbol: Symbole de trading
            side: Direction de la position
            
        Returns:
            bool: True si la corrélation est acceptable, False sinon
        """
        # TODO: Implémenter la vérification de corrélation
        # Pour l'instant, on retourne toujours True
        return True
        
    def on_trade(self, strategy: str, symbol: str, pnl: float):
        """
        Met à jour les statistiques après un trade.
        
        Args:
            strategy: Nom de la stratégie
            symbol: Symbole de trading
            pnl: Profit/Perte du trade
        """
        # Mettre à jour le capital
        self.update_capital(pnl)
        
        # Mettre à jour les stats journalières
        self.daily_stats['trades'] += 1
        
        # Mettre à jour les stats de la stratégie
        if strategy in self.strategy_stats:
            self.strategy_stats[strategy]['trades'] += 1
            self.strategy_stats[strategy]['pnl'] += pnl
            
    def get_risk_metrics(self) -> dict:
        """
        Retourne les métriques de risque actuelles.
        
        Returns:
            dict: Métriques de risque
        """
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': (self.peak_capital - self.current_capital) / self.peak_capital,
            'daily_pnl': self.daily_stats['pnl'],
            'daily_trades': self.daily_stats['trades'],
            'open_positions': len(self.open_positions),
            'strategy_stats': self.strategy_stats
        } 