#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de gestion du risque pour le bot de scalping
Gère les règles de risque par trade, taille des positions et drawdown maximal
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    EXTREME = 'extreme'

@dataclass
class RiskMetrics:
    total_exposure: float
    daily_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    risk_level: RiskLevel
    timestamp: datetime

class RiskManager:
    """
    Classe pour gérer les risques de trading
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de risques
        
        Args:
            config (dict): Configuration des paramètres de risque
        """
        self.config = config
        
        # Paramètres de risque
        self.max_position_size = config.get('max_position_size', 1.0)  # Taille max par position
        self.max_total_exposure = config.get('max_total_exposure', 5.0)  # Exposition totale max
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% de perte max par jour
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% de drawdown max
        self.min_win_rate = config.get('min_win_rate', 0.5)  # 50% de win rate minimum
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 1.0)  # Ratio de Sharpe minimum
        
        # État du risque
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown_seen = 0.0
        self.trades_history: List[Dict] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Réinitialisation quotidienne
        self.last_reset = datetime.now()
        
        logger.info("Gestionnaire de risques initialisé")
    
    def can_open_position(self, size: float, leverage: float = 1.0) -> bool:
        """
        Vérifie si une nouvelle position peut être ouverte
        
        Args:
            size (float): Taille de la position
            leverage (float): Levier utilisé
            
        Returns:
            bool: True si la position peut être ouverte
        """
        # Vérification de la taille maximale
        if size > self.max_position_size:
            logger.warning(f"Taille de position trop grande: {size} > {self.max_position_size}")
            return False
        
        # Vérification de l'exposition totale
        new_exposure = self.current_exposure + (size * leverage)
        if new_exposure > self.max_total_exposure:
            logger.warning(f"Exposition totale trop élevée: {new_exposure} > {self.max_total_exposure}")
            return False
        
        # Vérification des pertes journalières
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"Perte journalière maximale atteinte: {self.daily_pnl}")
            return False
        
        # Vérification du drawdown
        if self.max_drawdown_seen > self.max_drawdown:
            logger.warning(f"Drawdown maximum atteint: {self.max_drawdown_seen}")
            return False
        
        return True
    
    def update_position(self, position_id: str, pnl: float, 
                       close_price: Optional[float] = None) -> None:
        """
        Met à jour les métriques de risque avec une position
        
        Args:
            position_id (str): ID de la position
            pnl (float): Profit/Perte de la position
            close_price (float, optional): Prix de fermeture
        """
        # Mise à jour du PnL journalier
        self.daily_pnl += pnl
        
        # Mise à jour du drawdown
        if pnl < 0:
            current_drawdown = abs(pnl) / self.max_total_exposure
            self.max_drawdown_seen = max(self.max_drawdown_seen, current_drawdown)
        
        # Enregistrement du trade
        trade = {
            'id': position_id,
            'pnl': pnl,
            'close_price': close_price,
            'timestamp': datetime.now()
        }
        self.trades_history.append(trade)
        
        # Calcul et enregistrement des métriques
        self._update_risk_metrics()
        
        # Réinitialisation quotidienne si nécessaire
        self._check_daily_reset()
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Récupère les métriques de risque actuelles
        
        Returns:
            RiskMetrics: Métriques de risque
        """
        return self.risk_metrics_history[-1] if self.risk_metrics_history else None
    
    def get_risk_level(self) -> RiskLevel:
        """
        Détermine le niveau de risque actuel
        
        Returns:
            RiskLevel: Niveau de risque
        """
        metrics = self.get_risk_metrics()
        if not metrics:
            return RiskLevel.MEDIUM
        
        # Score de risque basé sur plusieurs facteurs
        risk_score = 0
        
        # Exposition
        exposure_ratio = metrics.total_exposure / self.max_total_exposure
        if exposure_ratio > 0.8:
            risk_score += 3
        elif exposure_ratio > 0.5:
            risk_score += 2
        elif exposure_ratio > 0.2:
            risk_score += 1
        
        # Drawdown
        if metrics.max_drawdown > self.max_drawdown * 0.8:
            risk_score += 3
        elif metrics.max_drawdown > self.max_drawdown * 0.5:
            risk_score += 2
        elif metrics.max_drawdown > self.max_drawdown * 0.2:
            risk_score += 1
        
        # Win rate
        if metrics.win_rate < self.min_win_rate * 0.8:
            risk_score += 3
        elif metrics.win_rate < self.min_win_rate * 0.9:
            risk_score += 2
        elif metrics.win_rate < self.min_win_rate:
            risk_score += 1
        
        # Sharpe ratio
        if metrics.sharpe_ratio < self.min_sharpe_ratio * 0.8:
            risk_score += 3
        elif metrics.sharpe_ratio < self.min_sharpe_ratio * 0.9:
            risk_score += 2
        elif metrics.sharpe_ratio < self.min_sharpe_ratio:
            risk_score += 1
        
        # Détermination du niveau de risque
        if risk_score >= 8:
            return RiskLevel.EXTREME
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _update_risk_metrics(self) -> None:
        """
        Met à jour les métriques de risque
        """
        if not self.trades_history:
            return
        
        # Calcul des métriques
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calcul du ratio de Sharpe
        returns = [t['pnl'] for t in self.trades_history]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0.0
        sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0.0
        
        # Création des métriques
        metrics = RiskMetrics(
            total_exposure=self.current_exposure,
            daily_pnl=self.daily_pnl,
            max_drawdown=self.max_drawdown_seen,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            risk_level=self.get_risk_level(),
            timestamp=datetime.now()
        )
        
        self.risk_metrics_history.append(metrics)
    
    def _check_daily_reset(self) -> None:
        """
        Vérifie et effectue la réinitialisation quotidienne si nécessaire
        """
        now = datetime.now()
        if now.date() > self.last_reset.date():
            # Réinitialisation des métriques journalières
            self.daily_pnl = 0.0
            self.max_drawdown_seen = 0.0
            self.last_reset = now
            
            logger.info("Réinitialisation quotidienne des métriques de risque")