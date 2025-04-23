#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calcul des indicateurs techniques
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any

class TechnicalIndicators:
    """Classe de calcul des indicateurs techniques"""
    
    def __init__(self, rsi_period: int = 14, rsi_overbought: float = 70, rsi_oversold: float = 30):
        """
        Initialise les paramètres des indicateurs
        
        Args:
            rsi_period: Période du RSI
            rsi_overbought: Niveau de surachat du RSI
            rsi_oversold: Niveau de survente du RSI
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.logger = logging.getLogger('Indicators')
        
    def calculate_rsi(self, closes: List[float]) -> Optional[float]:
        """
        Calcule le RSI
        
        Args:
            closes: Liste des prix de clôture
            
        Returns:
            float: Valeur du RSI
        """
        try:
            if len(closes) < self.rsi_period + 1:
                return None
                
            # Calcul des variations
            deltas = np.diff(closes)
            
            # Séparation des gains et pertes
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Moyennes des gains et pertes
            avg_gain = np.mean(gains[:self.rsi_period])
            avg_loss = np.mean(losses[:self.rsi_period])
            
            if avg_loss == 0:
                return 100.0
                
            # Calcul du RS et du RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du RSI: {e}")
            return None
            
    def analyze_signals(self, closes: List[float]) -> Dict[str, Any]:
        """
        Analyse les signaux de trading
        
        Args:
            closes: Liste des prix de clôture
            
        Returns:
            Dict: Signaux d'achat/vente
        """
        signals = {
            'buy': False,
            'sell': False,
            'rsi': None
        }
        
        # Calcul du RSI
        rsi = self.calculate_rsi(closes)
        if rsi is None:
            return signals
            
        signals['rsi'] = rsi
        
        # Analyse des signaux
        if rsi <= self.rsi_oversold:
            signals['buy'] = True
        elif rsi >= self.rsi_overbought:
            signals['sell'] = True
            
        return signals
        
    def get_position_size(self, balance: float, risk_percent: float) -> float:
        """
        Calcule la taille de la position
        
        Args:
            balance: Solde disponible
            risk_percent: Risque par trade en pourcentage
            
        Returns:
            float: Taille de la position
        """
        try:
            return balance * (risk_percent / 100)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.0 