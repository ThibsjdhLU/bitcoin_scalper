#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module d'indicateurs techniques pour le bot de scalping
Utilise pandas pour calculer les indicateurs techniques
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Classe pour calculer les indicateurs techniques en utilisant pandas
    """
    
    def __init__(self, config=None):
        """
        Initialise les indicateurs techniques
        
        Args:
            config (dict): Configuration des indicateurs
        """
        self.config = config or {}
        
        # Paramètres des indicateurs
        self.ema_short_period = self.config.get('ema_short_period', 5)
        self.ema_long_period = self.config.get('ema_long_period', 13)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.stoch_k = self.config.get('stoch_k', 14)
        self.stoch_d = self.config.get('stoch_d', 3)
        self.atr_period = self.config.get('atr_period', 14)
        
        # Stockage des valeurs
        self.indicators = {
            'ema_short': None,
            'ema_long': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'stoch_k': None,
            'stoch_d': None,
            'atr': None
        }
        self.last_update = None
        
        logger.info("Indicateurs techniques initialisés")
    
    def calculate_ema(self, data, period):
        """Calcule l'EMA"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period):
        """Calcule le RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast_period, slow_period, signal_period):
        """Calcule le MACD"""
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
    
    def calculate_stochastic(self, high, low, close, k_period, d_period):
        """Calcule le Stochastique"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def calculate_atr(self, high, low, close, period):
        """Calcule l'ATR"""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr = tr.max(axis=1)
        return tr.rolling(window=period).mean()
    
    def update(self, data):
        """
        Met à jour tous les indicateurs techniques
        
        Args:
            data (pd.DataFrame): Données OHLCV
        """
        try:
            # Calcul des EMAs
            self.indicators['ema_short'] = self.calculate_ema(data['close'], self.ema_short_period)
            self.indicators['ema_long'] = self.calculate_ema(data['close'], self.ema_long_period)
            
            # Calcul du RSI
            self.indicators['rsi'] = self.calculate_rsi(data['close'], self.rsi_period)
            
            # Calcul du MACD
            macd, signal = self.calculate_macd(data['close'], self.macd_fast, self.macd_slow, self.macd_signal)
            self.indicators['macd'] = macd
            self.indicators['macd_signal'] = signal
            
            # Calcul du Stochastique
            k, d = self.calculate_stochastic(data['high'], data['low'], data['close'], self.stoch_k, self.stoch_d)
            self.indicators['stoch_k'] = k
            self.indicators['stoch_d'] = d
            
            # Calcul de l'ATR
            self.indicators['atr'] = self.calculate_atr(data['high'], data['low'], data['close'], self.atr_period)
            
            self.last_update = data.index[-1]
            logger.info("Indicateurs mis à jour")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des indicateurs: {str(e)}")
            raise
    
    def get(self, indicator):
        """
        Récupère la dernière valeur d'un indicateur
        
        Args:
            indicator (str): Nom de l'indicateur
            
        Returns:
            float: Dernière valeur de l'indicateur
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicateur {indicator} non trouvé")
            
        values = self.indicators[indicator]
        if values is None or len(values) == 0:
            return None
            
        return values.iloc[-1]
    
    def get_all(self):
        """
        Récupère toutes les valeurs des indicateurs
        
        Returns:
            dict: Dictionnaire des dernières valeurs des indicateurs
        """
        return {k: self.get(k) for k in self.indicators.keys()}

    def calculate(self, data):
        """
        Calcule tous les indicateurs techniques
        
        Args:
            data (pd.DataFrame): Données OHLCV
        
        Returns:
            pd.DataFrame: DataFrame des indicateurs calculés
        """
        if data is None or len(data) < self.ema_long_period:
            logger.warning(f"Données insuffisantes pour calculer les indicateurs (min: {self.ema_long_period})")
            return None

        try:
            # Création d'une copie pour éviter les modifications sur les données originales
            df = data.copy()
            
            # Calcul des EMAs
            df['ema_short'] = self.calculate_ema(df['close'], self.ema_short_period)
            df['ema_long'] = self.calculate_ema(df['close'], self.ema_long_period)
            
            # Calcul du RSI
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            
            # Calcul du MACD
            macd, signal = self.calculate_macd(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = macd - signal
            
            # Calcul du Stochastique
            k, d = self.calculate_stochastic(df['high'], df['low'], df['close'], self.stoch_k, self.stoch_d)
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            # Calcul de l'ATR
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            
            return df
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            return None
    
    def update_params(self, **kwargs):
        """
        Met à jour les paramètres des indicateurs
        
        Args:
            **kwargs: Nouveaux paramètres
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Paramètre {key} mis à jour: {value}")
    
    def is_crossing_up(self, short_name, long_name):
        """
        Vérifie si un indicateur croise à la hausse un autre
        
        Args:
            short_name (str): Nom de l'indicateur court
            long_name (str): Nom de l'indicateur long
            
        Returns:
            bool: True si croisement à la hausse
        """
        if not self.last_update or short_name not in self.indicators or long_name not in self.indicators:
            return False
            
        return (self.indicators[short_name].iloc[-1] > self.indicators[long_name].iloc[-1] and 
                self.indicators[short_name].iloc[-2] <= self.indicators[long_name].iloc[-2])
    
    def is_crossing_down(self, short_name, long_name):
        """
        Vérifie si un indicateur croise à la baisse un autre
        
        Args:
            short_name (str): Nom de l'indicateur court
            long_name (str): Nom de l'indicateur long
            
        Returns:
            bool: True si croisement à la baisse
        """
        if not self.last_update or short_name not in self.indicators or long_name not in self.indicators:
            return False
            
        return (self.indicators[short_name].iloc[-1] < self.indicators[long_name].iloc[-1] and 
                self.indicators[short_name].iloc[-2] >= self.indicators[long_name].iloc[-2])
    
    def get_atr_value(self, multiplier=1.0):
        """
        Récupère la valeur ATR actuelle avec un multiplicateur
        Utile pour calculer les niveaux SL/TP basés sur la volatilité
        
        Args:
            multiplier (float): Multiplicateur à appliquer à l'ATR
            
        Returns:
            float: Valeur ATR * multiplicateur
        """
        atr = self.get('atr')
        if atr is None:
            return None
        return atr * multiplier

    def get_latest_values(self):
        """
        Récupère les dernières valeurs de tous les indicateurs
        
        Returns:
            dict: Dictionnaire des dernières valeurs des indicateurs
        """
        try:
            latest_values = {}
            for indicator, values in self.indicators.items():
                if values is not None and len(values) > 0:
                    latest_values[indicator] = float(values.iloc[-1])
                else:
                    latest_values[indicator] = None
            return latest_values
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des dernières valeurs: {str(e)}")
            return None