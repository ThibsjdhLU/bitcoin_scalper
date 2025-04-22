#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bot de scalping Bitcoin pour AvaTrade
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from .api_connector import APIConnector, Platform

class BitcoinScalper:
    """Bot de scalping Bitcoin"""
    
    def __init__(self):
        """Initialisation du bot"""
        self.logger = logging.getLogger(__name__)
        self.load_config()
        self.setup_connector()
        self.setup_indicators()
        
    def load_config(self):
        """Chargement de la configuration"""
        load_dotenv()
        
        self.config = {
            'symbol': os.getenv('SYMBOL', 'BTCUSD'),
            'timeframe': os.getenv('TIMEFRAME', '1m'),
            'volume_min': float(os.getenv('VOLUME_MIN', 0.01)),
            'volume_max': float(os.getenv('VOLUME_MAX', 1.0)),
            'risk_percent': float(os.getenv('RISK_PERCENT', 1.0))
        }
        
    def setup_connector(self):
        """Configuration du connecteur API"""
        credentials = {
            "login": os.getenv("AVATRADE_LOGIN"),
            "password": os.getenv("AVATRADE_PASSWORD"),
            "server": os.getenv("AVATRADE_SERVER")
        }
        
        self.connector = APIConnector(
            platform=Platform.AVATRADE,
            credentials=credentials,
            symbol=self.config['symbol'],
            timeframe=self.config['timeframe']
        )
        
        if not self.connector.is_connected:
            raise ConnectionError("Impossible de se connecter à AvaTrade")
            
    def setup_indicators(self):
        """Configuration des indicateurs techniques"""
        self.indicators = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ema_fast': 9,
            'ema_slow': 21,
            'atr_period': 14
        }
        
    def calculate_indicators(self, data):
        """Calcul des indicateurs techniques"""
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicators['rsi_period']).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        data['ema_fast'] = data['close'].ewm(span=self.indicators['ema_fast']).mean()
        data['ema_slow'] = data['close'].ewm(span=self.indicators['ema_slow']).mean()
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['atr'] = true_range.rolling(window=self.indicators['atr_period']).mean()
        
        return data
        
    def check_entry_conditions(self, data):
        """Vérification des conditions d'entrée"""
        last_row = data.iloc[-1]
        
        # Conditions d'achat
        buy_conditions = [
            last_row['rsi'] < self.indicators['rsi_oversold'],
            last_row['ema_fast'] > last_row['ema_slow'],
            last_row['close'] > last_row['ema_fast']
        ]
        
        # Conditions de vente
        sell_conditions = [
            last_row['rsi'] > self.indicators['rsi_overbought'],
            last_row['ema_fast'] < last_row['ema_slow'],
            last_row['close'] < last_row['ema_fast']
        ]
        
        if all(buy_conditions):
            return 'buy'
        elif all(sell_conditions):
            return 'sell'
        
        return None
        
    def calculate_position_size(self, data):
        """Calcul de la taille de la position"""
        account_info = self.connector.get_account_info()
        if account_info is None:
            return self.config['volume_min']
            
        balance = account_info['balance']
        risk_amount = balance * (self.config['risk_percent'] / 100)
        
        # Utilisation de l'ATR pour le stop loss
        atr = data.iloc[-1]['atr']
        stop_loss_points = atr * 2  # Stop loss à 2 ATR
        
        # Calcul du volume en fonction du risque
        volume = risk_amount / stop_loss_points
        
        # Limites de volume
        volume = max(self.config['volume_min'], min(volume, self.config['volume_max']))
        
        return round(volume, 2)
        
    def execute_trade(self, signal, data):
        """Exécution du trade"""
        try:
            # Calcul de la taille de la position
            volume = self.calculate_position_size(data)
            
            # Prix actuel
            current_price = data.iloc[-1]['close']
            
            # Calcul des niveaux de SL et TP
            atr = data.iloc[-1]['atr']
            sl_distance = atr * 2
            tp_distance = atr * 3
            
            if signal == 'buy':
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
            
            # Création de l'ordre
            order = {
                'type': signal,
                'price': current_price,
                'volume': volume,
                'sl': sl,
                'tp': tp
            }
            
            # Exécution de l'ordre
            result = self.connector.execute_order(order)
            
            if result:
                self.logger.info(f"Ordre exécuté: {result}")
                return True
            else:
                self.logger.error("Échec de l'exécution de l'ordre")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du trade: {e}")
            return False
            
    def run(self):
        """Boucle principale du bot"""
        self.logger.info("Démarrage du bot de scalping...")
        
        while True:
            try:
                # Récupération des données
                data = self.connector.get_market_data(bars=100)
                if data is None:
                    self.logger.error("Impossible de récupérer les données")
                    time.sleep(60)
                    continue
                
                # Calcul des indicateurs
                data = self.calculate_indicators(data)
                
                # Vérification des conditions d'entrée
                signal = self.check_entry_conditions(data)
                
                if signal:
                    self.logger.info(f"Signal détecté: {signal}")
                    if self.execute_trade(signal, data):
                        self.logger.info("Trade exécuté avec succès")
                    else:
                        self.logger.error("Échec de l'exécution du trade")
                
                # Attente avant la prochaine itération
                time.sleep(60)  # Attente d'une minute
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {e}")
                time.sleep(60)
                
    def stop(self):
        """Arrêt du bot"""
        self.logger.info("Arrêt du bot...")
        self.connector.disconnect()
        self.logger.info("Bot arrêté") 