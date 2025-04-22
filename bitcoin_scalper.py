#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bot de scalping Bitcoin avancé
Version 2.0 avec adversarial testing, détection de régime, et apprentissage méta-continu
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class BitcoinScalper:
    """Bot de scalping Bitcoin avancé"""
    
    def __init__(self):
        """Initialisation du bot"""
        self.setup_logging()
        self.load_config()
        self.connect_to_mt5()
        self.setup_components()
        
    def setup_logging(self):
        """Configuration du logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bitcoin_scalper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Chargement de la configuration"""
        load_dotenv()
        self.symbol = "BTCUSD"
        self.timeframe = mt5.TIMEFRAME_M5
        self.volume = float(os.getenv('TRADE_VOLUME', '0.01'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        
    def connect_to_mt5(self):
        """Connexion à MT5"""
        self.logger.info("Connexion à MT5...")
        
        # Initialisation de MT5
        if not mt5.initialize():
            self.logger.error("Échec de l'initialisation MT5")
            raise RuntimeError("Échec de l'initialisation MT5")
        
        # Connexion au compte
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not mt5.login(login, password, server):
            self.logger.error("Échec de connexion au compte MT5")
            raise RuntimeError("Échec de connexion au compte MT5")
        
        # Vérification de la connexion
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            self.logger.error("Impossible de récupérer les informations du compte")
            raise ConnectionError("Impossible de récupérer les informations du compte")
        
        self.logger.info(f"Connecté à MT5 - Compte: {account_info.login}")
        self.connected = True
        
    def setup_components(self):
        """Configuration des composants"""
        self.regime_detector = RandomForestClassifier(n_estimators=100)
        self.rsi_period = 14
        self.ema_period = 20
        self.atr_period = 14
        
    def get_market_data(self, bars=1000):
        """Récupération des données de marché"""
        try:
            # Récupération des données
            rates = mt5.copy_rates_from_pos(
                self.symbol, self.timeframe, 0, bars
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Pas de données disponibles pour {self.symbol}")
                return None
            
            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Renommage des colonnes
            df.rename(columns={
                'time': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return None
            
    def calculate_indicators(self, df):
        """Calcul des indicateurs techniques"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        return df
        
    def detect_market_regime(self, df):
        """Détection du régime de marché"""
        features = ['rsi', 'ema', 'atr']
        X = df[features].values
        y_pred = self.regime_detector.predict(X[-1:])
        return y_pred[0]
        
    def check_entry_conditions(self, df):
        """Vérification des conditions d'entrée"""
        last_row = df.iloc[-1]
        
        if last_row['rsi'] < 30 and last_row['close'] > last_row['ema']:
            return 'BUY'
        elif last_row['rsi'] > 70 and last_row['close'] < last_row['ema']:
            return 'SELL'
        
        return None
        
    def calculate_position_size(self, df):
        """Calcul de la taille de la position"""
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Impossible d'obtenir les informations du compte")
            return 0.0
            
        equity = account_info.equity
        risk_amount = equity * self.risk_per_trade
        
        last_atr = df['atr'].iloc[-1]
        if last_atr == 0:
            return 0.0
            
        position_size = risk_amount / last_atr
        return round(position_size, 2)
        
    def execute_order(self, order_type, df):
        """Exécution du trade"""
        try:
            # Calcul de la taille de la position
            position_size = self.calculate_position_size(df)
            
            if position_size <= 0:
                self.logger.warning("Taille de position invalide")
                return False
            
            # Prix actuel
            last_price = df['close'].iloc[-1]
            
            # Calcul des niveaux de SL et TP
            last_atr = df['atr'].iloc[-1]
            sl = last_price - 2 * last_atr
            tp = last_price + 3 * last_atr
            
            if order_type == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
            else:
                order_type = mt5.ORDER_TYPE_SELL
            
            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position_size,
                "type": order_type,
                "price": last_price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "python",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Envoi de l'ordre
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Échec d'exécution de l'ordre: {result.comment}")
                return False
            
            self.logger.info(f"Ordre {order_type} exécuté avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du trade: {e}")
            return False
            
    def run(self):
        """Boucle principale du bot"""
        self.logger.info("Démarrage du bot de scalping...")
        
        while True:
            try:
                # Récupération des données
                df = self.get_market_data()
                if df is None:
                    self.logger.error("Impossible de récupérer les données")
                    time.sleep(5)
                    continue
                
                # Calcul des indicateurs
                df = self.calculate_indicators(df)
                
                # Détection du régime
                regime = self.detect_market_regime(df)
                self.logger.info(f"Régime de marché détecté: {regime}")
                
                # Vérification des conditions d'entrée
                signal = self.check_entry_conditions(df)
                
                if signal:
                    self.logger.info(f"Signal détecté: {signal}")
                    if self.execute_order(signal, df):
                        self.logger.info("Trade exécuté avec succès")
                    else:
                        self.logger.error("Échec de l'exécution du trade")
                
                # Attente avant la prochaine itération
                time.sleep(5)  # Attente de 5 secondes
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {e}")
                time.sleep(5)
                
    def stop(self):
        """Arrêt du bot"""
        self.logger.info("Arrêt du bot...")
        mt5.shutdown()
        self.logger.info("Bot arrêté")

def main():
    """Fonction principale"""
    try:
        # Création et démarrage du bot
        bot = BitcoinScalper()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        if 'bot' in locals():
            bot.stop()

if __name__ == "__main__":
    main() 