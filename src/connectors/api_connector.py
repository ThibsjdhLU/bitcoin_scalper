#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import logging
from enum import Enum
from ..exchange.mt5_connection import MT5Connection

class Platform(Enum):
    AVATRADE = "AVATRADE"

class APIConnector:
    """Connecteur API pour AvaTrade via MetaTrader 5"""
    
    def __init__(self, platform: Platform, credentials: dict, symbol: str, timeframe: str):
        """
        Initialisation du connecteur
        
        Args:
            platform: Plateforme de trading (AvaTrade)
            credentials: Dictionnaire contenant login, password et server
            symbol: Symbole à trader (ex: BTCUSD)
            timeframe: Timeframe pour les données (ex: 1m)
        """
        self.logger = logging.getLogger(__name__)
        self.platform = platform
        self.symbol = symbol
        self.timeframe = self._convert_timeframe(timeframe)
        self.is_connected = False
        
        # Utilisation de la connexion singleton
        self.mt5_connection = MT5Connection()
        self.is_connected = self.mt5_connection.initialize(
            login=int(credentials["login"]),
            password=credentials["password"],
            server=credentials["server"]
        )
        
        if self.is_connected:
            self.logger.info("Connexion à AvaTrade établie avec succès")
        else:
            self.logger.error("Échec de la connexion à AvaTrade")
        
    def _convert_timeframe(self, timeframe: str) -> int:
        """Conversion du timeframe en format MT5"""
        timeframe_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)
        
    def get_market_data(self, bars: int = 100) -> pd.DataFrame:
        """
        Récupération des données de marché
        
        Args:
            bars: Nombre de barres à récupérer
            
        Returns:
            DataFrame contenant les données OHLCV
        """
        if not self.is_connected:
            self.logger.error("Non connecté à AvaTrade")
            return None
            
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            self.logger.error(f"Échec de récupération des données: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def get_account_info(self) -> dict:
        """Récupération des informations du compte"""
        if not self.is_connected:
            return None
            
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error(f"Échec de récupération des infos compte: {mt5.last_error()}")
            return None
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }
        
    def execute_order(self, order: dict) -> bool:
        """
        Exécution d'un ordre
        
        Args:
            order: Dictionnaire contenant les détails de l'ordre
                  (type, price, volume, sl, tp)
        """
        if not self.is_connected:
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": order['volume'],
            "type": mt5.ORDER_TYPE_BUY if order['type'] == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": order['price'],
            "sl": order['sl'],
            "tp": order['tp'],
            "devslippage": 20,
            "magic": 234000,
            "comment": "python-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Échec de l'exécution de l'ordre: {result.comment}")
            return False
            
        self.logger.info(f"Ordre exécuté: {result.comment}")
        return True
        
    def disconnect(self):
        """Déconnexion de MetaTrader 5"""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            self.logger.info("Déconnecté d'AvaTrade") 