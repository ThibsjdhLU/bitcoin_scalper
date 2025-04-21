#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de connexion API pour le bot de scalping
Supporte MetaTrader 5 et AvaTrade pour l'accès aux marchés crypto
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
from enum import Enum

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

class Platform(Enum):
    """Types de plateformes supportées"""
    MT5 = "mt5"
    AVATRADE = "avatrade"
    CCXT = "ccxt"  # Fallback

class OrderType(Enum):
    """Types d'ordres supportés"""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"

class TimeFrame(Enum):
    """Timeframes disponibles"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class APIConnector:
    """
    Classe de connexion à l'API de trading
    Supporte plusieurs plateformes avec une interface unifiée
    """
    
    def __init__(self, platform, credentials, symbol="BTCUSD", timeframe="1m"):
        """
        Initialisation de la connexion API
        
        Args:
            platform (str): Plateforme de trading ('mt5', 'avatrade', 'ccxt')
            credentials (dict): Informations d'authentification pour la plateforme
            symbol (str): Symbole/paire de trading
            timeframe (str): Intervalle de temps pour les données
        """
        self.logger = logging.getLogger(__name__)
        self.platform = Platform(platform.lower())
        self.credentials = credentials
        self.symbol = symbol
        self.timeframe = timeframe
        self.connected = False
        self.client = None
        
        # Mappage des timeframes entre différentes plateformes
        self.mt5_timeframes = {
            "1m": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else None,
            "5m": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None,
            "15m": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else None,
            "30m": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else None,
            "1h": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None,
            "4h": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
            "1d": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
        }
        
        # Connexion à la plateforme sélectionnée
        self.connect()
    
    @property
    def is_connected(self):
        """Retourne l'état de la connexion"""
        return self.connected
    
    def connect(self):
        """Établit la connexion avec la plateforme de trading"""
        if self.platform == Platform.MT5:
            if not MT5_AVAILABLE:
                raise ImportError("La bibliothèque MetaTrader5 n'est pas installée")
            
            # Initialisation de MT5
            if not mt5.initialize():
                raise ConnectionError(f"Échec de l'initialisation MT5: {mt5.last_error()}")
            
            # Connexion au compte MT5
            result = mt5.login(
                login=int(self.credentials.get("login")),
                password=self.credentials.get("password"),
                server=self.credentials.get("server")
            )
            
            if not result:
                error = mt5.last_error()
                mt5.shutdown()
                raise ConnectionError(f"Échec de la connexion MT5: {error}")
            
            # Vérification de la connexion
            account_info = mt5.account_info()
            if account_info is None:
                mt5.shutdown()
                raise ConnectionError("Impossible de récupérer les informations du compte")
            
            self.client = mt5
            self.connected = True
            self.logger.info(f"Connecté à MT5 - Compte: {account_info.login}")
            return True
            
        elif self.platform == Platform.AVATRADE:
            # AvaTrade utilise également MT5 comme backend
            if not MT5_AVAILABLE:
                raise ImportError("La bibliothèque MetaTrader5 n'est pas installée")
            
            # Initialisation de MT5 pour AvaTrade
            if not mt5.initialize():
                raise ConnectionError(f"Échec de l'initialisation MT5 (AvaTrade): {mt5.last_error()}")
            
            # Connexion au compte AvaTrade via MT5
            result = mt5.login(
                login=int(self.credentials.get("login")),
                password=self.credentials.get("password"),
                server=self.credentials.get("server")
            )
            
            if not result:
                error = mt5.last_error()
                mt5.shutdown()
                raise ConnectionError(f"Échec de la connexion AvaTrade: {error}")
            
            # Vérification de la connexion
            account_info = mt5.account_info()
            if account_info is None:
                mt5.shutdown()
                raise ConnectionError("Impossible de récupérer les informations du compte AvaTrade")
            
            self.client = mt5
            self.connected = True
            self.logger.info(f"Connecté à AvaTrade via MT5 - Compte: {account_info.login}")
            return True
        
        elif self.platform == Platform.CCXT:
            if not CCXT_AVAILABLE:
                raise ImportError("La bibliothèque CCXT n'est pas installée")
            
            # Récupération de l'exchange spécifié (par exemple 'binance')
            exchange_id = self.credentials.get("exchange", "binance")
            
            # Création de l'instance d'exchange
            exchange_class = getattr(ccxt, exchange_id)
            self.client = exchange_class({
                'apiKey': self.credentials.get("api_key"),
                'secret': self.credentials.get("api_secret"),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Pour le trading de futures
                }
            })
            
            # Test de connexion avec fetchBalance
            try:
                self.client.fetch_balance()
                self.logger.info(f"Connecté à {exchange_id.capitalize()} via CCXT")
            except Exception as e:
                raise ConnectionError(f"Échec de la connexion CCXT: {e}")
        
        else:
            raise ValueError(f"Plateforme non supportée: {self.platform}")
        
        self.logger.info(f"Connexion établie avec {self.platform.value}")
    
    def disconnect(self):
        """Fermeture de la connexion avec la plateforme"""
        if not self.connected:
            return
        
        try:
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                mt5.shutdown()
            
            self.connected = False
            self.logger.info(f"Déconnexion de {self.platform.value}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la déconnexion: {e}")
    
    def get_market_data(self, bars=100):
        """
        Récupère les données de marché récentes
        
        Args:
            bars (int): Nombre de barres à récupérer
            
        Returns:
            pandas.DataFrame: Données de marché formatées
        """
        if not self.connected:
            self.logger.warning("Non connecté à la plateforme")
            return None
        
        try:
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                # Conversion du timeframe
                mt5_timeframe = self.mt5_timeframes.get(self.timeframe)
                if mt5_timeframe is None:
                    raise ValueError(f"Timeframe non supporté: {self.timeframe}")
                
                # Récupération des données
                rates = mt5.copy_rates_from_pos(
                    self.symbol, mt5_timeframe, 0, bars
                )
                
                if rates is None or len(rates) == 0:
                    raise ValueError(f"Pas de données disponibles pour {self.symbol}")
                
                # Conversion en DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Renommage des colonnes pour uniformisation
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
            
            elif self.platform == Platform.CCXT:
                # Récupération des données via CCXT
                ohlcv = self.client.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    limit=bars
                )
                
                # Conversion en DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.drop('timestamp', axis=1, inplace=True)
                
                return df
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return None
    
    def execute_order(self, signal):
        """
        Exécute un ordre en fonction du signal généré
        
        Args:
            signal (dict): Signal de trading avec les informations d'ordre
                {
                    'type': 'buy'/'sell',
                    'price': float,
                    'volume': float,
                    'sl': float,
                    'tp': float
                }
                
        Returns:
            dict: Résultat de l'exécution d'ordre ou None en cas d'échec
        """
        if not self.connected:
            self.logger.warning("Non connecté à la plateforme")
            return None
        
        try:
            signal_type = signal['type'].lower()
            price = signal['price']
            volume = signal['volume']
            sl = signal.get('sl')
            tp = signal.get('tp')
            
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                # Configuration de la requête d'ordre MT5
                order_type = None
                if signal_type == 'buy':
                    order_type = mt5.ORDER_TYPE_BUY
                elif signal_type == 'sell':
                    order_type = mt5.ORDER_TYPE_SELL
                elif signal_type == 'buy_limit':
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT
                elif signal_type == 'sell_limit':
                    order_type = mt5.ORDER_TYPE_SELL_LIMIT
                
                if order_type is None:
                    raise ValueError(f"Type d'ordre non supporté: {signal_type}")
                
                # Préparation de la requête
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 10,  # Déviation de prix acceptable en points
                    "magic": 234000,  # Identifiant du bot
                    "comment": "Scalper Adaptatif",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Envoi de l'ordre
                result = mt5.order_send(request)
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    raise Exception(f"Échec de l'ordre: {result.retcode}, {result.comment}")
                
                # Formatage du résultat
                return {
                    'order_id': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'type': signal_type,
                    'sl': sl,
                    'tp': tp,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            elif self.platform == Platform.CCXT:
                # Exécution via CCXT
                order_params = {
                    'stopLoss': {'price': sl} if sl else None,
                    'takeProfit': {'price': tp} if tp else None
                }
                
                if signal_type in ['buy', 'sell']:
                    # Ordre au marché
                    order = self.client.create_order(
                        symbol=self.symbol,
                        type='market',
                        side=signal_type,
                        amount=volume,
                        params=order_params
                    )
                
                elif signal_type in ['buy_limit', 'sell_limit']:
                    # Ordre limite
                    side = 'buy' if signal_type == 'buy_limit' else 'sell'
                    order = self.client.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side=side,
                        amount=volume,
                        price=price,
                        params=order_params
                    )
                
                else:
                    raise ValueError(f"Type d'ordre non supporté: {signal_type}")
                
                # Formatage du résultat
                return {
                    'order_id': order['id'],
                    'volume': volume,
                    'price': price if 'price' in order else order['price'],
                    'type': signal_type,
                    'sl': sl,
                    'tp': tp,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'ordre: {e}")
            return None
    
    def manage_open_orders(self):
        """
        Gère les ordres ouverts (vérification des SL/TP dynamiques, ajustement, etc.)
        """
        if not self.connected:
            return
        
        try:
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                # Récupération des positions ouvertes
                positions = mt5.positions_get(symbol=self.symbol)
                
                if positions:
                    for position in positions:
                        # Validation et ajustement des positions si nécessaire
                        # Par exemple, mise à jour d'un trailing stop
                        pass
            
            elif self.platform == Platform.CCXT:
                # Récupération des positions ouvertes via CCXT
                positions = self.client.fetch_open_orders(symbol=self.symbol)
                
                if positions:
                    for position in positions:
                        # Validation et ajustement des positions si nécessaire
                        pass
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la gestion des ordres ouverts: {e}")
    
    def close_all_positions(self):
        """
        Ferme toutes les positions ouvertes sur le symbole
        
        Returns:
            bool: Succès de l'opération
        """
        if not self.connected:
            return False
        
        try:
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                # Récupération des positions ouvertes
                positions = mt5.positions_get(symbol=self.symbol)
                
                if not positions:
                    return True
                
                success = True
                for position in positions:
                    # Préparation de l'ordre de fermeture
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "position": position.ticket,
                        "comment": "Fermeture par bot",
                    }
                    
                    # Envoi de l'ordre
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.logger.error(f"Erreur lors de la fermeture de la position {position.ticket}: {result.retcode}")
                        success = False
                
                return success
            
            elif self.platform == Platform.CCXT:
                # Fermeture de toutes les positions via CCXT
                positions = self.client.fetch_positions([self.symbol])
                
                if not positions:
                    return True
                
                success = True
                for position in positions:
                    if float(position['contracts']) > 0:
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        self.client.create_order(
                            symbol=self.symbol,
                            type='market',
                            side=side,
                            amount=float(position['contracts']),
                            params={'reduceOnly': True}
                        )
                
                return success
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la fermeture des positions: {e}")
            return False
    
    def get_account_info(self):
        """
        Récupère les informations du compte
        
        Returns:
            dict: Informations du compte
        """
        if not self.connected:
            return None
        
        try:
            if self.platform in [Platform.MT5, Platform.AVATRADE]:
                # Récupération des informations du compte MT5
                account_info = mt5.account_info()
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                }
            
            elif self.platform == Platform.CCXT:
                # Récupération des informations du compte via CCXT
                balance = self.client.fetch_balance()
                return {
                    'balance': balance['total']['USDT'] if 'USDT' in balance['total'] else 0,
                    'equity': balance['free']['USDT'] if 'USDT' in balance['free'] else 0,
                    'margin': balance['used']['USDT'] if 'USDT' in balance['used'] else 0,
                    'free_margin': balance['free']['USDT'] if 'USDT' in balance['free'] else 0,
                }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des informations du compte: {e}")
            return None
    
    def can_trade(self):
        """Vérifie si le compte a les permissions de trading nécessaires"""
        if not self.connected:
            return False
            
        try:
            if self.platform == Platform.MT5:
                # Vérification des permissions MT5
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Impossible d'obtenir les informations du compte")
                    return False
                return account_info.trade_allowed
                
            elif self.platform == Platform.AVATRADE:
                # Vérification des permissions AvaTrade
                # À implémenter selon l'API AvaTrade
                return True
                
            else:
                self.logger.error(f"Plateforme non supportée: {self.platform}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification des permissions de trading: {e}")
            return False
    
    def check_symbol(self, symbol):
        """Vérifie si le symbole est disponible pour le trading"""
        try:
            if self.platform == Platform.MT5:
                # Vérification du symbole sur MT5
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.logger.error(f"Symbole {symbol} non trouvé")
                    return False
                return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED
                
            elif self.platform == Platform.AVATRADE:
                # Vérification du symbole sur AvaTrade
                # À implémenter selon l'API AvaTrade
                return True
                
            else:
                self.logger.error(f"Plateforme non supportée: {self.platform}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification du symbole {symbol}: {e}")
            return False