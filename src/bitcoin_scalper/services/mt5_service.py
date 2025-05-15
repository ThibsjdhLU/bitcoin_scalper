"""
Service de gestion de la connexion et des opérations MetaTrader 5.
"""

import logging
from pathlib import Path
import json
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import os
import MetaTrader5 as mt5
from threading import Lock
import time
import threading

from config.unified_config import config

logger = logging.getLogger(__name__)

class MT5Service:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(MT5Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.connected = False  # Ensure this attribute is initialized
            
        self.config = config
        self.connected_lock = threading.Lock()
        self._last_connection_attempt = 0
        self.connected_cooldown = 30  # Augmentation du délai entre les tentatives
        self._max_retries = 3
        self._retry_count = 0
        self._demo_mode = False  # Désactiver le mode démo par défaut
        self.lock = Lock()
        self._connect_lock = threading.Lock()
        logger.info("Configuration chargée avec succès")
        
    def connect(self) -> bool:
        with self.connected_lock:
            if self.connected:
                return True
                
            # Reset the retry count if the last attempt was more than 2 minutes ago
            if time.time() - self._last_connection_attempt > 120:
                self._retry_count = 0
                
            # New reconnection logic with exponential backoff
            backoff_time = min(2 ** self._retry_count, 30)
            time.sleep(backoff_time)
            
            # Vérifier le nombre maximum de tentatives
            if self._retry_count >= self._max_retries:
                logger.error("Nombre maximum de tentatives de connexion atteint")
                return False
                
            logger.info("Tentative de connexion à MT5...")
            
            try:
                # Vérifier si MT5 est déjà initialisé
                if mt5.terminal_info():
                    logger.info("MT5 déjà initialisé, fermeture de la connexion existante...")
                    mt5.shutdown()
                    time.sleep(1)
                
                if not mt5.initialize():
                    error = mt5.last_error()
                    logger.error("Échec de l'initialisation MT5: %s", error)
                    self._retry_count += 1
                    return False
                    
                logger.info("MT5 initialisé avec succès, version: %s", mt5.version())
                
                # Récupérer les paramètres de connexion
                server = self.config.get('exchange.server')
                login = int(self.config.get('exchange.login'))
                password = self.config.get('exchange.password')
                
                if not all([server, login, password]):
                    logger.error("Paramètres de connexion manquants")
                    return False
                
                logger.info("Tentative de connexion au serveur %s...", server)
                
                if not mt5.login(login=login, password=password, server=server):
                    error = mt5.last_error()
                    logger.error("Échec de la connexion au serveur: %s", error)
                    self._retry_count += 1
                    mt5.shutdown()
                    return False
                    
                logger.info("Connecté avec succès à %s", server)
                account_info = mt5.account_info()
                if account_info:
                    logger.info("Compte: %d, Nom: %s", account_info.login, account_info.name)
                    logger.info("Solde: %.2f, Equity: %.2f", account_info.balance, account_info.equity)
                
                self.connected = True
                self._retry_count = 0  # Réinitialiser le compteur en cas de succès
                return True
                
            except Exception as e:
                logger.error("Erreur lors de la connexion à MT5: %s", str(e))
                self._retry_count += 1
                self.shutdown()
                return False
            
    def set_demo_mode(self, enabled: bool):
        """Active ou désactive le mode démo."""
        self._demo_mode = enabled
        logger.info(f"Mode démo {'activé' if enabled else 'désactivé'}")
        
        # Mettre à jour la configuration unifiée
        config.set("trading.demo_mode", enabled)
        config.save()
        
    def is_demo_mode(self) -> bool:
        """Indique si le mode démo est activé."""
        return self._demo_mode
            
    def get_positions(self) -> pd.DataFrame:
        """Récupère les positions ouvertes."""
        if not self.connected:
            self.connect()
        
        # Si mode démo ou si la connexion a échoué
        if self._demo_mode or not self.connected:
            return self._generate_demo_positions()
            
        try:
            positions = self.connected.get_positions()
            if positions:
                df = pd.DataFrame(positions)
                df['duration'] = (datetime.now() - pd.to_datetime(df['time'])).dt.total_seconds() / 3600
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions: {str(e)}")
            return self._generate_demo_positions() if self._demo_mode else pd.DataFrame()
            
    def get_account_info(self) -> Optional[Dict]:
        """Récupère les informations du compte."""
        try:
            if not self.connected:
                self.connect()
            
            account_info = mt5.account_info()
            if account_info is None:
                raise ValueError("Aucune donnée de compte")
            
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level
            }
        except Exception as e:
            logger.error(f"Erreur compte: {str(e)}")
            return None
            
    def get_price_history(self, symbol: str = "BTCUSD") -> Optional[pd.DataFrame]:
        """Récupère l'historique des prix pour un symbole donné."""
        with self.lock:
            if not self.connected:
                if not self.connect():
                    logger.error("Impossible de se connecter à MT5")
                    return None
            
            # Si mode démo
            if self._demo_mode:
                return self._generate_demo_price_history(symbol)
            
            try:
                # Récupérer les données des 30 derniers jours
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                # Récupérer les données OHLCV
                rates = mt5.copy_rates_range(
                    symbol, 
                    mt5.TIMEFRAME_H1, 
                    start_date, 
                    end_date
                ) if not self._demo_mode else self._generate_demo_price_history(symbol)
                
                if rates is None or len(rates) == 0:
                    logger.error(f"Aucune donnée trouvée pour {symbol}")
                    return None
                
                # Convertir en DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Renommer les colonnes pour correspondre au format attendu
                df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'tick_volume': 'volume'
                }, inplace=True)
                
                logger.info(f"Données récupérées pour {symbol}: {len(df)} lignes")
                return df
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de l'historique des prix: {str(e)}")
                return self._generate_demo_price_history(symbol) if self._demo_mode else None
            
    def get_available_symbols(self) -> List[str]:
        """Récupère la liste des symboles disponibles."""
        if not self.connected:
            if not self.connect():
                return []  # Échec de la connexion
        
        try:
            # Récupération directe via MT5
            symbols = mt5.symbols_get()
            return [s.name for s in symbols] if symbols else ["BTCUSD"]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des symboles: {str(e)}")
            return ["BTCUSD"]
            
    def _generate_demo_price_history(self, symbol: str) -> pd.DataFrame:
        """Génère des données de prix de démonstration."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq='1h')  # Correction du 'H' déprécié en 'h'
        
        base_price = 50000.0 if symbol == "BTCUSD" else (2000.0 if symbol == "ETHUSD" else 0.5)
        
        # Générer des prix simulés avec une tendance
        np.random.seed(42)  # Pour la reproductibilité
        
        # Créer une tendance sinusoïdale
        trend = np.sin(np.linspace(0, 4*np.pi, len(date_range))) * 0.1
        
        # Ajouter du bruit
        noise = np.random.normal(0, 0.01, len(date_range))
        
        # Combiner pour obtenir des variations de prix
        changes = trend + noise
        
        # Calculer les prix cumulatifs
        price_multiplier = 1.0
        for change in changes:
            price_multiplier *= (1 + change)
            
        close_prices = base_price * np.cumprod(1 + changes)
        
        # Générer OHLC
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(date_range))))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(date_range))))
        open_prices = (high_prices + low_prices) / 2
        
        # Volume
        volumes = np.random.lognormal(5, 1, len(date_range))
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=date_range)
        
        return df
        
    def _generate_demo_positions(self) -> pd.DataFrame:
        """Génère des positions de démonstration."""
        now = datetime.now()
        trade_times = [
            now - timedelta(days=5, hours=3),
            now - timedelta(days=3, hours=7),
            now - timedelta(days=2, hours=2),
            now - timedelta(days=1, hours=5),
            now - timedelta(hours=12)
        ]
        
        trades = []
        for i, trade_time in enumerate(trade_times):
            if i % 2 == 0:
                trade_type = 'BUY'
                price = 50000 + random.randint(-2000, 2000)
                profit = random.randint(50, 300)
            else:
                trade_type = 'SELL'
                price = 50000 + random.randint(-2000, 2000)
                profit = random.randint(-200, 50)
                
            trades.append({
                'time': trade_time,
                'type': trade_type,
                'price_open': price,
                'price_close': price + profit * (1 if trade_type == 'BUY' else -1),
                'profit': profit,
                'volume': 0.01,
                'duration': (now - trade_time).total_seconds() / 3600
            })
            
        return pd.DataFrame(trades)
        
    def _generate_demo_account_info(self) -> Dict:
        """Génère des informations de compte de démonstration."""
        return {
            'balance': 10000.0,
            'equity': 10200.0,
            'profit': 200.0,
            'margin': 1000.0,
            'free_margin': 9200.0,
            'margin_level': 1020.0
        }

    def check_mt5_installed(self) -> bool:
        """
        Vérifie si MetaTrader 5 est installé sur le système.
        
        Returns:
            bool: True si MT5 est installé, False sinon
        """
        # Chemins possibles pour l'installation de MT5
        possible_paths = [
            "C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe",
            "C:\\Program Files (x86)\\Ava Trade MT5 Terminal\\terminal.exe",
            "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            "C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe"
        ]
        
        # Vérifier si l'un des chemins existe
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"MetaTrader 5 trouvé à l'emplacement: {path}")
                return True
        
        # Si aucun chemin n'existe, MT5 n'est probablement pas installé
        logger.error("MetaTrader 5 n'a pas été trouvé sur le système. Veuillez l'installer depuis le site d'Avatrade.")
        return False 

    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Déconnexion MT5 réussie")

    def initialize(self):
        if not mt5.initialize():
            logger.error("Échec de l'initialisation MT5")
            return False
        return True 