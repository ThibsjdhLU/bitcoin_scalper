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

from config.unified_config import config

logger = logging.getLogger(__name__)

class MT5Service:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._connection = None
        self._load_config()
        self._initialized = True
        self._demo_mode = False  # Désactiver le mode démo par défaut
        
    def _load_config(self) -> None:
        """Charge la configuration depuis le système unifié."""
        try:
            # Récupérer la configuration depuis le système unifié
            self._config = {
                'exchange': {
                    'login': config.get('exchange.login', 'DEMO'),
                    'password': config.get('exchange.password', 'DEMO'),
                    'server': config.get('exchange.server', 'DEMO')
                },
                'trading': {
                    'demo_mode': config.get('trading.demo_mode', False)
                }
            }
            
            # Récupérer le mode démo depuis la configuration
            self._demo_mode = self._config['trading']['demo_mode']
            
            logger.info("Configuration chargée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            self._config = {
                'exchange': {
                    'login': 'DEMO',
                    'password': 'DEMO',
                    'server': 'DEMO'
                },
                'trading': {
                    'demo_mode': False
                }
            }
            
    def connect(self) -> bool:
        """Établit la connexion avec MT5."""
        try:
            from exchange.avatrader_mt5 import AvatraderMT5
            
            if self._connection is None:
                self._connection = AvatraderMT5(
                    login=self._config['exchange']['login'],
                    password=self._config['exchange']['password'],
                    server=self._config['exchange']['server']
                )
                
                # Vérifier si la connexion a réussi
                if not self._connection.connected:
                    logger.error("Échec de la connexion à MT5: AvatraderMT5 n'a pas pu se connecter")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion à MT5: {str(e)}")
            # Ne pas activer automatiquement le mode démo en cas d'erreur
            # self._demo_mode = True
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
        if not self._connection:
            self.connect()
        
        # Si mode démo ou si la connexion a échoué
        if self._demo_mode or not self._connection:
            return self._generate_demo_positions()
            
        try:
            positions = self._connection.get_positions()
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
        if not self._connection:
            self.connect()
            
        # Si mode démo ou si la connexion a échoué    
        if self._demo_mode or not self._connection:
            return self._generate_demo_account_info()
            
        try:
            return self._connection.get_account_info()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos compte: {str(e)}")
            return self._generate_demo_account_info() if self._demo_mode else None
            
    def get_price_history(self, symbol: str = "BTCUSD") -> Optional[pd.DataFrame]:
        """Récupère l'historique des prix pour un symbole donné."""
        if not self._connection:
            self.connect()
            
        # Si mode démo ou si la connexion a échoué
        if self._demo_mode or not self._connection:
            return self._generate_demo_price_history(symbol)
            
        try:
            return self._connection.get_price_history(symbol)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique des prix: {str(e)}")
            return self._generate_demo_price_history(symbol) if self._demo_mode else None
            
    def get_available_symbols(self) -> List[str]:
        """Récupère la liste des symboles disponibles."""
        if not self._connection:
            self.connect()
            
        # Si mode démo ou si la connexion a échoué
        if self._demo_mode or not self._connection:
            return ["BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "BCHUSD"]
            
        try:
            symbols = self._connection.get_available_symbols()
            if not symbols:
                return ["BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "BCHUSD"]
            return symbols
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des symboles: {str(e)}")
            return ["BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "BCHUSD"]
            
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