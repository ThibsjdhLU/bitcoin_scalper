"""
Module de gestion de la connexion à MetaTrader 5.
Gère l'authentification, la vérification des symboles et la reconnection automatique.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

import MetaTrader5 as mt5
from loguru import logger

class TimeFrame(Enum):
    """Timeframes supportés."""
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1

class MT5Connector:
    """
    Gère la connexion et les interactions avec MetaTrader 5.
    
    Attributes:
        config_path (str): Chemin vers le fichier de configuration
        server (str): Serveur MT5
        login (int): Identifiant de connexion
        password (str): Mot de passe
        symbols (List[str]): Liste des symboles à trader
        connected (bool): État de la connexion
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialise le connecteur MT5.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = config_path
        self.server = ""
        self.login = 0
        self.password = ""
        self.symbols: List[str] = []
        self.connected = False
        
        # Paramètres de reconnection
        self.max_retries = 5
        self.retry_delay = 5  # secondes
        self.connection_timeout = 30  # secondes
        self.last_connection_check = 0
        self.connection_check_interval = 60  # secondes
        
        # Charger la configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            mt5_config = config['broker']['mt5']
            self.server = mt5_config['server']
            self.login = int(mt5_config['login'])
            self.password = mt5_config['password']
            self.symbols = mt5_config['symbols']
            
            # Charger les paramètres de reconnection si présents
            if 'connection' in mt5_config:
                conn_config = mt5_config['connection']
                self.max_retries = conn_config.get('max_retries', self.max_retries)
                self.retry_delay = conn_config.get('retry_delay', self.retry_delay)
                self.connection_timeout = conn_config.get('timeout', self.connection_timeout)
                self.connection_check_interval = conn_config.get('check_interval', self.connection_check_interval)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            raise
    
    def connect(self) -> bool:
        """
        Établit la connexion avec MetaTrader 5.
        
        Returns:
            bool: True si la connexion est réussie, False sinon
        """
        if not mt5.initialize():
            logger.error(f"Échec de l'initialisation MT5: {mt5.last_error()}")
            return False
        
        # Tenter la connexion
        if not mt5.login(
            login=self.login,
            password=self.password,
            server=self.server
        ):
            logger.error(f"Échec de la connexion MT5: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        # Vérifier les symboles
        if not self._verify_symbols():
            logger.error("Échec de la vérification des symboles")
            mt5.shutdown()
            return False
        
        self.connected = True
        logger.info("Connexion MT5 établie avec succès")
        return True
    
    def _verify_symbols(self) -> bool:
        """
        Vérifie que tous les symboles configurés sont disponibles.
        
        Returns:
            bool: True si tous les symboles sont valides, False sinon
        """
        for symbol in self.symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbole non trouvé: {symbol}")
                return False
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Impossible d'activer le symbole: {symbol}")
                    return False
        return True
    
    def reconnect(self, max_attempts: int = 3, delay: int = 5) -> bool:
        """
        Tente de rétablir la connexion en cas de déconnexion.
        
        Args:
            max_attempts: Nombre maximum de tentatives
            delay: Délai entre les tentatives en secondes
            
        Returns:
            bool: True si la reconnection est réussie, False sinon
        """
        for attempt in range(max_attempts):
            logger.info(f"Tentative de reconnection {attempt + 1}/{max_attempts}")
            
            if self.connect():
                return True
            
            if attempt < max_attempts - 1:
                logger.warning(f"Échec de la reconnection, nouvelle tentative dans {delay} secondes")
                time.sleep(delay)
        
        logger.error("Échec de la reconnection après plusieurs tentatives")
        return False
    
    def disconnect(self) -> None:
        """Ferme la connexion MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Déconnexion MT5 effectuée")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Récupère les informations d'un symbole.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            Optional[Dict]: Informations du symbole ou None si non trouvé
        """
        if not self.connected:
            logger.error("Non connecté à MT5")
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbole non trouvé: {symbol}")
            return None
        
        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'digits': info.digits
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def ensure_connection(self) -> bool:
        """
        Vérifie et rétablit la connexion si nécessaire.
        
        Returns:
            bool: True si connecté, False sinon
        
        Raises:
            ConnectionError: Si la reconnexion échoue après max_retries tentatives
        """
        current_time = time.time()
        
        # Vérifier la connexion seulement si l'intervalle est dépassé
        if (current_time - self.last_connection_check) < self.connection_check_interval:
            return self.connected
            
        self.last_connection_check = current_time
        
        if self.connected and mt5.terminal_info() is not None:
            return True
            
        retries = 0
        while retries < self.max_retries:
            try:
                # Fermer une éventuelle connexion existante
                if mt5.terminal_info() is not None:
                    mt5.shutdown()
                    
                # Initialiser MT5 avec timeout
                if not mt5.initialize(timeout=self.connection_timeout):
                    raise ConnectionError(f"Échec d'initialisation MT5: {mt5.last_error()}")
                    
                # Se connecter au compte
                if not mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                ):
                    raise ConnectionError(f"Échec de connexion MT5: {mt5.last_error()}")
                    
                # Vérifier les symboles
                if not self._verify_symbols():
                    raise ConnectionError("Échec de la vérification des symboles")
                    
                self.connected = True
                logger.info("Connexion MT5 établie avec succès")
                
                # Afficher les symboles disponibles
                symbols = mt5.symbols_get()
                logger.info("Symboles disponibles sur MT5:")
                for symbol in symbols:
                    if "BTC" in symbol.name:
                        logger.info(f"- {symbol.name}")
                
                return True
                
            except Exception as e:
                retries += 1
                logger.warning(f"Tentative {retries}/{self.max_retries} échouée: {str(e)}")
                if retries < self.max_retries:
                    time.sleep(self.retry_delay * retries)  # Backoff exponentiel
                    
        self.connected = False
        raise ConnectionError(
            f"Impossible de se connecter à MT5 après {self.max_retries} tentatives"
        )
        
    def _safe_request(self, func, *args, **kwargs):
        """
        Exécute une requête MT5 avec gestion de la reconnexion.
        
        Args:
            func: Fonction MT5 à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Le résultat de la fonction
            
        Raises:
            ConnectionError: Si la requête échoue après reconnexion
            MT5Error: Pour les erreurs spécifiques à MT5
        """
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                # Vérifier la connexion avant chaque requête
                self.ensure_connection()
                result = func(*args, **kwargs)
                
                # Vérifier si le résultat est valide
                if result is None:
                    error = mt5.last_error()
                    if error[0] == mt5.TRADE_RETCODE_REQUOTE:
                        retries += 1
                        time.sleep(1)
                        continue
                    raise MT5Error(f"Erreur MT5: {error}")
                    
                return result
                
            except ConnectionError:
                retries += 1
                if retries < max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Erreur lors de la requête MT5: {str(e)}")
                raise MT5Error(f"Erreur MT5: {str(e)}")
            
    def get_rates(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_pos: int = 0,
        count: int = 1000
    ) -> Optional[np.ndarray]:
        """
        Récupère les données historiques.
        
        Args:
            symbol: Symbole à récupérer
            timeframe: Timeframe des données
            start_pos: Position de départ
            count: Nombre de périodes
            
        Returns:
            Optional[np.ndarray]: Données OHLCV ou None si erreur
        """
        if not self.connected:
            logger.error("Non connecté à MT5")
            return None
            
        try:
            # Vérifier que le symbole existe
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbole {symbol} non trouvé")
                return None
                
            # Récupérer les données
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, start_pos, count)
            if rates is None:
                logger.error(f"Échec de récupération des données pour {symbol}")
                return None
                
            # Vérifier le format des données
            if len(rates) == 0:
                logger.error(f"Aucune donnée récupérée pour {symbol}")
                return None
                
            # Convertir en numpy array
            rates_array = np.array(rates, dtype=[
                ('time', '<i8'),
                ('open', '<f8'),
                ('high', '<f8'),
                ('low', '<f8'),
                ('close', '<f8'),
                ('tick_volume', '<i8'),
                ('spread', '<i4'),
                ('real_volume', '<i8')
            ])
            
            # Vérifier que les données sont valides
            if np.isnan(rates_array['close']).any():
                logger.error(f"Données invalides: valeurs manquantes dans les données pour {symbol}")
                return None
                
            return rates_array
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
            return None
        
    def place_order(
        self,
        symbol: str,
        order_type: int,
        volume: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = ""
    ) -> Optional[dict]:
        """
        Place un ordre sur le marché.
        
        Args:
            symbol: Symbole à trader
            order_type: Type d'ordre (utiliser les constantes mt5.ORDER_TYPE_*)
            volume: Volume de l'ordre
            price: Prix de l'ordre
            stop_loss: Stop loss (optionnel)
            take_profit: Take profit (optionnel)
            comment: Commentaire (optionnel)
            
        Returns:
            Optional[dict]: Résultat de l'ordre ou None si erreur
        """
        if not self.connected:
            logger.error("Non connecté à MT5")
            return None
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if stop_loss > 0:
            request["sl"] = stop_loss
        if take_profit > 0:
            request["tp"] = take_profit
            
        # Envoyer l'ordre
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de l'ordre: {result.comment}")
            return None
            
        return {
            'order': result,
            'request': request
        }
        
    def __del__(self):
        """Nettoie la connexion à la destruction de l'objet."""
        if mt5.terminal_info() is not None:
            mt5.shutdown()
            logger.info("Connexion MT5 fermée") 