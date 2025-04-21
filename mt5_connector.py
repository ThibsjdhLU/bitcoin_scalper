import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
from typing import Optional, Dict, List, Tuple
import time
import os
from dotenv import load_dotenv
import logging
from functools import wraps

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mt5_connector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MT5Connector')

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Échec après {max_retries} tentatives: {str(e)}")
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class MT5Connector:
    def __init__(self):
        self.initialized = False
        self.connected = False
        self.account_info = None
        self.last_error = None
        self.max_retries = 3
        self.retry_delay = 1
        self.logger = logging.getLogger(__name__)
        
    @retry_on_error(max_retries=3, delay=1)
    def initialize(self) -> bool:
        """Initialise la connexion à MT5."""
        try:
            # Vérification si MT5 est déjà initialisé
            if mt5.initialize():
                self.initialized = True
                self.logger.info("MT5 déjà initialisé")
                return True

            # Vérification si MT5 est installé
            if not mt5.initialize():
                error = mt5.last_error()
                self.last_error = f"MT5 n'est pas installé ou n'est pas accessible: {error}"
                self.logger.error(self.last_error)
                return False

            # Vérification de la version
            version = mt5.version()
            if version is None:
                self.last_error = "Impossible de récupérer la version de MT5"
                self.logger.error(self.last_error)
                return False

            self.logger.info(f"Version MT5: {version}")
            
            # Vérification du terminal
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.last_error = "Impossible de récupérer les informations du terminal"
                self.logger.error(self.last_error)
                return False

            if not terminal_info.connected:
                self.last_error = "Le terminal n'est pas connecté au serveur"
                self.logger.error(self.last_error)
                return False

            if not terminal_info.trade_allowed:
                self.last_error = "Le trading automatique n'est pas autorisé"
                self.logger.error(self.last_error)
                return False

            self.initialized = True
            self.logger.info("MT5 initialisé avec succès")
            return True

        except Exception as e:
            self.last_error = f"Erreur lors de l'initialisation de MT5: {str(e)}"
            self.logger.error(self.last_error)
            return False

    @retry_on_error(max_retries=3, delay=1)
    def login(self, login: int, password: str, server: str) -> bool:
        """Connecte à un compte MT5."""
        try:
            # Vérification de l'initialisation
            if not self.initialized:
                self.last_error = "MT5 n'est pas initialisé"
                self.logger.error(self.last_error)
                return False

            # Vérification des paramètres
            if not all([login, password, server]):
                self.last_error = "Paramètres de connexion manquants"
                self.logger.error(self.last_error)
                return False

            # Vérification si déjà connecté
            if mt5.account_info() is not None:
                self.connected = True
                self.logger.info("Déjà connecté à un compte MT5")
                return True

            # Tentative de connexion
            self.logger.info(f"Tentative de connexion au compte {login} sur {server}")
            if not mt5.login(login, password, server):
                error = mt5.last_error()
                self.last_error = f"Erreur de connexion: {error}"
                self.logger.error(self.last_error)
                return False

            # Vérification de la connexion
            account_info = mt5.account_info()
            if account_info is None:
                self.last_error = "Impossible de récupérer les informations du compte"
                self.logger.error(self.last_error)
                return False

            self.connected = True
            self.account_info = account_info
            self.logger.info(f"Connecté au compte {account_info.login}")
            self.logger.info(f"Balance: {account_info.balance}")
            self.logger.info(f"Leverage: {account_info.leverage}")
            return True

        except Exception as e:
            self.last_error = f"Erreur lors de la connexion: {str(e)}"
            self.logger.error(self.last_error)
            return False

    def get_last_error(self) -> str:
        """Retourne la dernière erreur rencontrée"""
        return self.last_error or "Aucune erreur"
        
    def get_account_info(self) -> Optional[Dict]:
        """Récupère les informations du compte"""
        if not self.connected:
            return None
            
        if self.account_info is None:
            self.account_info = mt5.account_info()
            
        if self.account_info is None:
            return None
            
        return {
            "login": self.account_info.login,
            "balance": self.account_info.balance,
            "equity": self.account_info.equity,
            "margin": self.account_info.margin,
            "free_margin": self.account_info.margin_free,
            "leverage": self.account_info.leverage,
            "server": self.account_info.server,
            "currency": self.account_info.currency,
            "company": self.account_info.company,
            "trade_allowed": self.account_info.trade_allowed,
            "trade_expert": self.account_info.trade_expert
        }
        
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Récupère les informations sur un symbole"""
        try:
            if not self.check_connection():
                logger.error("Pas de connexion MT5 active")
                return None

            # Récupération des informations du symbole
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Impossible de récupérer les informations pour {symbol}")
                return None

            # Récupération du tick actuel
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Impossible de récupérer le tick pour {symbol}")
                return None

            return {
                "symbol": symbol_info.name,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": tick.ask - tick.bid,
                "digits": symbol_info.digits,
                "trade_mode": symbol_info.trade_mode,
                "trade_contract_size": symbol_info.trade_contract_size,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
                "point": symbol_info.point,
                "trade_stops_level": symbol_info.trade_stops_level,
                "trade_freeze_level": symbol_info.trade_freeze_level,
                "time_current": tick.time,
                "time_local": tick.time_local
            }

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations du symbole: {str(e)}")
            return None
        
    def get_available_symbols(self) -> List[str]:
        """Récupère la liste des symboles disponibles"""
        if not self.connected:
            return []
            
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
            
        return [symbol.name for symbol in symbols]
        
    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        Récupère les données OHLCV pour un symbole
        
        Args:
            symbol (str): Symbole
            timeframe (str): Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count (int): Nombre de bougies
            
        Returns:
            pd.DataFrame: Données OHLCV
        """
        try:
            if not self.check_connection():
                logger.error("Pas de connexion MT5 active")
                return pd.DataFrame()

            # Conversion du timeframe
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            if timeframe not in tf_map:
                logger.error(f"Timeframe {timeframe} non supporté")
                return pd.DataFrame()

            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"Impossible de récupérer les données pour {symbol}")
                return pd.DataFrame()

            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Log de la structure des données
            logger.info(f"Structure des données OHLCV pour {symbol}:")
            logger.info(f"Colonnes: {df.columns.tolist()}")
            logger.info(f"Exemple de données:\n{df.head()}")
            
            return df

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV: {str(e)}")
            logging.error(f"Erreur lors de la récupération des données OHLCV: {str(e)}")
            raise
        
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None) -> Optional[Dict]:
        """Place un ordre."""
        try:
            if not self.check_connection():
                return None

            # Vérification du symbole
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.last_error = f"Symbole {symbol} non trouvé"
                return None

            # Vérification du type d'ordre
            if order_type not in ['buy', 'sell']:
                self.last_error = f"Type d'ordre {order_type} non supporté"
                return None

            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price or symbol_info.ask if order_type == 'buy' else symbol_info.bid,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "python scalper",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Envoi de l'ordre
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.last_error = f"Erreur lors de l'envoi de l'ordre: {result.comment}"
                return None

            return {
                "order_id": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment
            }

        except Exception as e:
            self.last_error = f"Erreur lors du placement de l'ordre: {str(e)}"
            self.logger.error(self.last_error)
            return None
        
    def close_position(self, position_id: int) -> bool:
        """Ferme une position existante"""
        if not self.connected:
            return False
            
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            print(f"Position {position_id} non trouvée")
            return False
            
        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python scalper close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    def get_positions(self) -> List[Dict]:
        """Récupère toutes les positions ouvertes"""
        if not self.connected:
            return []
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [{
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": pos.volume,
            "price_open": pos.price_open,
            "sl": pos.sl,
            "tp": pos.tp,
            "price_current": pos.price_current,
            "profit": pos.profit
        } for pos in positions]
        
    def shutdown(self) -> None:
        """Ferme la connexion MT5."""
        try:
            if self.initialized:
                mt5.shutdown()
                self.initialized = False
                self.connected = False
                self.account_info = None
                self.logger.info("MT5 arrêté")
        except Exception as e:
            self.last_error = f"Erreur lors de l'arrêt de MT5: {str(e)}"
            self.logger.error(self.last_error)
        
    def is_autotrading_enabled(self) -> bool:
        """Vérifie si le trading automatique est activé"""
        if not self.initialized:
            return False
            
        try:
            # Vérification du trading automatique via MT5
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False
                
            return terminal_info.trade_allowed
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du trading automatique: {str(e)}")
            return False

    def can_trade(self, symbol: str) -> bool:
        """Vérifie si le trading est possible sur un symbole"""
        if not self.connected:
            return False
            
        try:
            # Vérification du symbole
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            # Vérification du mode de trading
            if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                return False
                
            # Vérification des permissions de trading
            account_info = self.get_account_info()
            if account_info is None or not account_info.get('trade_allowed', False):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du trading pour {symbol}: {str(e)}")
            return False

    def is_market_open(self, symbol: str) -> bool:
        """
        Vérifie si le marché est ouvert pour un symbole donné
        
        Args:
            symbol (str): Symbole à vérifier
            
        Returns:
            bool: True si le marché est ouvert, False sinon
        """
        try:
            if not self.connected:
                return False
                
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            # Pour BTCUSD, le marché est toujours ouvert 24/24
            if symbol == 'BTCUSD':
                return True
                
            # Pour les autres symboles, on vérifie les heures de trading
            current_time = datetime.now(pytz.UTC)
            trading_hours = self.get_trading_hours(symbol)
            
            if trading_hours is None:
                return False
                
            start_time = datetime.strptime(trading_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(trading_hours['end'], '%H:%M').time()
            
            current_time = current_time.time()
            return start_time <= current_time <= end_time
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du marché ouvert: {e}")
            return False

    def check_connection(self) -> bool:
        """Vérifie l'état de la connexion MT5."""
        try:
            if not self.initialized:
                self.last_error = "MT5 n'est pas initialisé"
                return False

            if not mt5.initialize():
                self.last_error = "MT5 n'est plus initialisé"
                self.initialized = False
                return False

            if not mt5.terminal_info().connected:
                self.last_error = "Le terminal n'est plus connecté au serveur"
                self.connected = False
                return False

            if mt5.account_info() is None:
                self.last_error = "Plus connecté au compte MT5"
                self.connected = False
                return False

            return True

        except Exception as e:
            self.last_error = f"Erreur lors de la vérification de la connexion: {str(e)}"
            self.logger.error(self.last_error)
            return False

    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Récupère les données historiques."""
        try:
            if not self.check_connection():
                return None

            # Conversion du timeframe
            tf_map = {
                '1m': mt5.TIMEFRAME_M1,
                '5m': mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30,
                '1h': mt5.TIMEFRAME_H1,
                '4h': mt5.TIMEFRAME_H4,
                '1d': mt5.TIMEFRAME_D1
            }
            
            if timeframe not in tf_map:
                self.last_error = f"Timeframe {timeframe} non supporté"
                return None

            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
            if rates is None:
                self.last_error = f"Impossible de récupérer les données pour {symbol}"
                return None

            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df

        except Exception as e:
            self.last_error = f"Erreur lors de la récupération des données historiques: {str(e)}"
            self.logger.error(self.last_error)
            return None 