"""
Module de connexion à MetaTrader 5 via Avatrader
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import os

logger = logging.getLogger(__name__)

class AvatraderMT5:
    """Classe de gestion de la connexion à MetaTrader 5"""
    
    def __init__(self, login: int, password: str, server: str):
        """
        Initialise la connexion à MT5
        
        Args:
            login (int): Numéro de compte
            password (str): Mot de passe
            server (str): Serveur MT5
        """
        # Convertir login en entier si c'est une chaîne
        try:
            self.login = int(login)
        except (ValueError, TypeError):
            logger.error(f"Login invalide: {login}. Le login doit être un nombre entier.")
            self.login = 0
            
        self.password = password
        self.server = server
        self.connected = False
        
        # Vérifier que les informations ne sont pas les valeurs par défaut
        if login == "DEMO" or password == "DEMO" or server == "DEMO":
            logger.warning("Vous utilisez des identifiants par défaut. Veuillez configurer vos identifiants réels.")
        
        # Tenter la connexion
        self.connect()
    
    def connect(self) -> bool:
        """Établit la connexion à MT5"""
        try:
            # Vérifier si MT5 est déjà initialisé
            if not mt5.terminal_info():
                logger.info("Initialisation de MT5...")
                
                # Chemin vers l'exécutable MT5 (ajustez selon votre installation)
                terminal_path = "C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe"
                if not os.path.exists(terminal_path):
                    terminal_path = ""  # Laisser MT5 trouver l'installation par défaut
                
                # Initialiser avec le chemin vers MT5 si disponible
                if not mt5.initialize(path=terminal_path):
                    error_code = mt5.last_error()
                    logger.error(f"Échec de l'initialisation MT5: Code {error_code}")
                    return False
            
            # Tentative de connexion
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            )
            
            if not authorized:
                error_code = mt5.last_error()
                logger.error(f"Échec de l'authentification MT5: Code {error_code}")
                
                # Afficher des informations supplémentaires sur l'erreur
                if error_code == 10000:
                    logger.error("Erreur de connexion à MT5: aucune erreur retournée")
                elif error_code == 10013:
                    logger.error("Erreur de connexion à MT5: identifiants invalides")
                elif error_code == 10014:
                    logger.error("Erreur de connexion à MT5: IP ou serveur non autorisé")
                elif error_code == 10015:
                    logger.error("Erreur de connexion à MT5: serveur invalide")
                elif error_code == 10016:
                    logger.error("Erreur de connexion à MT5: trop de connexions")
                else:
                    logger.error(f"Erreur de connexion à MT5: code {error_code}")
                
                mt5.shutdown()
                return False
            
            logger.info(f"Connexion MT5 établie au serveur {self.server}")
            logger.info(f"Version MT5: {mt5.version()}")
            
            # Afficher les informations du compte
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Compte: {account_info.login}, Nom: {account_info.name}")
                logger.info(f"Solde: {account_info.balance}, Equity: {account_info.equity}")
            
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Exception lors de la connexion à MT5: {str(e)}")
            try:
                mt5.shutdown()
            except:
                pass
            return False
    
    def disconnect(self):
        """Ferme la connexion MT5"""
        if self.connected:
            try:
                mt5.shutdown()
                self.connected = False
                logger.info("Connexion MT5 fermée")
            except Exception as e:
                logger.error(f"Erreur lors de la déconnexion: {str(e)}")
    
    def get_positions(self) -> list:
        """Récupère les positions ouvertes"""
        if not self.connected:
            if not self.connect():
                return []
            
        try:
            positions = mt5.positions_get()
            if positions is None:
                error_code = mt5.last_error()
                logger.error(f"Erreur lors de la récupération des positions: Code {error_code}")
                return []
                
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time)
            } for pos in positions]
        except Exception as e:
            logger.error(f"Exception lors de la récupération des positions: {str(e)}")
            return []
    
    def get_account_info(self) -> dict:
        """Récupère les informations du compte"""
        if not self.connected:
            if not self.connect():
                return None
            
        try:
            account_info = mt5.account_info()
            if account_info is None:
                error_code = mt5.last_error()
                logger.error(f"Erreur lors de la récupération des infos compte: Code {error_code}")
                return None
                
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free
            }
        except Exception as e:
            logger.error(f"Exception lors de la récupération des infos compte: {str(e)}")
            return None
    
    def get_price_history(self, symbol: str = "BTCUSD", timeframe: int = mt5.TIMEFRAME_M1, 
                         bars: int = 1000) -> pd.DataFrame:
        """
        Récupère l'historique des prix
        
        Args:
            symbol (str): Symbole à récupérer
            timeframe (int): Timeframe MT5
            bars (int): Nombre de barres
            
        Returns:
            pd.DataFrame: Données OHLCV
        """
        if not self.connected:
            if not self.connect():
                return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                error_code = mt5.last_error()
                logger.error(f"Erreur lors de la récupération de l'historique des prix: Code {error_code}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.set_index('time')
        except Exception as e:
            logger.error(f"Exception lors de la récupération de l'historique des prix: {str(e)}")
            return None
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None) -> dict:
        """
        Place un ordre
        
        Args:
            symbol (str): Symbole
            order_type (str): Type d'ordre ('BUY' ou 'SELL')
            volume (float): Volume
            price (float): Prix (optionnel)
            sl (float): Stop Loss (optionnel)
            tp (float): Take Profit (optionnel)
            
        Returns:
            dict: Résultat de l'ordre
        """
        if not self.connected:
            if not self.connect():
                return {'error': 'Non connecté'}
            
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if price is not None:
                request["price"] = price
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erreur lors du placement d'ordre: Code {result.retcode}, Description: {result.comment}")
                return {'error': f'Erreur {result.retcode}: {result.comment}'}
                
            logger.info(f"Ordre exécuté: #{result.order} {order_type} {symbol} {volume} lots à {result.price}")
            return {
                'order_id': result.order,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
        except Exception as e:
            logger.error(f"Exception lors du placement d'ordre: {str(e)}")
            return {'error': f'Exception: {str(e)}'}
    
    def get_available_symbols(self) -> list:
        """
        Récupère la liste des symboles disponibles.
        
        Returns:
            list: Liste des symboles disponibles
        """
        try:
            if not self.connected:
                if not self.connect():
                    logging.error("Non connecté à MT5, impossible de récupérer les symboles")
                    return []
            
            # Récupérer tous les symboles disponibles
            symbols = mt5.symbols_get()
            if symbols is None:
                error_code = mt5.last_error()
                logging.error(f"Erreur lors de la récupération des symboles: Code {error_code}")
                return []
            
            # Filtrer pour n'avoir que les symboles crypto
            crypto_symbols = [s.name for s in symbols if 'USD' in s.name and any(crypto in s.name for crypto in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH'])]
            
            if not crypto_symbols:
                # Si aucun symbole crypto n'est trouvé, renvoyer tous les symboles disponibles
                logging.warning("Aucun symbole crypto trouvé, utilisation de tous les symboles disponibles")
                return [s.name for s in symbols]
            
            return crypto_symbols
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des symboles: {str(e)}")
            return [] 
    
    def initialize(self) -> bool:
        """
        Alias de la méthode connect() pour compatibilité.
        
        Returns:
            bool: True si la connexion est établie
        """
        return self.connect() 