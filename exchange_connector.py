import logging
import ccxt
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ExchangeConnector:
    """
    Classe pour gérer la connexion avec l'exchange de trading
    """
    
    def __init__(self, config: Dict):
        """
        Initialise la connexion avec l'exchange
        
        Args:
            config (dict): Configuration de l'exchange (API key, secret, etc.)
        """
        self.config = config
        self.exchange_id = config.get('exchange_id', 'binance')
        
        # Initialisation de l'exchange
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
            'options': config.get('options', {}),
        })
        
        # Paramètres de trading
        self.default_symbol = config.get('default_symbol', 'BTC/USDT')
        self.default_timeframe = config.get('default_timeframe', '1m')
        self.max_retries = config.get('max_retries', 3)
        
        logger.info(f"Connecté à l'exchange {self.exchange_id}")
    
    def get_ticker(self, symbol: Optional[str] = None) -> Dict:
        """
        Récupère le ticker pour un symbole
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            Dict: Données du ticker
        """
        try:
            symbol = symbol or self.default_symbol
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Ticker récupéré pour {symbol}: {ticker}")
            return ticker
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker: {str(e)}")
            return {}
    
    def get_orderbook(self, symbol: Optional[str] = None, limit: int = 20) -> Dict:
        """
        Récupère le carnet d'ordres pour un symbole
        
        Args:
            symbol (str, optional): Symbole de trading
            limit (int): Nombre de niveaux à récupérer
            
        Returns:
            Dict: Carnet d'ordres
        """
        try:
            symbol = symbol or self.default_symbol
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            logger.debug(f"Carnet d'ordres récupéré pour {symbol}")
            return orderbook
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres: {str(e)}")
            return {}
    
    def get_ohlcv(self, symbol: Optional[str] = None, 
                  timeframe: Optional[str] = None,
                  since: Optional[int] = None,
                  limit: Optional[int] = None) -> List[List]:
        """
        Récupère les données OHLCV pour un symbole
        
        Args:
            symbol (str, optional): Symbole de trading
            timeframe (str, optional): Timeframe des données
            since (int, optional): Timestamp de début
            limit (int, optional): Nombre de bougies à récupérer
            
        Returns:
            List[List]: Données OHLCV
        """
        try:
            symbol = symbol or self.default_symbol
            timeframe = timeframe or self.default_timeframe
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logger.debug(f"Données OHLCV récupérées pour {symbol} ({timeframe})")
            return ohlcv
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV: {str(e)}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str,
                    amount: float, price: Optional[float] = None,
                    params: Dict = {}) -> Dict:
        """
        Crée un ordre sur l'exchange
        
        Args:
            symbol (str): Symbole de trading
            order_type (str): Type d'ordre (market, limit, etc.)
            side (str): Direction de l'ordre (buy, sell)
            amount (float): Quantité à trader
            price (float, optional): Prix pour les ordres limit
            params (dict): Paramètres additionnels
            
        Returns:
            Dict: Réponse de l'exchange
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params)
            logger.info(f"Ordre créé: {order}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre: {str(e)}")
            return {}
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """
        Annule un ordre sur l'exchange
        
        Args:
            order_id (str): ID de l'ordre
            symbol (str, optional): Symbole de trading
            
        Returns:
            Dict: Réponse de l'exchange
        """
        try:
            symbol = symbol or self.default_symbol
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Ordre annulé: {result}")
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            return {}
    
    def get_balance(self) -> Dict:
        """
        Récupère le solde du compte
        
        Returns:
            Dict: Solde du compte
        """
        try:
            balance = self.exchange.fetch_balance()
            logger.debug("Solde récupéré")
            return balance
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {str(e)}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres ouverts
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            List[Dict]: Liste des ordres ouverts
        """
        try:
            symbol = symbol or self.default_symbol
            orders = self.exchange.fetch_open_orders(symbol)
            logger.debug(f"Ordres ouverts récupérés pour {symbol}")
            return orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            return []
    
    def get_closed_orders(self, symbol: Optional[str] = None,
                         since: Optional[int] = None,
                         limit: Optional[int] = None) -> List[Dict]:
        """
        Récupère les ordres fermés
        
        Args:
            symbol (str, optional): Symbole de trading
            since (int, optional): Timestamp de début
            limit (int, optional): Nombre d'ordres à récupérer
            
        Returns:
            List[Dict]: Liste des ordres fermés
        """
        try:
            symbol = symbol or self.default_symbol
            orders = self.exchange.fetch_closed_orders(symbol, since, limit)
            logger.debug(f"Ordres fermés récupérés pour {symbol}")
            return orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres fermés: {str(e)}")
            return []
    
    def get_my_trades(self, symbol: Optional[str] = None,
                     since: Optional[int] = None,
                     limit: Optional[int] = None) -> List[Dict]:
        """
        Récupère l'historique des trades
        
        Args:
            symbol (str, optional): Symbole de trading
            since (int, optional): Timestamp de début
            limit (int, optional): Nombre de trades à récupérer
            
        Returns:
            List[Dict]: Liste des trades
        """
        try:
            symbol = symbol or self.default_symbol
            trades = self.exchange.fetch_my_trades(symbol, since, limit)
            logger.debug(f"Trades récupérés pour {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des trades: {str(e)}")
            return []
    
    def get_market_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Récupère les informations sur le marché
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            Dict: Informations sur le marché
        """
        try:
            symbol = symbol or self.default_symbol
            market = self.exchange.market(symbol)
            logger.debug(f"Informations marché récupérées pour {symbol}")
            return market
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations marché: {str(e)}")
            return {}
    
    def get_funding_rate(self, symbol: Optional[str] = None) -> Tuple[float, int]:
        """
        Récupère le taux de financement pour les contrats perpétuels
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            Tuple[float, int]: (taux de financement, timestamp)
        """
        try:
            symbol = symbol or self.default_symbol
            funding = self.exchange.fetch_funding_rate(symbol)
            rate = funding.get('fundingRate', 0.0)
            timestamp = funding.get('timestamp', 0)
            logger.debug(f"Taux de financement récupéré pour {symbol}: {rate}")
            return rate, timestamp
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du taux de financement: {str(e)}")
            return 0.0, 0
    
    def get_leverage_tiers(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les niveaux de levier disponibles
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            List[Dict]: Liste des niveaux de levier
        """
        try:
            symbol = symbol or self.default_symbol
            tiers = self.exchange.fetch_leverage_tiers([symbol])
            logger.debug(f"Niveaux de levier récupérés pour {symbol}")
            return tiers.get(symbol, [])
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des niveaux de levier: {str(e)}")
            return []
    
    def set_leverage(self, leverage: int, symbol: Optional[str] = None) -> bool:
        """
        Configure le levier pour un symbole
        
        Args:
            leverage (int): Niveau de levier
            symbol (str, optional): Symbole de trading
            
        Returns:
            bool: True si succès, False sinon
        """
        try:
            symbol = symbol or self.default_symbol
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Levier configuré à {leverage}x pour {symbol}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du levier: {str(e)}")
            return False
    
    def set_margin_mode(self, margin_mode: str, symbol: Optional[str] = None) -> bool:
        """
        Configure le mode de marge (isolated/cross)
        
        Args:
            margin_mode (str): Mode de marge ('isolated' ou 'cross')
            symbol (str, optional): Symbole de trading
            
        Returns:
            bool: True si succès, False sinon
        """
        try:
            symbol = symbol or self.default_symbol
            self.exchange.set_margin_mode(margin_mode, symbol)
            logger.info(f"Mode de marge configuré à {margin_mode} pour {symbol}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du mode de marge: {str(e)}")
            return False 