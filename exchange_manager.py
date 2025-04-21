import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ccxt
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    CANCELED = 'canceled'
    EXPIRED = 'expired'
    REJECTED = 'rejected'

@dataclass
class Order:
    id: str
    symbol: str
    type: str
    side: str
    price: Optional[float]
    amount: float
    status: OrderStatus
    timestamp: datetime
    datetime: datetime
    last_trade_timestamp: Optional[datetime]
    remaining: float
    cost: float
    fee: Dict
    average: Optional[float]
    filled: float
    trades: List[Dict]

class ExchangeManager:
    """
    Gestionnaire des interactions avec l'exchange
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire d'exchange
        
        Args:
            config (dict): Configuration du gestionnaire
        """
        self.config = config
        
        # Configuration de l'exchange
        exchange_id = config.get('exchange', 'binance')
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Pour le trading à terme
                'adjustForTimeDifference': True
            }
        })
        
        # Paramètres de trading
        self.symbols = config.get('symbols', ['BTC/USDT'])
        self.default_leverage = config.get('default_leverage', 1)
        self.default_margin_type = config.get('default_margin_type', 'isolated')
        
        # Initialisation
        self._initialize_exchange()
        
        logger.info(f"Gestionnaire d'exchange initialisé: {exchange_id}")
    
    def _initialize_exchange(self) -> None:
        """
        Initialise l'exchange avec les paramètres de base
        """
        try:
            # Chargement des marchés
            self.exchange.load_markets()
            
            # Configuration du levier par défaut
            for symbol in self.symbols:
                self.exchange.fapiPrivate_post_leverage({
                    'symbol': symbol.replace('/', ''),
                    'leverage': self.default_leverage
                })
                
                # Configuration du type de marge
                self.exchange.fapiPrivate_post_marginType({
                    'symbol': symbol.replace('/', ''),
                    'marginType': self.default_margin_type.upper()
                })
            
            logger.info("Exchange initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange: {str(e)}")
            raise
    
    def create_order(self, symbol: str, order_type: str, side: str,
                    amount: float, price: Optional[float] = None,
                    params: Dict = {}) -> Order:
        """
        Crée un ordre sur l'exchange
        
        Args:
            symbol (str): Symbole de trading
            order_type (str): Type d'ordre (market, limit, stop, stop_limit)
            side (str): Direction (buy, sell)
            amount (float): Quantité
            price (float, optional): Prix pour les ordres limit
            params (dict): Paramètres additionnels
            
        Returns:
            Order: Ordre créé
        """
        try:
            # Création de l'ordre
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            # Conversion en objet Order
            return Order(
                id=order['id'],
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                price=order.get('price'),
                amount=order['amount'],
                status=OrderStatus(order['status']),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                datetime=datetime.fromtimestamp(order['datetime'].timestamp() / 1000),
                last_trade_timestamp=datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order.get('lastTradeTimestamp') else None,
                remaining=order['remaining'],
                cost=order['cost'],
                fee=order['fee'],
                average=order.get('average'),
                filled=order['filled'],
                trades=order.get('trades', [])
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Order:
        """
        Annule un ordre existant
        
        Args:
            order_id (str): ID de l'ordre
            symbol (str): Symbole de trading
            
        Returns:
            Order: Ordre annulé
        """
        try:
            order = self.exchange.cancel_order(order_id, symbol)
            return Order(
                id=order['id'],
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                price=order.get('price'),
                amount=order['amount'],
                status=OrderStatus(order['status']),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                datetime=datetime.fromtimestamp(order['datetime'].timestamp() / 1000),
                last_trade_timestamp=datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order.get('lastTradeTimestamp') else None,
                remaining=order['remaining'],
                cost=order['cost'],
                fee=order['fee'],
                average=order.get('average'),
                filled=order['filled'],
                trades=order.get('trades', [])
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            raise
    
    def get_order(self, order_id: str, symbol: str) -> Order:
        """
        Récupère un ordre par son ID
        
        Args:
            order_id (str): ID de l'ordre
            symbol (str): Symbole de trading
            
        Returns:
            Order: Ordre trouvé
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return Order(
                id=order['id'],
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                price=order.get('price'),
                amount=order['amount'],
                status=OrderStatus(order['status']),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                datetime=datetime.fromtimestamp(order['datetime'].timestamp() / 1000),
                last_trade_timestamp=datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order.get('lastTradeTimestamp') else None,
                remaining=order['remaining'],
                cost=order['cost'],
                fee=order['fee'],
                average=order.get('average'),
                filled=order['filled'],
                trades=order.get('trades', [])
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'ordre: {str(e)}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Récupère les ordres ouverts
        
        Args:
            symbol (str, optional): Symbole de trading
            
        Returns:
            List[Order]: Liste des ordres ouverts
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return [
                Order(
                    id=order['id'],
                    symbol=order['symbol'],
                    type=order['type'],
                    side=order['side'],
                    price=order.get('price'),
                    amount=order['amount'],
                    status=OrderStatus(order['status']),
                    timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                    datetime=datetime.fromtimestamp(order['datetime'].timestamp() / 1000),
                    last_trade_timestamp=datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order.get('lastTradeTimestamp') else None,
                    remaining=order['remaining'],
                    cost=order['cost'],
                    fee=order['fee'],
                    average=order.get('average'),
                    filled=order['filled'],
                    trades=order.get('trades', [])
                )
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
    
    def get_balance(self, currency: str = 'USDT') -> Dict:
        """
        Récupère le solde d'une devise
        
        Args:
            currency (str): Devise
            
        Returns:
            Dict: Informations sur le solde
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance[currency]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {str(e)}")
            raise
    
    def get_position(self, symbol: str) -> Dict:
        """
        Récupère la position pour un symbole
        
        Args:
            symbol (str): Symbole de trading
            
        Returns:
            Dict: Informations sur la position
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            return positions[0] if positions else {}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la position: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère les informations de marché pour un symbole
        
        Args:
            symbol (str): Symbole de trading
            
        Returns:
            Dict: Informations de marché
        """
        try:
            return self.exchange.fetch_ticker(symbol)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker: {str(e)}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1m',
                  since: Optional[int] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Récupère les données OHLCV
        
        Args:
            symbol (str): Symbole de trading
            timeframe (str): Intervalle de temps
            since (int, optional): Timestamp de début
            limit (int, optional): Nombre de candles
            
        Returns:
            List[Dict]: Données OHLCV
        """
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV: {str(e)}")
            raise 