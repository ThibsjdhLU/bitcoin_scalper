import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

class OrderStatus(Enum):
    PENDING = 'pending'
    OPEN = 'open'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

@dataclass
class Order:
    id: str
    symbol: str
    type: OrderType
    side: OrderSide
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    create_time: datetime
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    fees: float = 0.0
    leverage: float = 1.0
    position_id: Optional[str] = None

class OrderManager:
    """
    Gestionnaire des ordres de trading
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire d'ordres
        
        Args:
            config (dict): Configuration du gestionnaire
        """
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        
        # Configuration
        self.max_orders = config.get('max_orders', 10)
        self.order_timeout = config.get('order_timeout', 60)  # 60 secondes
        self.default_leverage = config.get('default_leverage', 1)
        
        logger.info("Gestionnaire d'ordres initialisé")
    
    def create_market_order(self, symbol: str, side: OrderSide, size: float,
                          leverage: Optional[float] = None) -> Order:
        """
        Crée un ordre au marché
        
        Args:
            symbol (str): Symbole de trading
            side (OrderSide): Direction de l'ordre
            size (float): Taille de l'ordre
            leverage (float, optional): Levier utilisé
            
        Returns:
            Order: Nouvel ordre créé
        """
        if not self._can_create_order():
            raise ValueError("Impossible de créer un nouvel ordre")
        
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side=side,
            size=size,
            price=None,
            stop_price=None,
            status=OrderStatus.PENDING,
            create_time=datetime.now(),
            leverage=leverage or self.default_leverage
        )
        
        self.orders[order_id] = order
        logger.info(f"Ordre au marché créé: {order_id}")
        
        return order
    
    def create_limit_order(self, symbol: str, side: OrderSide, size: float,
                          price: float, leverage: Optional[float] = None) -> Order:
        """
        Crée un ordre limite
        
        Args:
            symbol (str): Symbole de trading
            side (OrderSide): Direction de l'ordre
            size (float): Taille de l'ordre
            price (float): Prix limite
            leverage (float, optional): Levier utilisé
            
        Returns:
            Order: Nouvel ordre créé
        """
        if not self._can_create_order():
            raise ValueError("Impossible de créer un nouvel ordre")
        
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order = Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.LIMIT,
            side=side,
            size=size,
            price=price,
            stop_price=None,
            status=OrderStatus.PENDING,
            create_time=datetime.now(),
            leverage=leverage or self.default_leverage
        )
        
        self.orders[order_id] = order
        logger.info(f"Ordre limite créé: {order_id}")
        
        return order
    
    def create_stop_order(self, symbol: str, side: OrderSide, size: float,
                         stop_price: float, limit_price: Optional[float] = None,
                         leverage: Optional[float] = None) -> Order:
        """
        Crée un ordre stop ou stop-limit
        
        Args:
            symbol (str): Symbole de trading
            side (OrderSide): Direction de l'ordre
            size (float): Taille de l'ordre
            stop_price (float): Prix stop
            limit_price (float, optional): Prix limite pour stop-limit
            leverage (float, optional): Levier utilisé
            
        Returns:
            Order: Nouvel ordre créé
        """
        if not self._can_create_order():
            raise ValueError("Impossible de créer un nouvel ordre")
        
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order = Order(
            id=order_id,
            symbol=symbol,
            type=order_type,
            side=side,
            size=size,
            price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            create_time=datetime.now(),
            leverage=leverage or self.default_leverage
        )
        
        self.orders[order_id] = order
        logger.info(f"Ordre stop créé: {order_id}")
        
        return order
    
    def cancel_order(self, order_id: str) -> Order:
        """
        Annule un ordre existant
        
        Args:
            order_id (str): ID de l'ordre
            
        Returns:
            Order: Ordre annulé
        """
        if order_id not in self.orders:
            raise ValueError(f"Ordre {order_id} non trouvé")
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            raise ValueError(f"Impossible d'annuler l'ordre {order_id}")
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Ordre annulé: {order_id}")
        
        return order
    
    def fill_order(self, order_id: str, fill_price: float,
                   fees: Optional[float] = None) -> Order:
        """
        Remplit un ordre existant
        
        Args:
            order_id (str): ID de l'ordre
            fill_price (float): Prix de remplissage
            fees (float, optional): Frais de trading
            
        Returns:
            Order: Ordre rempli
        """
        if order_id not in self.orders:
            raise ValueError(f"Ordre {order_id} non trouvé")
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            raise ValueError(f"Impossible de remplir l'ordre {order_id}")
        
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_time = datetime.now()
        order.fees = fees or 0.0
        
        # Déplacement vers l'historique
        self.filled_orders.append(order)
        del self.orders[order_id]
        
        logger.info(f"Ordre rempli: {order_id} au prix {fill_price}")
        
        return order
    
    def update_orders(self, current_price: float) -> List[Order]:
        """
        Met à jour les ordres avec le prix actuel
        
        Args:
            current_price (float): Prix actuel
            
        Returns:
            List[Order]: Ordres mis à jour
        """
        updated_orders = []
        
        for order in list(self.orders.values()):
            # Vérification du timeout
            if (datetime.now() - order.create_time).total_seconds() > self.order_timeout:
                order.status = OrderStatus.EXPIRED
                self.filled_orders.append(order)
                del self.orders[order.id]
                logger.info(f"Ordre expiré: {order.id}")
                updated_orders.append(order)
                continue
            
            # Vérification des ordres stop
            if order.type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    if order.type == OrderType.STOP:
                        self.fill_order(order.id, current_price)
                        updated_orders.append(order)
                    else:
                        order.status = OrderStatus.OPEN
                        updated_orders.append(order)
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    if order.type == OrderType.STOP:
                        self.fill_order(order.id, current_price)
                        updated_orders.append(order)
                    else:
                        order.status = OrderStatus.OPEN
                        updated_orders.append(order)
            
            # Vérification des ordres limite
            elif order.type == OrderType.LIMIT and order.status == OrderStatus.OPEN:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    self.fill_order(order.id, order.price)
                    updated_orders.append(order)
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    self.fill_order(order.id, order.price)
                    updated_orders.append(order)
        
        return updated_orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Récupère un ordre par son ID
        
        Args:
            order_id (str): ID de l'ordre
            
        Returns:
            Optional[Order]: Ordre trouvé ou None
        """
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Récupère les ordres ouverts
        
        Args:
            symbol (str, optional): Filtrer par symbole
            
        Returns:
            List[Order]: Liste des ordres ouverts
        """
        orders = [o for o in self.orders.values() 
                 if o.status in [OrderStatus.PENDING, OrderStatus.OPEN]]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
            
        return orders
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Récupère l'historique des ordres remplis
        
        Args:
            symbol (str, optional): Filtrer par symbole
            
        Returns:
            List[Order]: Liste des ordres remplis
        """
        orders = self.filled_orders
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
            
        return orders
    
    def get_order_metrics(self) -> Dict:
        """
        Calcule les métriques des ordres
        
        Returns:
            Dict: Métriques des ordres
        """
        total_fees = sum(o.fees for o in self.filled_orders)
        filled_volume = sum(o.size * o.fill_price 
                          for o in self.filled_orders if o.fill_price)
        
        return {
            'open_orders': len(self.orders),
            'total_orders': len(self.filled_orders),
            'total_fees': total_fees,
            'filled_volume': filled_volume
        }
    
    def _can_create_order(self) -> bool:
        """
        Vérifie si un nouvel ordre peut être créé
        
        Returns:
            bool: True si un ordre peut être créé
        """
        if len(self.orders) >= self.max_orders:
            logger.warning("Nombre maximum d'ordres atteint")
            return False
            
        return True 