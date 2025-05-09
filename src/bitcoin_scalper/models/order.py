"""
Module de définition des classes d'ordres de trading.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    """Sens de l'ordre."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Type d'ordre."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    """Statut de l'ordre."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Classe représentant un ordre de trading."""
    symbol: str
    side: OrderSide
    type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[int] = None
    strategy: Optional[str] = None
    timestamp: datetime = datetime.now()
    comment: str = "" 