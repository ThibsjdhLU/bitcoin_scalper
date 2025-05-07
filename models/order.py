"""
Modèles pour les ordres de trading.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

class OrderSide(Enum):
    """Type d'ordre (achat/vente)."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Type d'ordre (market/limit/stop)."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    """Statut de l'ordre."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class Order:
    """
    Représente un ordre de trading.
    
    Attributes:
        symbol: Symbole à trader
        side: Type d'ordre (achat/vente)
        type: Type d'ordre (market/limit/stop)
        volume: Volume de l'ordre
        price: Prix de l'ordre
        sl: Stop loss
        tp: Take profit
        status: Statut de l'ordre
        timestamp: Date/heure de création
        strategy: Nom de la stratégie
        metadata: Métadonnées additionnelles
    """
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialise un ordre.
        
        Args:
            symbol: Symbole à trader
            side: Type d'ordre (achat/vente)
            type: Type d'ordre (market/limit/stop)
            volume: Volume de l'ordre
            price: Prix de l'ordre (optionnel)
            sl: Stop loss (optionnel)
            tp: Take profit (optionnel)
            strategy: Nom de la stratégie (optionnel)
            metadata: Métadonnées additionnelles (optionnel)
        """
        self.symbol = symbol
        self.side = side
        self.type = type
        self.volume = volume
        self.price = price
        self.sl = sl
        self.tp = tp
        self.status = OrderStatus.PENDING
        self.timestamp = datetime.now()
        self.strategy = strategy
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        """Représentation string de l'ordre."""
        return (
            f"{self.side.value} {self.symbol} "
            f"Volume: {self.volume} "
            f"Price: {self.price} "
            f"SL: {self.sl} "
            f"TP: {self.tp} "
            f"Status: {self.status.value}"
        ) 