"""
Module d'exécution des ordres de trading.
Gère les ordres market, limit et stop avec gestion des erreurs.
"""
import json
from enum import Enum
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import MetaTrader5 as mt5
from loguru import logger

from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from models.order import Order, OrderSide, OrderStatus, OrderType

class OrderType(Enum):
    """Types d'ordres supportés avec mapping direct vers MT5."""
    MARKET_BUY = 0  # mt5.ORDER_TYPE_BUY
    MARKET_SELL = 1  # mt5.ORDER_TYPE_SELL
    LIMIT_BUY = 2  # mt5.ORDER_TYPE_BUY_LIMIT
    LIMIT_SELL = 3  # mt5.ORDER_TYPE_SELL_LIMIT
    STOP_BUY = 4  # mt5.ORDER_TYPE_BUY_STOP
    STOP_SELL = 5  # mt5.ORDER_TYPE_SELL_STOP

    @classmethod
    def from_string(cls, order_type: str, side: str) -> 'OrderType':
        """
        Convertit une chaîne de type d'ordre et un côté en OrderType.
        
        Args:
            order_type: Type d'ordre ('MARKET', 'LIMIT', 'STOP')
            side: Côté de l'ordre ('BUY', 'SELL')
            
        Returns:
            OrderType correspondant
        
        Raises:
            ValueError: Si la combinaison est invalide
        """
        key = f"{order_type}_{side}"
        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid order type/side combination: {key}")
    
    @property
    def is_limit(self) -> bool:
        """Indique si c'est un ordre limite."""
        return self in [OrderType.LIMIT_BUY, OrderType.LIMIT_SELL]
    
    @property
    def is_stop(self) -> bool:
        """Indique si c'est un ordre stop."""
        return self in [OrderType.STOP_BUY, OrderType.STOP_SELL]
    
    @property
    def is_market(self) -> bool:
        """Indique si c'est un ordre au marché."""
        return self in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]
    
    @property
    def is_buy(self) -> bool:
        """Indique si c'est un ordre d'achat."""
        return self in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY, OrderType.STOP_BUY]

class OrderSide(Enum):
    """Sens de l'ordre."""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderStatus:
    """État d'un ordre."""
    order_id: int
    symbol: str
    type: str
    volume: float
    price: float
    stop_loss: float
    take_profit: float
    filled_volume: float
    remaining_volume: float
    status: str
    comment: str
    timestamp: datetime

class OrderExecutor:
    """
    Exécuteur d'ordres de trading.
    
    Cette classe gère l'exécution des ordres, le suivi des positions
    et la mise à jour des ordres en attente.
    """
    
    def __init__(
        self,
        mt5_connector: MT5Connector,
        risk_manager: RiskManager
    ):
        """
        Initialise l'exécuteur d'ordres.
        
        Args:
            mt5_connector: Connecteur MT5
            risk_manager: Gestionnaire de risques
        """
        self.mt5_connector = mt5_connector
        self.risk_manager = risk_manager
        self.pending_orders = {}
        
        logger.info("OrderExecutor initialisé")
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open("config/config.json", 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            raise
    
    def _validate_order_params(
        self,
        symbol: str,
        volume: float,
        order_type: OrderType,
        side: OrderSide,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        Valide les paramètres de l'ordre.
        
        Args:
            symbol: Symbole à trader
            volume: Volume de l'ordre
            order_type: Type d'ordre
            side: Sens de l'ordre
            price: Prix pour les ordres limit/stop
            sl: Stop loss
            tp: Take profit
            
        Returns:
            bool: True si les paramètres sont valides, False sinon
        """
        # Vérifier la connexion
        if not self.mt5_connector.connected:
            logger.error("Non connecté à MT5")
            return False
        
        # Vérifier le symbole
        symbol_info = self.mt5_connector.get_symbol_info(symbol)
        if symbol_info is None:
            return False
        
        # Vérifier le volume
        if not (symbol_info['volume_min'] <= volume <= symbol_info['volume_max']):
            logger.error(f"Volume invalide: {volume}")
            return False
        
        # Vérifier le prix pour les ordres limit/stop
        if (order_type.is_limit or order_type.is_stop) and price is None:
            logger.error("Prix requis pour les ordres limit/stop")
            return False
        
        return True
    
    def execute_market_order(
        self,
        symbol: str,
        volume: float,
        side: OrderSide,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Exécute un ordre au marché.
        
        Args:
            symbol: Symbole à trader
            volume: Volume de l'ordre
            side: Sens de l'ordre
            sl: Stop loss
            tp: Take profit
            
        Returns:
            Tuple[bool, Optional[int]]: (Succès, ID de l'ordre)
        """
        order_type = OrderType.MARKET_BUY if side == OrderSide.BUY else OrderType.MARKET_SELL
        if not self._validate_order_params(symbol, volume, order_type, side, sl=sl, tp=tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "deviation": 20,
            "magic": 234000,
            "comment": "python market order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Ajouter SL/TP si spécifiés
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Envoyer l'ordre
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de l'ordre: {result.comment}")
            return False, None
        
        logger.info(f"Ordre exécuté: {result.order}")
        return True, result.order
    
    def execute_limit_order(
        self,
        symbol: str,
        volume: float,
        side: OrderSide,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Place un ordre limite.
        
        Args:
            symbol: Symbole à trader
            volume: Volume de l'ordre
            side: Sens de l'ordre
            price: Prix limite
            sl: Stop loss
            tp: Take profit
            
        Returns:
            Tuple[bool, Optional[int]]: (Succès, ID de l'ordre)
        """
        order_type = OrderType.LIMIT_BUY if side == OrderSide.BUY else OrderType.LIMIT_SELL
        if not self._validate_order_params(symbol, volume, order_type, side, price, sl, tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python limit order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Ajouter SL/TP si spécifiés
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Envoyer l'ordre
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de l'ordre: {result.comment}")
            return False, None
        
        logger.info(f"Ordre limite placé: {result.order}")
        return True, result.order
    
    def execute_stop_order(
        self,
        symbol: str,
        volume: float,
        side: OrderSide,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Place un ordre stop.
        
        Args:
            symbol: Symbole à trader
            volume: Volume de l'ordre
            side: Sens de l'ordre
            price: Prix stop
            sl: Stop loss
            tp: Take profit
            
        Returns:
            Tuple[bool, Optional[int]]: (Succès, ID de l'ordre)
        """
        order_type = OrderType.STOP_BUY if side == OrderSide.BUY else OrderType.STOP_SELL
        if not self._validate_order_params(symbol, volume, order_type, side, price, sl, tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python stop order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Ajouter SL/TP si spécifiés
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Envoyer l'ordre
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de l'ordre: {result.comment}")
            return False, None
        
        logger.info(f"Ordre stop placé: {result.order}")
        return True, result.order
    
    def modify_order(
        self,
        order_id: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        Modifie un ordre existant.
        
        Args:
            order_id: ID de l'ordre à modifier
            sl: Nouveau stop loss
            tp: Nouveau take profit
            
        Returns:
            bool: True si la modification est réussie, False sinon
        """
        if not self.mt5_connector.connected:
            logger.error("Non connecté à MT5")
            return False
        
        # Récupérer l'ordre
        order = mt5.order_get(order_id)
        if order is None:
            logger.error(f"Ordre non trouvé: {order_id}")
            return False
        
        # Préparer la requête de modification
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": order_id,
            "symbol": order.symbol,
            "volume": order.volume_initial,
            "type": order.type,
            "position": order.position_id,
            "price": order.price_open,
            "sl": sl if sl is not None else order.sl,
            "tp": tp if tp is not None else order.tp,
        }
        
        # Envoyer la modification
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de la modification: {result.comment}")
            return False
        
        logger.info(f"Ordre modifié: {order_id}")
        return True
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Annule un ordre en attente.
        
        Args:
            order_id: ID de l'ordre à annuler
            
        Returns:
            bool: True si l'annulation est réussie, False sinon
        """
        if not self.mt5_connector.connected:
            logger.error("Non connecté à MT5")
            return False
        
        # Préparer la requête d'annulation
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order_id,
        }
        
        # Envoyer l'annulation
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Échec de l'annulation: {result.comment}")
            return False
        
        logger.info(f"Ordre annulé: {order_id}")
        return True
    
    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = ""
    ) -> Optional[OrderStatus]:
        """
        Place un ordre de trading.
        
        Args:
            symbol: Symbole à trader
            order_type: Type d'ordre ('MARKET', 'LIMIT', 'STOP')
            volume: Volume de l'ordre
            price: Prix pour les ordres limit/stop
            stop_loss: Stop loss
            take_profit: Take profit
            comment: Commentaire sur l'ordre
            
        Returns:
            Optional[OrderStatus]: État de l'ordre ou None en cas d'erreur
        """
        try:
            # Déterminer le côté de l'ordre en fonction du signal
            side = OrderSide.BUY if order_type == "BUY" else OrderSide.SELL
            
            # Convertir le type d'ordre en OrderType
            order_type = OrderType.from_string("MARKET", side.value)
            
            # Exécuter l'ordre
            success, order_id = self.execute_market_order(
                symbol=symbol,
                volume=volume,
                side=side,
                sl=stop_loss if stop_loss > 0 else None,
                tp=take_profit if take_profit > 0 else None
            )
            
            if not success:
                return None
            
            # Vérifier l'état de l'ordre
            return self.check_order_status(order_id)
            
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre: {str(e)}")
            return None
        
    def check_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """
        Vérifie l'état d'un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            OrderStatus: État mis à jour ou None si ordre non trouvé
        """
        if order_id not in self.pending_orders:
            return None
            
        order = self.pending_orders[order_id]
        
        # Récupérer l'historique des trades pour cet ordre
        trades = self.mt5_connector._safe_request(
            mt5.history_deals_get,
            from_date=order.timestamp
        )
        
        if not trades:
            return order
            
        # Calculer le volume exécuté
        filled_volume = sum(
            trade.volume
            for trade in trades
            if trade.order == order_id
        )
        
        # Mettre à jour le statut
        order.filled_volume = filled_volume
        order.remaining_volume = order.volume - filled_volume
        
        # Déterminer le statut
        if order.remaining_volume == 0:
            order.status = 'FILLED'
            del self.pending_orders[order_id]
            logger.info(f"Ordre {order_id} complètement exécuté")
        elif order.filled_volume > 0:
            order.status = 'PARTIALLY_FILLED'
            logger.warning(
                f"Ordre {order_id} partiellement exécuté "
                f"({order.filled_volume}/{order.volume})"
            )
        
        return order
        
    def check_all_pending_orders(self):
        """Vérifie tous les ordres en attente."""
        for order_id in list(self.pending_orders.keys()):
            self.check_order_status(order_id)
            
    def get_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """
        Récupère l'état d'un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            OrderStatus: État de l'ordre ou None si non trouvé
        """
        return self.pending_orders.get(order_id)

    def execute_order(self, order: Order) -> bool:
        """
        Exécute un ordre de trading.
        
        Args:
            order: Ordre à exécuter
            
        Returns:
            bool: True si l'ordre a été exécuté avec succès
        """
        try:
            # Vérifier si l'ordre peut être exécuté
            can_open, adjusted_volume = self.risk_manager.can_open_position(
                order.symbol,
                order.volume,
                order.strategy
            )
            
            if not can_open:
                return False
            
            # Utiliser le volume ajusté
            order.volume = adjusted_volume
            
            # Exécuter l'ordre
            if order.side == OrderSide.BUY:
                result = self.mt5_connector.open_buy_position(
                    order.symbol,
                    order.volume,
                    order.sl,
                    order.tp
                )
            else:
                result = self.mt5_connector.open_sell_position(
                    order.symbol,
                    order.volume,
                    order.sl,
                    order.tp
                )
            
            if result:
                logger.info(f"Ordre exécuté avec succès: {order}")
                return True
            else:
                logger.error(f"Échec de l'ordre: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre: {str(e)}")
            return False 