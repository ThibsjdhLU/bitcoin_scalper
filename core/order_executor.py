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

class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

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
    Gère l'exécution des ordres de trading.
    
    Attributes:
        connector (MT5Connector): Instance du connecteur MT5
        config_path (str): Chemin vers le fichier de configuration
    """
    
    def __init__(self, connector: MT5Connector, config_path: str = "config/config.json"):
        """
        Initialise l'exécuteur d'ordres.
        
        Args:
            connector: Instance du connecteur MT5
            config_path: Chemin vers le fichier de configuration
        """
        self.connector = connector
        self.config_path = config_path
        self._load_config()
        self.pending_orders: Dict[int, OrderStatus] = {}
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, 'r') as f:
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
        if not self.connector.connected:
            logger.error("Non connecté à MT5")
            return False
        
        # Vérifier le symbole
        symbol_info = self.connector.get_symbol_info(symbol)
        if symbol_info is None:
            return False
        
        # Vérifier le volume
        if not (symbol_info['volume_min'] <= volume <= symbol_info['volume_max']):
            logger.error(f"Volume invalide: {volume}")
            return False
        
        # Vérifier le prix pour les ordres limit/stop
        if order_type in [OrderType.LIMIT, OrderType.STOP] and price is None:
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
        if not self._validate_order_params(symbol, volume, OrderType.MARKET, side, sl=sl, tp=tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL,
            "deviation": 20,
            "magic": 234000,
            "comment": "python market order",
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
        if not self._validate_order_params(symbol, volume, OrderType.LIMIT, side, price, sl, tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY_LIMIT if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT,
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
        if not self._validate_order_params(symbol, volume, OrderType.STOP, side, price, sl, tp):
            return False, None
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY_STOP if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_STOP,
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
        if not self.connector.connected:
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
        if not self.connector.connected:
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
        Place un ordre avec gestion des exécutions partielles.
        
        Args:
            symbol: Symbole
            order_type: Type d'ordre ('BUY' ou 'SELL')
            volume: Volume
            price: Prix
            stop_loss: Stop loss (optionnel)
            take_profit: Take profit (optionnel)
            comment: Commentaire (optionnel)
            
        Returns:
            OrderStatus: État de l'ordre ou None si erreur
        """
        # Convertir le type d'ordre
        mt5_order_type = (
            mt5.ORDER_TYPE_BUY if order_type == 'BUY'
            else mt5.ORDER_TYPE_SELL
        )
        
        # Placer l'ordre
        result = self.connector.place_order(
            symbol=symbol,
            order_type=mt5_order_type,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )
        
        if not result or 'order' not in result:
            logger.error(f"Échec de placement d'ordre: {result}")
            return None
            
        # Créer le statut de l'ordre
        order = result['order']
        status = OrderStatus(
            order_id=order.ticket,
            symbol=symbol,
            type=order_type,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            filled_volume=0.0,
            remaining_volume=volume,
            status='PENDING',
            comment=comment,
            timestamp=datetime.now()
        )
        
        # Enregistrer l'ordre en attente
        self.pending_orders[order.ticket] = status
        
        # Vérifier immédiatement l'état
        self.check_order_status(order.ticket)
        
        return status
        
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
        trades = self.connector._safe_request(
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