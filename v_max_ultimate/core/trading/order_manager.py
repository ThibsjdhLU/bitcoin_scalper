import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from ..risk.risk_manager import RiskManager

class OrderManager:
    """
    Gestionnaire d'ordres de trading
    """
    def __init__(self, config: Dict, risk_manager: RiskManager):
        self.config = config
        self.risk_manager = risk_manager
        this.logger = logging.getLogger(__name__)
        
    def place_market_order(self,
                          symbol: str,
                          order_type: str,
                          volume: float,
                          sl: float = 0.0,
                          tp: float = 0.0,
                          comment: str = "") -> Tuple[bool, str]:
        """
        Place un ordre au marché
        
        Args:
            symbol: Symbole de trading
            order_type: Type d'ordre (BUY/SELL)
            volume: Volume en lots
            sl: Stop loss en points
            tp: Take profit en points
            comment: Commentaire sur l'ordre
            
        Returns:
            Tuple[bool, str]: (Succès, Message)
        """
        try:
            if not mt5.initialize():
                return False, "Échec de l'initialisation MT5"
                
            # Vérifier les limites de risque
            if not this.risk_manager.check_order_limits(symbol, volume):
                return False, "Limites de risque dépassées"
                
            # Préparer la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).ask if order_type == "BUY" else mt5.symbol_info_tick(symbol).bid,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Envoyer l'ordre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Échec de l'ordre: {result.comment}"
                
            return True, f"Ordre placé avec succès: {result.order}"
            
        except Exception as e:
            this.logger.error(f"Erreur lors du placement de l'ordre: {str(e)}")
            return False, str(e)
            
    def modify_order(self,
                    order_id: int,
                    sl: Optional[float] = None,
                    tp: Optional[float] = None) -> Tuple[bool, str]:
        """
        Modifie un ordre existant
        
        Args:
            order_id: ID de l'ordre
            sl: Nouveau stop loss
            tp: Nouveau take profit
            
        Returns:
            Tuple[bool, str]: (Succès, Message)
        """
        try:
            if not mt5.initialize():
                return False, "Échec de l'initialisation MT5"
                
            # Récupérer l'ordre
            order = mt5.orders_get(ticket=order_id)
            if not order:
                return False, "Ordre non trouvé"
                
            # Préparer la requête
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": order_id,
                "price": order[0].price_open,
                "sl": sl if sl is not None else order[0].sl,
                "tp": tp if tp is not None else order[0].tp,
            }
            
            # Modifier l'ordre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Échec de la modification: {result.comment}"
                
            return True, "Ordre modifié avec succès"
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la modification de l'ordre: {str(e)}")
            return False, str(e)
            
    def close_order(self,
                   order_id: int,
                   volume: Optional[float] = None) -> Tuple[bool, str]:
        """
        Ferme un ordre
        
        Args:
            order_id: ID de l'ordre
            volume: Volume à fermer (optionnel)
            
        Returns:
            Tuple[bool, str]: (Succès, Message)
        """
        try:
            if not mt5.initialize():
                return False, "Échec de l'initialisation MT5"
                
            # Récupérer l'ordre
            order = mt5.orders_get(ticket=order_id)
            if not order:
                return False, "Ordre non trouvé"
                
            # Volume à fermer
            close_volume = volume if volume is not None else order[0].volume_initial
            
            # Préparer la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "order": order_id,
                "symbol": order[0].symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if order[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": order_id,
                "price": mt5.symbol_info_tick(order[0].symbol).bid if order[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(order[0].symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Fermeture",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Fermer l'ordre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Échec de la fermeture: {result.comment}"
                
            return True, "Ordre fermé avec succès"
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la fermeture de l'ordre: {str(e)}")
            return False, str(e)
            
    def get_open_orders(self) -> List[Dict]:
        """
        Récupère tous les ordres ouverts
        
        Returns:
            List[Dict]: Liste des ordres ouverts
        """
        try:
            if not mt5.initialize():
                return []
                
            orders = mt5.orders_get()
            if not orders:
                return []
                
            return [{
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': 'BUY' if order.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': order.volume_initial,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'profit': order.profit,
                'comment': order.comment
            } for order in orders]
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération des ordres: {str(e)}")
            return []
            
    def get_order_history(self,
                         start_date: datetime,
                         end_date: datetime) -> List[Dict]:
        """
        Récupère l'historique des ordres
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            List[Dict]: Liste des ordres historiques
        """
        try:
            if not mt5.initialize():
                return []
                
            history = mt5.history_deals_get(start_date, end_date)
            if not history:
                return []
                
            return [{
                'ticket': deal.ticket,
                'symbol': deal.symbol,
                'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                'volume': deal.volume,
                'price': deal.price,
                'profit': deal.profit,
                'time': deal.time,
                'comment': deal.comment
            } for deal in history]
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
            return [] 