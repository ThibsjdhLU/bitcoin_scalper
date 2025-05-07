from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PartialFillHandler:
    """
    Gère les ordres partiellement remplis et leurs ajustements.
    """
    
    def __init__(self, connector, position_manager):
        self.connector = connector
        self.position_manager = position_manager
        self.partial_orders: Dict[int, Dict] = {}  # ticket -> order info
        self.max_wait_time = 300  # secondes
        self.min_fill_ratio = 0.5  # 50% minimum
        
    def handle_partial_fill(
        self,
        ticket: int,
        requested_volume: float,
        filled_volume: float,
        remaining_volume: float,
        symbol: str,
        side: str,
        price: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """
        Gère un ordre partiellement rempli.
        
        Args:
            ticket: Ticket de l'ordre
            requested_volume: Volume demandé
            filled_volume: Volume rempli
            remaining_volume: Volume restant
            symbol: Symbole
            side: Sens de l'ordre
            price: Prix
            stop_loss: Stop loss
            take_profit: Take profit
            
        Returns:
            bool: True si gestion réussie
        """
        try:
            fill_ratio = filled_volume / requested_volume
            
            # Si le remplissage est suffisant, on garde l'ordre tel quel
            if fill_ratio >= self.min_fill_ratio:
                logger.info(f"Ordre {ticket} partiellement rempli ({fill_ratio:.2%}) - suffisant")
                return True
                
            # Enregistrer l'ordre partiel
            self.partial_orders[ticket] = {
                'symbol': symbol,
                'side': side,
                'requested_volume': requested_volume,
                'filled_volume': filled_volume,
                'remaining_volume': remaining_volume,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now()
            }
            
            # Annuler l'ordre partiel
            if not self.connector.cancel_order(ticket):
                logger.error(f"Échec d'annulation de l'ordre partiel {ticket}")
                return False
                
            # Placer un nouvel ordre avec le volume restant
            new_ticket = self.position_manager.open_position(
                symbol=symbol,
                volume=remaining_volume,
                side=side,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="PARTIAL_FILL",
                params={'original_ticket': ticket}
            )
            
            if new_ticket:
                logger.info(f"Nouvel ordre placé pour le volume restant: {new_ticket}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la gestion de l'ordre partiel: {str(e)}")
            return False
            
    def check_partial_orders(self) -> None:
        """Vérifie et nettoie les ordres partiels anciens."""
        current_time = datetime.now()
        to_remove = []
        
        for ticket, order in self.partial_orders.items():
            if (current_time - order['timestamp']) > timedelta(seconds=self.max_wait_time):
                logger.warning(f"Ordre partiel {ticket} trop ancien - nettoyage")
                to_remove.append(ticket)
                
        for ticket in to_remove:
            del self.partial_orders[ticket]
            
    def get_partial_orders(self) -> List[Dict]:
        """Retourne la liste des ordres partiels actifs."""
        return list(self.partial_orders.values())
        
    def clear_partial_order(self, ticket: int) -> bool:
        """
        Nettoie un ordre partiel.
        
        Args:
            ticket: Ticket de l'ordre
            
        Returns:
            bool: True si nettoyage réussi
        """
        if ticket in self.partial_orders:
            del self.partial_orders[ticket]
            return True
        return False 