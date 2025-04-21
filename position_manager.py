import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = 'long'
    SHORT = 'short'

class PositionStatus(Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    PENDING = 'pending'

@dataclass
class Position:
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    leverage: float
    stop_loss: float
    take_profit: float
    status: PositionStatus
    entry_time: datetime
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    pnl: Optional[float] = None
    fees: float = 0.0

class PositionManager:
    """
    Gestionnaire des positions de trading
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de positions
        
        Args:
            config (dict): Configuration du gestionnaire
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Configuration
        self.max_positions = config.get('max_positions', 3)
        self.default_leverage = config.get('default_leverage', 1)
        self.position_timeout = config.get('position_timeout', 3600)  # 1 heure
        
        logger.info("Gestionnaire de positions initialisé")
    
    def can_open_position(self, symbol: str) -> bool:
        """
        Vérifie si une nouvelle position peut être ouverte
        
        Args:
            symbol (str): Symbole de trading
            
        Returns:
            bool: True si une position peut être ouverte
        """
        # Vérification du nombre de positions ouvertes
        open_positions = len([p for p in self.positions.values() 
                            if p.status == PositionStatus.OPEN])
        if open_positions >= self.max_positions:
            logger.warning("Nombre maximum de positions atteint")
            return False
        
        # Vérification des positions existantes sur le symbole
        symbol_positions = len([p for p in self.positions.values() 
                              if p.symbol == symbol and p.status == PositionStatus.OPEN])
        if symbol_positions > 0:
            logger.warning(f"Position déjà ouverte sur {symbol}")
            return False
        
        return True
    
    def open_position(self, symbol: str, side: PositionSide, entry_price: float, 
                     size: float, leverage: int = 1, stop_loss: float = None, 
                     take_profit: float = None) -> Optional[Position]:
        """
        Ouvre une nouvelle position
        
        Args:
            symbol (str): Symbole de trading
            side (PositionSide): Côté de la position (LONG/SHORT)
            entry_price (float): Prix d'entrée
            size (float): Taille de la position
            leverage (int): Levier utilisé
            stop_loss (float, optional): Niveau de stop loss
            take_profit (float, optional): Niveau de take profit
            
        Returns:
            Optional[Position]: Position ouverte ou None si échec
        """
        try:
            # Vérification des paramètres
            if not symbol or entry_price <= 0 or size <= 0:
                logger.error("Paramètres invalides pour l'ouverture de position")
                return None
                
            # Vérification du nombre de positions
            if not self.can_open_position(symbol):
                logger.warning(f"Impossible d'ouvrir une nouvelle position pour {symbol}")
                return None
                
            # Création de l'ID unique
            position_id = f"{symbol}_{side.value}_{int(time.time())}"
            
            # Création de la position
            position = Position(
                id=position_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=PositionStatus.OPEN,
                entry_time=datetime.now()
            )
            
            # Ajout à la liste des positions ouvertes
            self.positions[position_id] = position
            
            logger.info(f"Position ouverte: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture de la position: {e}")
            return None
            
    def close_position(self, position_id: str, exit_price: float) -> Optional[Position]:
        """
        Ferme une position existante
        
        Args:
            position_id (str): ID de la position
            exit_price (float): Prix de sortie
            
        Returns:
            Optional[Position]: Position fermée ou None si échec
        """
        try:
            # Vérification de l'existence de la position
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} non trouvée")
                return None
                
            position = self.positions[position_id]
            
            # Calcul du PnL
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
                
            # Mise à jour de la position
            position.close_price = exit_price
            position.close_time = datetime.now()
            position.status = PositionStatus.CLOSED
            position.pnl = pnl
            
            # Déplacement vers les positions fermées
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            logger.info(f"Position fermée: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la position: {e}")
            return None
    
    def update_position(self, position_id: str, current_price: float) -> None:
        """
        Met à jour une position avec le prix actuel
        
        Args:
            position_id (str): ID de la position
            current_price (float): Prix actuel
        """
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            return
        
        # Vérification du stop loss
        if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
           (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
            self.close_position(position_id, position.stop_loss)
            logger.info(f"Stop loss déclenché pour {position_id}")
            return
        
        # Vérification du take profit
        if (position.side == PositionSide.LONG and current_price >= position.take_profit) or \
           (position.side == PositionSide.SHORT and current_price <= position.take_profit):
            self.close_position(position_id, position.take_profit)
            logger.info(f"Take profit déclenché pour {position_id}")
            return
        
        # Vérification du timeout
        if (datetime.now() - position.entry_time).total_seconds() > self.position_timeout:
            self.close_position(position_id, position.entry_price)
            logger.info(f"Timeout déclenché pour {position_id}")
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Récupère une position par son ID
        
        Args:
            position_id (str): ID de la position
            
        Returns:
            Optional[Position]: Position trouvée ou None
        """
        return self.positions.get(position_id)
    
    def get_open_positions(self) -> List[Position]:
        """
        Récupère toutes les positions ouvertes
        
        Returns:
            List[Position]: Liste des positions ouvertes
        """
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_closed_positions(self) -> List[Position]:
        """
        Récupère l'historique des positions fermées
        
        Returns:
            List[Position]: Liste des positions fermées
        """
        return self.closed_positions
    
    def get_position_metrics(self) -> Dict:
        """
        Calcule les métriques des positions
        
        Returns:
            Dict: Métriques des positions
        """
        open_pnl = sum(p.pnl or 0 for p in self.positions.values())
        closed_pnl = sum(p.pnl or 0 for p in self.closed_positions)
        total_fees = sum(p.fees for p in self.closed_positions)
        
        return {
            'open_positions': len(self.positions),
            'total_positions': len(self.closed_positions),
            'open_pnl': open_pnl,
            'closed_pnl': closed_pnl,
            'total_pnl': open_pnl + closed_pnl,
            'total_fees': total_fees
        } 