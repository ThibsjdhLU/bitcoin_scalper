import logging
from typing import Optional, Dict, Any
from ..mt5_connector import MT5Connector

class BaseStrategy:
    def __init__(self, mt5_connector: MT5Connector):
        self.mt5 = mt5_connector
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.last_error: Optional[str] = None
        
    def check_mt5_connection(self) -> bool:
        """Vérifie la connexion MT5 et retourne True si tout est OK."""
        connection_status = self.mt5.check_connection()
        if not connection_status["connected"]:
            self.last_error = connection_status["message"]
            self.logger.error(f"Erreur de connexion MT5: {self.last_error}")
            return False
        return True
        
    def start(self) -> bool:
        """Démarre la stratégie de trading."""
        if not self.check_mt5_connection():
            return False
            
        try:
            self.is_running = True
            self.logger.info("Stratégie démarrée avec succès")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Erreur lors du démarrage de la stratégie: {self.last_error}")
            return False
            
    def stop(self) -> None:
        """Arrête la stratégie de trading."""
        self.is_running = False
        self.logger.info("Stratégie arrêtée")
        
    def execute_trade(self, symbol: str, volume: float, order_type: str) -> Dict[str, Any]:
        """Exécute un ordre de trading avec gestion des erreurs."""
        if not self.check_mt5_connection():
            return {"success": False, "error": self.last_error}
            
        try:
            # Logique d'exécution du trade à implémenter dans les classes dérivées
            pass
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Erreur lors de l'exécution du trade: {self.last_error}")
            return {"success": False, "error": self.last_error}
            
    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état actuel de la stratégie."""
        return {
            "is_running": self.is_running,
            "last_error": self.last_error,
            "mt5_connected": self.mt5.check_connection()["connected"]
        } 