"""
Gestionnaire de rafraîchissement des données pour NiceGUI.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from nicegui import ui

logger = logging.getLogger(__name__)

class RefreshManager:
    def __init__(self, app_state):
        self.app_state = app_state
        self._refresh_interval = 10
        self._is_running = False
        self._last_refresh = None
        self._data_loaded = False
        self._update_callbacks: Dict[str, Callable] = {}
        
    @property
    def running(self) -> bool:
        """Retourne l'état d'exécution"""
        return self._is_running
    
    @property
    def refresh_interval(self) -> int:
        """Retourne l'intervalle de rafraîchissement"""
        return self._refresh_interval
    
    @refresh_interval.setter
    def refresh_interval(self, value: int):
        """Définit l'intervalle de rafraîchissement"""
        if 1 <= value <= 60:
            self._refresh_interval = value
        else:
            logger.warning(f"Valeur invalide pour l'intervalle: {value}. Garde {self._refresh_interval}s")
    
    def register_update_callback(self, name: str, callback: Callable):
        """Enregistre un callback pour les mises à jour"""
        self._update_callbacks[name] = callback
    
    async def start(self):
        """Démarre le rafraîchissement automatique"""
        if self._is_running:
            logger.debug("RefreshManager déjà en cours d'exécution")
            return
        
        self._is_running = True
        self.app_state.set('bot_status', "Actif")
        
        while self._is_running:
            try:
                await self._refresh_data()
                await asyncio.sleep(self._refresh_interval)
            except Exception as e:
                logger.error(f"Erreur lors du rafraîchissement: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Arrête le rafraîchissement automatique"""
        self._is_running = False
        self.app_state.set('bot_status', "Inactif")
        logger.info("RefreshManager arrêté")
    
    async def _refresh_data(self):
        """Rafraîchit les données"""
        try:
            # Exécuter tous les callbacks de mise à jour
            for name, callback in self._update_callbacks.items():
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"Erreur dans le callback {name}: {e}")
            
            self._last_refresh = datetime.now()
            self.app_state.set('last_refresh', self._last_refresh)
            
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement des données: {e}")
            raise 