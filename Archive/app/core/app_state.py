"""
Gestion de l'état de l'application et des tâches en arrière-plan.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from nicegui import ui
import logging

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self._state: Dict[str, Any] = {
            'bot_status': "Inactif",
            'last_refresh': None,
            'refresh_interval': 10,
            'indicators': {
                'macd_signal': 9,
                'macd_fast': 12,
                'macd_slow': 26,
            },
            'data_loaded': False,
            'current_strategy': 'MACD',
            'pnl': 0.0,
            'positions': 0,
            'trades': 0
        }
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._callbacks: Dict[str, list] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de l'état"""
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Définit une valeur dans l'état et notifie les callbacks"""
        self._state[key] = value
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                callback(value)
    
    def register_callback(self, key: str, callback: callable) -> None:
        """Enregistre un callback pour une clé donnée"""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    async def start_background_task(self, name: str, coro) -> None:
        """Démarre une tâche en arrière-plan"""
        if name in self._background_tasks:
            logger.warning(f"La tâche {name} est déjà en cours d'exécution")
            return
        
        task = asyncio.create_task(coro)
        self._background_tasks[name] = task
        
        try:
            await task
        except Exception as e:
            logger.error(f"Erreur dans la tâche {name}: {e}")
        finally:
            if name in self._background_tasks:
                del self._background_tasks[name]
    
    def stop_background_task(self, name: str) -> None:
        """Arrête une tâche en arrière-plan"""
        if name in self._background_tasks:
            self._background_tasks[name].cancel()
            del self._background_tasks[name]
    
    def stop_all_tasks(self) -> None:
        """Arrête toutes les tâches en arrière-plan"""
        for name in list(self._background_tasks.keys()):
            self.stop_background_task(name)

# Instance globale de l'état de l'application
app_state = AppState() 