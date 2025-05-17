"""
Application principale du bot de trading Bitcoin.
Interface utilisateur Streamlit pour le contrôle et le monitoring du bot.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
from pathlib import Path
import json
import os
import threading
import queue
import warnings
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import MetaTrader5 as mt5
import queue
import traceback

# Ajouter en haut de app.py
from config.unified_config import UnifiedConfig

from bot.services import DashboardService
from ui.style import apply_css
from ui.header import header, check_critical_alerts
from ui.controls import refresh_controls, symbol_selector
from ui.charts import price_chart
from ui.stats import statistics, trades_history
from ui.logs import logs_console
from ui.config_panel import config_panel
from bot.services.refresh_manager import RefreshManager

# Ignorer les avertissements de contexte manquant
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Création du dossier logs s'il n'existe pas
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Changé en DEBUG pour plus de détails
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "trading_bot.log")
    ]
)
logger = logging.getLogger(__name__)

logger.info("Démarrage de l'application...")

# Configuration de la page Streamlit
try:
    st.set_page_config(
        page_title="Bitcoin Trading Bot",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    logger.info("Configuration de la page Streamlit effectuée")
except Exception as e:
    logger.error(f"Erreur lors de la configuration de la page Streamlit: {e}")
    raise

# Ensure the login is an integer
os.environ['MT5_LOGIN'] = '101490774' # Replace with your actual login ID
logger.info("Variables d'environnement configurées")

class RefreshManager:
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._is_running = False
        self._last_refresh = None
        self._refresh_interval = 10
        self._data_loaded = False
        self.data_queue = queue.Queue()
        logger.info("RefreshManager initialisé")
    @property
    def running(self):
        """Retourne l'état d'exécution du thread"""
        with self._lock:
            return self._is_running and self._thread is not None and self._thread.is_alive()
    @property
    def refresh_interval(self):
        """Get the refresh interval (in seconds)."""
        return self._refresh_interval

    @refresh_interval.setter
    def refresh_interval(self, value: int):
        """Set the refresh interval (in seconds)."""
        if 1 <= value <= 60:
            self._refresh_interval = value
        else:
            logger.warning(f"Valeur invalide pour l'intervalle: {value}. Garde {self._refresh_interval}s")

    def start(self):
        with self._lock:
            if self.running:
                logger.debug("RefreshManager déjà en cours d'exécution")
                return
                
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()
            self._is_running = True
            logger.info("RefreshManager démarré")

    def stop(self):
        with self._lock:
            if not self.running:
                return
                
            logger.info("Arrêt du RefreshManager...")
            self._stop_event.set()
            
            if self._thread and self._thread.is_alive():
                try:
                    self._thread.join(timeout=5)
                except Exception as e:
                    logger.error("Erreur lors de l'arrêt du thread: %s", str(e))
                    
            self._is_running = False
            logger.info("RefreshManager arrêté")

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                ctx = get_script_run_ctx()
                if not ctx:
                    self._create_new_context()  # Nouvelle méthode critique
                
                # Déplacer le traitement lourd hors du main thread
                self.data_queue.put(dashboard_service.update_data)
                
                time.sleep(self._refresh_interval)
                
            except Exception as e:
                logger.error(f"Crash boucle: {traceback.format_exc()}")
                self.stop()

    def _create_new_context(self):
        # Implementation of _create_new_context method
        pass

    def _load_data(self):
        try:
            # Chargement initial des données
            pass
        except Exception as e:
            logger.error("Erreur lors du chargement initial: %s", str(e))

    def _refresh_data(self):
        try:
            # Rafraîchissement des données
            pass
        except Exception as e:
            logger.error("Erreur lors du rafraîchissement: %s", str(e))

# Initialisation des services
dashboard_service = DashboardService()

# Initialisation :
if 'refresh_manager' not in st.session_state:
    st.session_state.refresh_manager = RefreshManager()
refresh_manager = st.session_state.refresh_manager

def main():
    """Fonction principale."""
    try:
        # Explicit initialization
        if 'bot_status' not in st.session_state:
            st.session_state.bot_status = "Inactif"
        
        # Create context before threads
        ctx = get_script_run_ctx()
        if ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        
        if not st.session_state.get('indicators_initialized', False):
            dashboard_service._initialize_session_state()  # Force l'initialisation
            st.session_state.indicators_initialized = True
        
        if not st.session_state.get('initialized'):
            logger.info("Initialisation complète de l'application")
            dashboard_service._initialize_session_state()
            indicators = {
                'macd_signal': min(max(int(config.get("strategies.macd.signal_period", 9)), 1), 200),
                'macd_fast': min(max(int(config.get("strategies.macd.fast_period", 12)), 1), 200),
                'macd_slow': min(max(int(config.get("strategies.macd.slow_period", 26)), 1), 200),
            }
            # Variables de base
            st.session_state.initialized = True
            st.session_state.last_refresh = datetime.now()
            st.session_state.refresh_interval = 10
            st.session_state.bot_status = "Inactif"
            st.session_state.indicators = indicators
            
            # Configuration initiale
            apply_css()
            config.load_env()  # Chargement des variables d'environnement
            
            # Connexion MT5 une seule fois
            if not dashboard_service.mt5_service.connected:
                dashboard_service.mt5_service.connect()
            
            # Chargement initial des données
            try:
                dashboard_service.update_data()
                st.session_state.data_loaded = True
                logger.info("Données initiales chargées")
            except Exception as e:
                logger.error(f"Erreur critique: {str(e)}", exc_info=True)
                st.error("Une erreur technique est survenue. Veuillez réinitialiser l'application.")
        
        if 'indicators' not in st.session_state:
            st.session_state.indicators = {
                'show_rsi': True,  # or whatever default value you want
                # Initialize other indicators as needed
            }

        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []

        if 'task_queue' not in st.session_state:
            st.session_state.task_queue = queue.PriorityQueue(maxsize=100)

        # Interface utilisateur
        with st.container():
            # Section de débogage
            with st.expander("🔍 Débogage - État des Données", expanded=False):
                debug_info = {
                    "Statut MT5": "Connecté" if mt5.terminal_info() else "Déconnecté",
                    "Dernière mise à jour": st.session_state.last_refresh,
                    "Auto-refresh": "Actif" if refresh_manager.running else "Inactif"
                }
                st.json(debug_info)

            # Disposition principale
            config_panel()
            header()
            refresh_controls()
            symbol_selector()
            check_critical_alerts()

            # Affichage des données
            col1, col2 = st.columns([3, 1])
            with col1:
                price_chart()
            with col2:
                statistics()
                trades_history()

            logs_console()

    except RuntimeError as e:
        logger.error(f"Erreur de boucle d'événements: {str(e)}")
        st.error("Une erreur est survenue lors de l'exécution de l'application.")
    finally:
        logger.info("Nettoyage des ressources...")
        refresh_manager.stop()
        
        # Enhanced thread state check
        if (hasattr(refresh_manager, '_thread') and
            refresh_manager._thread is not None and
            refresh_manager._thread.is_alive()):
            try:
                refresh_manager._thread.join(timeout=1)
                logger.info("Thread RefreshManager arrêté")
            except RuntimeError as e:
                logger.error(f"Erreur d'arrêt du thread : {str(e)}")
        
        # MT5 disconnection
        try:
            if (hasattr(dashboard_service, 'mt5_service') and
                dashboard_service.mt5_service.connected):
                dashboard_service.mt5_service.shutdown()
                logger.info("Déconnexion MT5 confirmée")
        except Exception as e:
            logger.error(f"Erreur déconnexion MT5 : {str(e)}")
        
        time.sleep(0.2)

if __name__ == "__main__":
    # Ensure session state is initialized
    dashboard_service = DashboardService()  # This will call _initialize_session_state
    main()