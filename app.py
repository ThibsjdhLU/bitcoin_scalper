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
import nest_asyncio
import MetaTrader5 as mt5
import queue

# Ajouter en haut de app.py
from config.unified_config import config  # Import manquant

from src.bitcoin_scalper.services import DashboardService

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

try:
    nest_asyncio.apply()
    logger.info("nest_asyncio appliqué avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'application de nest_asyncio: {e}")
    raise

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
            return self._is_running
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
            if self._is_running:
                logger.debug("RefreshManager déjà en cours d'exécution")
                return
                
            self._stop_event.clear()
            self._is_running = True
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()
            logger.info("RefreshManager démarré")

    def stop(self):
        with self._lock:
            if not self._is_running:
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
                if not self._data_loaded:
                    logger.info("Chargement initial des données...")
                    dashboard_service.update_data()  # Appel explicite
                    self._data_loaded = True
                else:
                    logger.debug("Rafraîchissement des données...")
                    dashboard_service.update_data()  # Appel explicite
                self._last_refresh = time.time()
                time.sleep(self._refresh_interval)  # Respecte l'intervalle
            except Exception as e:
                logger.error(f"Erreur: {str(e)}")
                time.sleep(5)
                
                
        logger.info("Boucle de rafraîchissement terminée")

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

# Style CSS global
def apply_css():
    """Applique le style CSS personnalisé."""
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .header-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .bot-status {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .refresh-info {
        font-size: 0.8rem;
        color: #AAAAAA;
    }
    .stButton>button {
        width: 100%;
    }
    .chart-container {
        background-color: #121212;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stats-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        height: 100%;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #AAAAAA;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    .positive-delta {
        color: #00FF88;
    }
    .negative-delta {
        color: #FF5588;
    }
    .trades-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .dataframe {
        width: 100%;
    }
    .dataframe th {
        background-color: #2D2D2D !important;
    }
    .profit-positive {
        color: #00FF88 !important;
        font-weight: bold;
    }
    .profit-negative {
        color: #FF5588 !important;
        font-weight: bold;
    }
    .type-buy {
        color: #00AAFF !important;
    }
    .type-sell {
        color: #FFAA00 !important;
    }
    .logs-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .log-console {
        background-color: #0A0A0A;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        height: 350px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        border: 1px solid #333;
    }
    .log-info {
        color: #00FF00;
        display: block;
        margin-bottom: 3px;
    }
    .log-error {
        color: #FF5588;
        font-weight: bold;
        display: block;
        margin-bottom: 3px;
    }
    .log-warning {
        color: #FFAA00;
        display: block;
        margin-bottom: 3px;
    }
    .highlight-buy {
        color: #00FF88;
        font-weight: bold;
        background-color: rgba(0, 255, 136, 0.1);
        padding: 0 3px;
        border-radius: 3px;
    }
    .highlight-sell {
        color: #FF5588;
        font-weight: bold;
        background-color: rgba(255, 85, 136, 0.1);
        padding: 0 3px;
        border-radius: 3px;
    }
    .highlight-hold {
        color: #FFAA00;
        font-weight: bold;
        background-color: rgba(255, 170, 0, 0.1);
        padding: 0 3px;
        border-radius: 3px;
    }
    .highlight-decision {
        background-color: rgba(0, 170, 255, 0.3);
        font-weight: bold;
        padding: 0 3px;
        border-radius: 3px;
    }
    .highlight-success {
        color: #00FF88;
        font-weight: bold;
    }
    .highlight-neutral {
        color: #FFAA00;
        font-weight: bold;
    }
    .highlight-strategy {
        color: #00AAFF;
        font-weight: bold;
    }
    .refresh-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .config-section {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .config-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #00AAFF;
    }
    .alert {
        background-color: #FF5588;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def header():
    """Affiche l'en-tête avec le statut du bot et les contrôles."""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("💹 Bitcoin Trading Bot")
        
        status_color = {
            "Actif": "🟢",
            "Inactif": "🔴",
            "Erreur": "⚠️"
        }.get(st.session_state.bot_status, "🔴")
        
        st.markdown(f'<div class="bot-status">**Statut:** {status_color} {st.session_state.bot_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="refresh-info">Dernier rafraîchissement: {st.session_state.last_refresh.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    with col2:
        if 'confirm_action' not in st.session_state or st.session_state.confirm_action is None:
            st.button("▶️ Démarrer", key="start", on_click=dashboard_service.handle_bot_action, args=("start",))
            st.button("⏹️ Arrêter", key="stop", on_click=dashboard_service.handle_bot_action, args=("stop",))
            st.button("🔄 Réinitialiser", key="reset", on_click=dashboard_service.handle_bot_action, args=("reset",))
        else:
            action_text = {
                "start": "démarrer", 
                "stop": "arrêter", 
                "reset": "réinitialiser"
            }.get(st.session_state.confirm_action, "effectuer l'action")
            
            st.warning(f"Confirmer pour {action_text} le bot?")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("✓ Confirmer", key="confirm", on_click=dashboard_service.handle_bot_action, args=(st.session_state.confirm_action,))
            with col_b:
                st.button("✗ Annuler", on_click=dashboard_service.handle_bot_action, args=("cancel",))
    
    st.markdown('</div>', unsafe_allow_html=True)

def refresh_controls():
    """Affiche les contrôles de rafraîchissement."""
    st.markdown('<div class="refresh-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        if st.button("🔄 Rafraîchir", key="refresh"):
            try:
                dashboard_service.update_data()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du rafraîchissement: {e}")
    
    with col2:
        refresh_interval = st.slider(
            "Intervalle de rafraîchissement (secondes)",
            min_value=1,
            max_value=60,
            value=10,
            key="refresh_interval_slider",
            on_change=lambda: setattr(refresh_manager, 'refresh_interval', st.session_state.refresh_interval_slider)
        )
    
    with col3:
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)

        # Démarrer ou arrêter le gestionnaire de rafraîchissement en fonction de l'état de la case
        if st.session_state.auto_refresh:
            if not refresh_manager.running:
                refresh_manager.start()
        else:
            if refresh_manager.running:
                refresh_manager.stop()
    
    with col4:
        if st.session_state.auto_refresh:
            now = datetime.now()
            if 'last_refresh' in st.session_state:
                time_since_refresh = (now - st.session_state.last_refresh).total_seconds()
                time_until_refresh = max(0, refresh_manager.refresh_interval - time_since_refresh)
                st.markdown(f"<div style='text-align: center;'>Prochain rafraîchissement dans: {int(time_until_refresh)}s</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def symbol_selector():
    """Affiche le sélecteur de symbole."""
    st.markdown('<div class="refresh-container">', unsafe_allow_html=True)
    try:
        available_symbols = dashboard_service.get_available_symbols()
    except Exception as e:
        st.error(f"Erreur de connexion MT5 : {str(e)}")
        available_symbols = ["BTCUSD"]  # Valeur par défaut
    st.session_state.available_symbols = available_symbols
    
    selected_symbol = st.selectbox(
        "Sélectionner une paire",
        available_symbols,
        index=available_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in available_symbols else 0,
        key="symbol_selector"
    )
    
    if selected_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = selected_symbol
        dashboard_service.update_data()
    
    st.markdown('</div>', unsafe_allow_html=True)

def price_chart():
    """Affiche le graphique des prix en temps réel."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Graphique des Prix")
    
    def price_chart():
        """Affiche le graphique des prix en temps réel."""
        with st.expander("Options du graphique"):
            # Correction syntaxique du clamping
            raw_signal = int(config.get("strategies.macd.signal_period", 9))
            macd_signal = max(min(raw_signal, 200), 1)  # Clamping correct
            
            raw_fast = int(config.get("strategies.macd.fast_period", 12))
            macd_fast = max(min(raw_fast, 200), 1)
            
            raw_slow = int(config.get("strategies.macd.slow_period", 26))
            macd_slow = max(min(raw_slow, 200), 1)

            # Ajustement automatique slow > fast
            if macd_slow <= macd_fast:
                macd_slow = macd_fast + 1
                config.set("strategies.macd.slow_period", macd_slow)
                logger.info(f"Ajustement automatique: MACD slow → {macd_slow}")

            # Interface utilisateur
            st.session_state.indicators['macd_signal'] = st.number_input(
                "MACD Signal",
                min_value=1,
                max_value=200,
                value=macd_signal,
                step=1
            )

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.indicators['show_sma'] = st.checkbox("SMA", 
                value=st.session_state.indicators.get('show_sma', True))
            
            if st.session_state.indicators['show_sma']:
                st.session_state.indicators['sma_period'] = st.number_input(
                    "Période SMA", 
                    min_value=1, 
                    max_value=200, 
                    value=st.session_state.indicators.get('sma_period', 20),
                    step=1
                )
                
            st.session_state.indicators['show_ema'] = st.checkbox("EMA", 
                value=st.session_state.indicators.get('show_ema', True))
            
            if st.session_state.indicators['show_ema']:
                st.session_state.indicators['ema_period'] = st.number_input(
                    "Période EMA", 
                    min_value=1, 
                    max_value=200, 
                    value=st.session_state.indicators.get('ema_period', 9),
                    step=1
                )
        
        with col2:
            st.session_state.indicators['show_bollinger'] = st.checkbox("Bandes de Bollinger", 
                value=st.session_state.indicators.get('show_bollinger', True))
            
            if st.session_state.indicators['show_bollinger']:
                st.session_state.indicators['bollinger_period'] = st.number_input(
                    "Période Bollinger", 
                    min_value=1, 
                    max_value=200, 
                    value=st.session_state.indicators.get('bollinger_period', 20),
                    step=1
                )
            
            st.session_state.indicators['show_rsi'] = st.checkbox("RSI", 
                value=st.session_state.indicators.get('show_rsi', True))
            
            if st.session_state.indicators['show_rsi']:
                st.session_state.indicators['rsi_period'] = st.number_input(
                    "Période RSI", 
                    min_value=1, 
                    max_value=200, 
                    value=st.session_state.indicators.get('rsi_period', 14),
                    step=1
                )
        
        with col3:
            st.session_state.indicators['show_macd'] = st.checkbox("MACD", 
                value=st.session_state.indicators.get('show_macd', True))
            
            if st.session_state.indicators['show_macd']:
                st.session_state.indicators['macd_fast'] = st.number_input(
                    "MACD Rapide", 
                    min_value=1, 
                    max_value=200, 
                    value=macd_fast,
                    step=1
                )
                
                st.session_state.indicators['macd_slow'] = st.number_input(
                    "MACD Lent", 
                    min_value=1, 
                    max_value=200, 
                    value=macd_slow,
                    step=1
                )
                
                st.session_state.indicators['macd_signal'] = st.number_input(
                    "MACD Signal", 
                    min_value=1, 
                    max_value=200, 
                    value=macd_signal,
                    step=1
                )
    
    fig = dashboard_service.create_price_chart(with_indicators=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
def statistics():
    """Affiche les statistiques en temps réel."""
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    st.subheader("📈 Statistiques")
    
    stats = dashboard_service.calculate_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_class = "positive-delta" if st.session_state.account_stats['profit'] >= 0 else "negative-delta"
        delta_prefix = "+" if st.session_state.account_stats['profit'] > 0 else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Solde</div>
            <div class="metric-value">${st.session_state.account_stats['balance']:.2f}</div>
            <div class="metric-delta {delta_class}">{delta_prefix}${st.session_state.account_stats['profit']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_rate = stats['win_rate']
        winning_trades = stats['winning_trades']
        total_trades = stats['total_trades']
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Taux de Réussite</div>
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-delta">{winning_trades}/{total_trades} trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_profit = stats['avg_profit']
        profit_class = "positive-delta" if avg_profit >= 0 else "negative-delta"
        profit_prefix = "+" if avg_profit > 0 else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Profit Moyen</div>
            <div class="metric-value">{profit_prefix}${avg_profit:.2f}</div>
            <div class="metric-delta">par trade</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_drawdown = stats['max_drawdown']
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Drawdown Max</div>
            <div class="metric-value">{max_drawdown:.2f}%</div>
            <div class="metric-delta negative-delta">de perte maximale</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def trades_history():
    """Affiche l'historique des trades."""
    st.markdown('<div class="trades-container">', unsafe_allow_html=True)
    st.subheader("🧾 Historique des Trades")
    
    trades_df = st.session_state.trades_history
    
    if trades_df is not None and not trades_df.empty:
        display_df = trades_df.copy()
        
        if 'time' in display_df.columns:
            renamed_columns = {
                'time': 'Date',
                'type': 'Type',
                'price_open': 'Prix d\'entrée',
                'price_close': 'Prix de sortie',
                'profit': 'PnL',
                'duration': 'Durée (h)'
            }
            
            columns_to_display = [col for col, new_name in renamed_columns.items() if col in display_df.columns]
            
            display_df = display_df[columns_to_display].copy()
            new_names = {col: renamed_columns[col] for col in columns_to_display}
            display_df.rename(columns=new_names, inplace=True)
            
            numeric_cols = ['Prix d\'entrée', 'Prix de sortie', 'PnL', 'Durée (h)']
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            if 'Date' in display_df.columns:
                display_df = display_df.sort_values('Date', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300,
                column_config={
                    "Date": st.column_config.DatetimeColumn(
                        "Date",
                        format="DD/MM/YYYY HH:mm:ss",
                        width="medium"
                    ),
                    "Prix d'entrée": st.column_config.NumberColumn(
                        "Prix d'entrée",
                        format="%.2f $",
                        width="small"
                    ),
                    "Prix de sortie": st.column_config.NumberColumn(
                        "Prix de sortie",
                        format="%.2f $",
                        width="small"
                    ),
                    "PnL": st.column_config.NumberColumn(
                        "PnL",
                        format="%.2f $",
                        width="small"
                    ),
                    "Durée (h)": st.column_config.NumberColumn(
                        "Durée (h)",
                        format="%.2f",
                        width="small"
                    )
                }
            )
            
            if 'PnL' in display_df.columns:
                total_profit = display_df['PnL'].sum()
                profit_class = "positive-delta" if total_profit >= 0 else "negative-delta"
                profit_prefix = "+" if total_profit > 0 else ""
                
                st.markdown(f"""
                <div style="text-align: right; margin-top: 10px;">
                    <span>Profit total: <span class="{profit_class}">{profit_prefix}${total_profit:.2f}</span></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Données incomplètes. Certaines colonnes sont manquantes.")
    else:
        st.info("Aucun trade disponible. Le bot n'a pas encore effectué de transactions.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def logs_console():
    """Affiche la console des logs."""
    st.markdown('<div class="logs-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("📝 Journal d'activité du Bot")
    with col2:
        filter_type = st.selectbox(
            "Filtrer par", 
            ["Tous", "INFO", "WARNING", "ERROR", "Signaux", "Stratégies"],
            key="log_filter"
        )
    with col3:
        st.button("🗑️ Effacer", key="clear_logs", on_click=lambda: st.session_state.__setitem__('log_messages', []))
    
    log_file = log_dir / "trading_bot.log"
    try:
        if not log_file.exists():
            log_file.touch()
            logger.info("Fichier de logs créé")
            
        encodings = ['utf-8', 'latin-1', 'cp1252']
        current_logs = []
        
        for encoding in encodings:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    current_logs = f.readlines()
                break
            except UnicodeDecodeError:
                continue
            
        if not current_logs:
            st.warning("Impossible de lire les logs avec les encodages supportés")
            return
            
        if 'log_messages' not in st.session_state or len(current_logs) > len(st.session_state.log_messages):
            st.session_state.log_messages = current_logs
            st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la lecture des logs: {e}")
        current_logs = []
    
    filtered_logs = st.session_state.get('log_messages', [])
    if filter_type != "Tous":
        if filter_type in ["INFO", "WARNING", "ERROR"]:
            filtered_logs = [log for log in filtered_logs if f"[{filter_type}]" in log]
        elif filter_type == "Signaux":
            filtered_logs = [log for log in filtered_logs if any(kw in log for kw in ["ACHAT", "VENTE", "CONSERVER", "Signal"])]
        elif filter_type == "Stratégies":
            filtered_logs = [log for log in filtered_logs if any(kw in log for kw in ["EMA", "RSI", "MACD", "Bollinger", "Combinaison"])]
    
    colored_logs = []
    for log in filtered_logs:
        if "[INFO]" in log:
            log_html = f'<span class="log-info">{log}</span>'
        elif "[ERROR]" in log:
            log_html = f'<span class="log-error">{log}</span>'
        elif "[WARNING]" in log:
            log_html = f'<span class="log-warning">{log}</span>'
        else:
            log_html = f'<span class="log-info">{log}</span>'
        
        for keyword, css_class in [
            ("ACHAT", "highlight-buy"),
            ("VENTE", "highlight-sell"),
            ("CONSERVER", "highlight-hold"),
            ("DÉCISION FINALE", "highlight-decision"),
            ("✅", "highlight-success"),
            ("⏸️", "highlight-neutral"),
            ("EMA Crossover", "highlight-strategy"),
            ("RSI", "highlight-strategy"),
            ("MACD", "highlight-strategy"),
            ("Bollinger", "highlight-strategy"),
            ("Combinaison", "highlight-strategy")
        ]:
            if keyword in log_html:
                log_html = log_html.replace(keyword, f'<span class="{css_class}">{keyword}</span>')
        
        colored_logs.append(log_html)
    
    log_text = "<br>".join(colored_logs)
    st.markdown(f'<div class="log-console" style="max-height: 400px; overflow-y: auto;">{log_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def config_panel():
    """Affiche le panneau de configuration dans la barre latérale."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-title">Paramètres de Trading</div>', unsafe_allow_html=True)
        
        demo_mode = st.checkbox(
            "Mode démo",
            value=dashboard_service.mt5_service.is_demo_mode(),
            help="Génère des données synthétiques si activé. Désactivez pour utiliser uniquement les données réelles."
        )
        if st.session_state.get('demo_mode', None) != demo_mode:
            dashboard_service.mt5_service.set_demo_mode(demo_mode)
            st.session_state.demo_mode = demo_mode
            if not demo_mode:
                st.info("Mode démo désactivé. Assurez-vous que vos identifiants sont corrects.")
                dashboard_service.update_data()
        
        initial_capital = st.number_input(
            "Capital initial (USDT)",
            min_value=100.0,
            max_value=100000.0,
            value=float(st.session_state.trading_params['initial_capital']),
            step=100.0
        )
        
        risk_per_trade = st.slider(
            "Risque par trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.trading_params['risk_per_trade']),
            step=0.1
        )
        
        strategies = ["EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Combinaison"]
        selected_strategies = st.multiselect(
            "Stratégies",
            strategies,
            default=[strategy for strategy in strategies if strategy in st.session_state.trading_params['strategy']]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Gestion des Risques"):
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state.trading_params['take_profit']),
                step=0.1
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state.trading_params['stop_loss']),
                step=0.1
            )
            
            trailing_stop = st.checkbox(
                "Utiliser Trailing Stop",
                value=st.session_state.trading_params['trailing_stop']
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Paramètres Avancés"):
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            
            time_frame = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=2
            )
            
            max_trades = st.number_input(
                "Nombre max de trades simultanés",
                min_value=1,
                max_value=10,
                value=3
            )
            
            leverage = st.slider(
                "Levier",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("💾 Enregistrer", type="primary"):
            params = {
                'initial_capital': initial_capital,
                'risk_per_trade': risk_per_trade,
                'strategy': selected_strategies,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'trailing_stop': trailing_stop,
                'time_frame': time_frame,
                'max_trades': max_trades,
                'leverage': leverage
            }
            
            dashboard_service.save_trading_params(params)
            st.success("✅ Configuration sauvegardée.")

def check_critical_alerts():
    """Vérifie s'il y a des alertes critiques à afficher."""
    max_drawdown = st.session_state.account_stats.get('max_drawdown', 0)
    if max_drawdown > 15:
        st.markdown(
            f'<div class="alert">⚠️ ALERTE: Drawdown élevé détecté ({max_drawdown:.2f}%)</div>',
            unsafe_allow_html=True
        )

ui_lock = threading.Lock()  # À déclarer au niveau global

def main():
    """Fonction principale."""
    try:
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
        
        try:
            # Arrêt des threads d'abord
            if refresh_manager.running:
                refresh_manager.stop()
                logger.info("Threads d'arrière-plan arrêtés")
            
            # Déconnexion MT5
            if dashboard_service.mt5_service.connected:
                dashboard_service.mt5_service.shutdown()
                logger.info("Déconnexion MT5 réussie")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage : {str(e)}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    nest_asyncio.apply()
    main()