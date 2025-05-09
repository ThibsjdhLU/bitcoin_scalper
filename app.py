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

from src.bitcoin_scalper.services import DashboardService

# Création du dossier logs s'il n'existe pas
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "trading_bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Bitcoin Trading Bot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RefreshManager:
    def __init__(self, dashboard_service):
        self.dashboard_service = dashboard_service
        self.refresh_queue = queue.Queue()
        self.last_refresh = datetime.now()
        self.refresh_interval = 10
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._refresh_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"Rafraîchissement automatique démarré (intervalle: {self.refresh_interval}s)")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            logger.info("Rafraîchissement automatique arrêté")

    def _refresh_loop(self):
        while self.running:
            try:
                # Mettre à jour les données
                self.dashboard_service.update_data()
                self.last_refresh = datetime.now()
                
                # Forcer le rafraîchissement de l'interface
                st.rerun()
                
                # Attendre l'intervalle spécifié
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Erreur dans la boucle de rafraîchissement: {e}")
                time.sleep(1)

    def get_latest_data(self):
        try:
            latest_data = None
            while not self.refresh_queue.empty():
                latest_data = self.refresh_queue.get_nowait()
            return latest_data
        except queue.Empty:
            return None

# Initialisation du service dashboard et du gestionnaire de rafraîchissement
dashboard_service = DashboardService()
refresh_manager = RefreshManager(dashboard_service)

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
        # Afficher les contrôles en fonction de l'état de confirmation
        if st.session_state.confirm_action is None or st.session_state.confirm_action is False:
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
                st.button("✗ Annuler", key="cancel", on_click=dashboard_service.handle_bot_action, args=("cancel",))
    
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
        auto_refresh = st.checkbox("Rafraîchissement auto", value=True, key="auto_refresh")
        if auto_refresh and not refresh_manager.running:
            refresh_manager.start()
        elif not auto_refresh and refresh_manager.running:
            refresh_manager.stop()
    
    with col4:
        if auto_refresh:
            now = datetime.now()
            if 'last_refresh' in st.session_state:
                time_since_refresh = (now - st.session_state.last_refresh).total_seconds()
                time_until_refresh = max(0, refresh_manager.refresh_interval - time_since_refresh)
                st.markdown(f"<div style='text-align: center;'>Prochain rafraîchissement dans: {int(time_until_refresh)}s</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Dans la fonction refresh_controls
    if auto_refresh:
        latest_data = refresh_manager.get_latest_data()
        if latest_data:
            try:
                # Mettre à jour les données dans le dashboard
                dashboard_service.update_data()  # Ne pas passer d'arguments ici
                st.session_state.last_refresh = latest_data['timestamp']
                st.rerun()
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des données: {e}")

def symbol_selector():
    """Affiche le sélecteur de symbole."""
    st.markdown('<div class="refresh-container">', unsafe_allow_html=True)
    
    # Récupérer les symboles disponibles
    available_symbols = dashboard_service.get_available_symbols()
    st.session_state.available_symbols = available_symbols
    
    # Sélecteur de symbole
    selected_symbol = st.selectbox(
        "Sélectionner une paire",
        available_symbols,
        index=available_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in available_symbols else 0,
        key="symbol_selector"
    )
    
    # Mettre à jour si changé
    if selected_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = selected_symbol
        dashboard_service.update_data()
    
    st.markdown('</div>', unsafe_allow_html=True)

def price_chart():
    """Affiche le graphique des prix en temps réel."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Graphique des Prix")
    
    # Interface pour les indicateurs
    with st.expander("Options du graphique"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.indicators['show_sma'] = st.checkbox("SMA", value=st.session_state.indicators['show_sma'])
            if st.session_state.indicators['show_sma']:
                st.session_state.indicators['sma_period'] = st.number_input("Période SMA", min_value=1, max_value=200, value=st.session_state.indicators['sma_period'])
                
            st.session_state.indicators['show_ema'] = st.checkbox("EMA", value=st.session_state.indicators['show_ema'])
            if st.session_state.indicators['show_ema']:
                st.session_state.indicators['ema_period'] = st.number_input("Période EMA", min_value=1, max_value=200, value=st.session_state.indicators['ema_period'])
        
        with col2:
            st.session_state.indicators['show_bollinger'] = st.checkbox("Bandes de Bollinger", value=st.session_state.indicators['show_bollinger'])
            if st.session_state.indicators['show_bollinger']:
                st.session_state.indicators['bollinger_period'] = st.number_input("Période Bollinger", min_value=1, max_value=200, value=st.session_state.indicators['bollinger_period'])
            
            st.session_state.indicators['show_rsi'] = st.checkbox("RSI", value=st.session_state.indicators['show_rsi'])
            if st.session_state.indicators['show_rsi']:
                st.session_state.indicators['rsi_period'] = st.number_input("Période RSI", min_value=1, max_value=200, value=st.session_state.indicators['rsi_period'])
        
        with col3:
            st.session_state.indicators['show_macd'] = st.checkbox("MACD", value=st.session_state.indicators['show_macd'])
            if st.session_state.indicators['show_macd']:
                st.session_state.indicators['macd_fast'] = st.number_input("MACD Rapide", min_value=1, max_value=200, value=st.session_state.indicators['macd_fast'])
                st.session_state.indicators['macd_slow'] = st.number_input("MACD Lent", min_value=1, max_value=200, value=st.session_state.indicators['macd_slow'])
                st.session_state.indicators['macd_signal'] = st.number_input("MACD Signal", min_value=1, max_value=200, value=st.session_state.indicators['macd_signal'])
    
    # Créer et afficher le graphique
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
        # Solde actuel
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
        # Taux de réussite
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
        # Profit moyen par trade
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
        # Drawdown maximum
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
        # Filtrer et formater les colonnes pour l'affichage
        display_df = trades_df.copy()
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        if 'time' in display_df.columns:
            # Renommer et réorganiser les colonnes
            renamed_columns = {
                'time': 'Date',
                'type': 'Type',
                'price_open': 'Prix d\'entrée',
                'price_close': 'Prix de sortie',
                'profit': 'PnL',
                'duration': 'Durée (h)'
            }
            
            # Sélectionner seulement les colonnes qui existent
            columns_to_display = [col for col, new_name in renamed_columns.items() if col in display_df.columns]
            
            display_df = display_df[columns_to_display].copy()
            
            # Renommer les colonnes existantes
            new_names = {col: renamed_columns[col] for col in columns_to_display}
            display_df.rename(columns=new_names, inplace=True)
            
            # Arrondir les valeurs numériques
            numeric_cols = ['Prix d\'entrée', 'Prix de sortie', 'PnL', 'Durée (h)']
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            # Trier par date décroissante
            if 'Date' in display_df.columns:
                display_df = display_df.sort_values('Date', ascending=False)
            
            # Afficher le tableau avec des configurations de colonnes
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
            
            # Résumé des trades
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
    
    # Titre et contrôles en ligne
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
    
    # Lire les logs depuis le fichier
    log_file = log_dir / "trading_bot.log"
    try:
        if not log_file.exists():
            log_file.touch()
            logger.info("Fichier de logs créé")
            
        # Essayer différents encodages
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
            
        # Mettre à jour les logs dans la session uniquement si de nouveaux logs sont disponibles
        if 'log_messages' not in st.session_state or len(current_logs) > len(st.session_state.log_messages):
            st.session_state.log_messages = current_logs
            # Forcer le rafraîchissement uniquement si de nouveaux logs sont disponibles
            st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la lecture des logs: {e}")
        current_logs = []
    
    # Filtrer les logs selon la sélection
    filtered_logs = st.session_state.get('log_messages', [])
    if filter_type != "Tous":
        if filter_type in ["INFO", "WARNING", "ERROR"]:
            filtered_logs = [log for log in filtered_logs if f"[{filter_type}]" in log]
        elif filter_type == "Signaux":
            filtered_logs = [log for log in filtered_logs if any(kw in log for kw in ["ACHAT", "VENTE", "CONSERVER", "Signal"])]
        elif filter_type == "Stratégies":
            filtered_logs = [log for log in filtered_logs if any(kw in log for kw in ["EMA", "RSI", "MACD", "Bollinger", "Combinaison"])]
    
    # Traiter les logs pour ajouter des couleurs et mise en forme
    colored_logs = []
    for log in filtered_logs:
        # Coloration selon le niveau de log
        if "[INFO]" in log:
            log_html = f'<span class="log-info">{log}</span>'
        elif "[ERROR]" in log:
            log_html = f'<span class="log-error">{log}</span>'
        elif "[WARNING]" in log:
            log_html = f'<span class="log-warning">{log}</span>'
        else:
            log_html = f'<span class="log-info">{log}</span>'
        
        # Mise en évidence des mots-clés importants
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
    
    # Afficher les logs avec une hauteur plus grande et un conteneur scrollable
    st.markdown(f'<div class="log-console" style="max-height: 400px; overflow-y: auto;">{log_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def config_panel():
    """Affiche le panneau de configuration dans la barre latérale."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Section Trading
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-title">Paramètres de Trading</div>', unsafe_allow_html=True)
        
        # Mode démo (ajout)
        demo_mode = st.checkbox(
            "Mode démo",
            value=dashboard_service.mt5_service.is_demo_mode(),
            help="Génère des données synthétiques si activé. Désactivez pour utiliser uniquement les données réelles."
        )
        if st.session_state.get('demo_mode', None) != demo_mode:
            dashboard_service.mt5_service.set_demo_mode(demo_mode)
            st.session_state.demo_mode = demo_mode
            # Forcer un rafraîchissement pour appliquer les changements
            if not demo_mode:
                st.info("Mode démo désactivé. Assurez-vous que vos identifiants sont corrects.")
                dashboard_service.update_data()
        
        # Montant initial
        initial_capital = st.number_input(
            "Capital initial (USDT)",
            min_value=100.0,
            max_value=100000.0,
            value=float(st.session_state.trading_params['initial_capital']),
            step=100.0
        )
        
        # Risque par trade
        risk_per_trade = st.slider(
            "Risque par trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.trading_params['risk_per_trade']),
            step=0.1
        )
        
        # Stratégie
        strategies = ["EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Combinaison"]
        selected_strategies = st.multiselect(
            "Stratégies",
            strategies,
            default=[strategy for strategy in strategies if strategy in st.session_state.trading_params['strategy']]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section Gestion des risques
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
        
        # Section Avancée
        with st.expander("Paramètres Avancés"):
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            
            time_frame = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=2  # 15m par défaut
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
        
        # Bouton de sauvegarde
        if st.button("💾 Enregistrer", type="primary"):
            # Mettre à jour les paramètres
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
            
            # Sauvegarder les paramètres
            dashboard_service.save_trading_params(params)
            st.success("✅ Configuration sauvegardée.")

def check_critical_alerts():
    """Vérifie s'il y a des alertes critiques à afficher."""
    # Exemple: alerte si le drawdown dépasse un certain seuil
    max_drawdown = st.session_state.account_stats.get('max_drawdown', 0)
    if max_drawdown > 15:
        st.markdown(
            f'<div class="alert">⚠️ ALERTE: Drawdown élevé détecté ({max_drawdown:.2f}%)</div>',
            unsafe_allow_html=True
        )

def main():
    """Fonction principale."""
    # Initialisation des variables de session
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 10
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Réinitialiser l'état si nécessaire
    if 'bot_status' not in st.session_state:
        st.session_state.bot_status = "Inactif"
    
    # Appliquer le style CSS
    apply_css()
    
    # Afficher le panneau de configuration
    config_panel()
    
    # En-tête
    header()
    
    # Contrôles de rafraîchissement
    refresh_controls()
    
    # Sélecteur de symbole
    symbol_selector()
    
    # Vérifier les alertes critiques
    check_critical_alerts()
    
    # Afficher les logs en position plus importante (avant le graphique)
    logs_console()
    
    # Disposition en deux colonnes pour le graphique et les statistiques
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Afficher le graphique des prix
        price_chart()
    
    with col2:
        # Afficher les statistiques
        statistics()
        # Afficher l'historique des trades
        trades_history()
    
    # Si c'est le premier chargement, charger les données
    if not st.session_state.data_loaded:
        try:
            dashboard_service.update_data()
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
        except Exception as e:
            st.error(f"Erreur lors du chargement initial: {e}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # S'assurer que le thread est arrêté à la fermeture
        refresh_manager.stop()