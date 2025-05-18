"""
Application principale du bot de trading Bitcoin.
Interface utilisateur NiceGUI pour le contrôle et le monitoring du bot.
"""

import asyncio
from pathlib import Path
import sys
import logging
from datetime import datetime
import warnings
from nicegui import ui, app
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.unified_config import config
from app.core.app_state import app_state
from app.core.refresh_manager import RefreshManager
from app.services.trading_service import TradingService
from app.ui.components import PriceChart, LogConsole, ControlPanel, StatsPanel, Sidebar, Notifier, t

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialisation des services
trading_service = TradingService(app_state)
refresh_manager = RefreshManager(app_state)

# Composants UI globaux
price_chart = None
log_console = None
stats_panel = None
pnl_history = []

async def update_ui():
    """Met à jour l'interface utilisateur"""
    try:
        # Mise à jour du graphique
        market_data = app_state.get('market_data')
        if market_data is not None:
            price_chart.update(market_data)
        # Mise à jour des statistiques
        pnl = app_state.get('pnl', 0.0)
        positions = app_state.get('positions', 0)
        trades = app_state.get('trades', 0)
        global pnl_history
        pnl_history.append(pnl)
        stats_panel.update(pnl, positions, trades, pnl_history=pnl_history[-30:])
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'UI: {e}")

@ui.page('/')
def main():
    global price_chart, log_console, stats_panel
    Sidebar()  # Ajout de la barre latérale
    # Switch dark/light mode
    with ui.row().classes('justify-end p-2'):
        ui.toggle(['🌙', '☀️'], value='🌙', on_change=lambda e: ui.query('body').classes('bg-slate-900' if e.value == '🌙' else 'bg-white'))
    # En-tête modernisé
    with ui.header().classes('bg-blue-600 text-white shadow-lg rounded-b-xl'):
        with ui.row().classes('items-center justify-between w-full'):
            ui.image('/static/logo.png').classes('w-12 h-12 mr-4')
            ui.label('Bitcoin Trading Bot').classes('text-2xl font-bold')
            with ui.row().classes('items-center'):
                ui.label(f"{t('Statut')}: {app_state.get('bot_status')}").classes('mr-4')
                ui.label(f"{t('Dernière mise à jour')}: {app_state.get('last_refresh')}")
                ui.image('/static/avatar.png').classes('w-10 h-10 rounded-full ml-4')
    # Panneau d'alertes (placeholder)
    with ui.card().classes('w-full bg-red-100 text-red-800 my-2'):
        ui.label('Aucune alerte critique').classes('font-bold')
    # Corps principal responsive
    with ui.row().classes('w-full p-4 flex-wrap'):
        # Panneau de configuration
        control_panel = ControlPanel(
            on_start=lambda: asyncio.create_task(start_bot()),
            on_stop=lambda: asyncio.create_task(stop_bot()),
            on_reset=lambda: asyncio.create_task(reset_bot())
        )
        # Graphique et statistiques
        with ui.card().classes('w-3/4 min-w-[350px]'):
            price_chart = PriceChart()
            stats_panel = StatsPanel()
    # Console de logs
    log_console = LogConsole()
    # Footer
    with ui.footer().classes('bg-slate-800 text-white text-center p-2 mt-8'):
        ui.label('Bitcoin Scalper v1.0 © 2024')
    # Enregistrer les callbacks de mise à jour
    refresh_manager.register_update_callback('market_data', trading_service.update_market_data)
    refresh_manager.register_update_callback('positions', trading_service.update_positions)
    refresh_manager.register_update_callback('trades', trading_service.update_trades_history)
    refresh_manager.register_update_callback('ui', update_ui)
    # Connexion MT5
    asyncio.create_task(trading_service.connect())

async def start_bot():
    log_console.add_log("Démarrage du bot...", level='success')
    Notifier.toast('Bot démarré', type_='success')
    await refresh_manager.start()

async def stop_bot():
    log_console.add_log("Arrêt du bot...", level='warning')
    Notifier.toast('Bot arrêté', type_='error')
    await refresh_manager.stop()

async def reset_bot():
    log_console.add_log("Réinitialisation du bot...", level='info')
    Notifier.toast('Bot réinitialisé', type_='info')
    await stop_bot()
    await trading_service.disconnect()
    await trading_service.connect()
    app_state.set('pnl', 0.0)
    app_state.set('positions', 0)
    app_state.set('trades', 0)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run()