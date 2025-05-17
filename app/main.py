"""
Application principale du bot de trading Bitcoin.
"""

import logging
import traceback
import socket
from nicegui import ui, app
from ui.dashboard import BotStatus, PerformanceMetrics, RecentTrades, EmergencyStop
from ui.analysis import PerformanceMetrics as AnalysisMetrics, BenchmarkComparison, RiskMetrics, TradeAnalysis
from ui.backtest import BacktestResults, ModelComparison, ParameterHeatmap, ModelParameters
from ui.settings import TradingSettings, NotificationSettings, APISettings
from ui.components import Sidebar, Notifier, PriceChart, LogConsole

# Configuration du moteur de rendu
app.native.start_args['gui'] = 'qt'

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Trouve un port disponible."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Aucun port disponible entre {start_port} et {start_port + max_attempts - 1}")

# État global de l'application
app_state = {
    'is_running': False,
    'pnl': 0.0,
    'positions': [],
    'trades': [],
    'metrics': {
        'daily': 0.0,
        'weekly': 0.0,
        'monthly': 0.0
    }
}

def show_error_dialog(exc: Exception):
    """Affiche une boîte de dialogue d'erreur."""
    try:
        with ui.dialog() as dialog, ui.card():
            dialog.props("no-esc-dismiss no-backdrop-dismiss")
            ui.label(f"Une erreur est survenue: {exc}").classes('text-red-500')
            ui.label(traceback.format_exc()).classes(
                "bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative"
            ).style("white-space: pre-wrap")
            with ui.row().classes('w-full justify-end'):
                ui.button('Fermer', on_click=dialog.close).classes('bg-red-500 text-white')
            dialog.open()
    except Exception:
        logger.error("Impossible d'afficher la boîte de dialogue d'erreur")

# Gestionnaire d'exceptions global
app.on_exception(show_error_dialog)

def update_ui():
    """Met à jour l'interface utilisateur avec les dernières données."""
    try:
        # Mise à jour des composants du dashboard
        bot_status.update(app_state['is_running'])
        performance_metrics.update(app_state['metrics'])
        recent_trades.update(app_state['trades'])
        
        # Mise à jour des composants d'analyse
        analysis_metrics.update(app_state['metrics'])
        risk_metrics.update(app_state['metrics'])
        trade_analysis.update(app_state['trades'])
        
        # Mise à jour du graphique de prix
        price_chart.update(app_state['positions'])
        
        # Mise à jour de la console de logs
        log_console.update()
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'UI: {e}")
        notifier.error(f"Erreur UI: {e}")

def main():
    """Fonction principale de l'application."""
    try:
        # Configuration de la page
        ui.page_title('Bitcoin Trading Bot')
        ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">')
        
        # Layout principal
        with ui.row().classes('w-full h-screen bg-slate-900'):
            # Sidebar
            sidebar = Sidebar()
            
            # Contenu principal
            with ui.column().classes('flex-1 p-4 gap-4'):
                # En-tête
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Dashboard').classes('text-2xl font-bold text-white')
                    emergency_stop = EmergencyStop()
                
                # Dashboard
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('w-1/3'):
                        bot_status = BotStatus()
                        performance_metrics = PerformanceMetrics()
                    with ui.column().classes('w-2/3'):
                        price_chart = PriceChart()
                
                # Métriques et trades récents
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('w-2/3'):
                        recent_trades = RecentTrades()
                    with ui.column().classes('w-1/3'):
                        log_console = LogConsole()
                
                # Analyse
                with ui.tabs().classes('w-full') as tabs:
                    with ui.tab('Performance'):
                        with ui.column().classes('w-full gap-4'):
                            analysis_metrics = AnalysisMetrics()
                            benchmark_comparison = BenchmarkComparison()
                    with ui.tab('Risque'):
                        with ui.column().classes('w-full gap-4'):
                            risk_metrics = RiskMetrics()
                            trade_analysis = TradeAnalysis()
                    with ui.tab('Backtest'):
                        with ui.column().classes('w-full gap-4'):
                            backtest_results = BacktestResults()
                            model_comparison = ModelComparison()
                            parameter_heatmap = ParameterHeatmap()
                            model_parameters = ModelParameters()
                    with ui.tab('Paramètres'):
                        with ui.column().classes('w-full gap-4'):
                            trading_settings = TradingSettings()
                            notification_settings = NotificationSettings()
                            api_settings = APISettings()
        
        # Notifications
        notifier = Notifier()
        
        # Mise à jour périodique de l'UI
        ui.timer(1.0, update_ui)
        
        logger.info("Application démarrée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application: {e}")
        show_error_dialog(e)
        raise

if __name__ in {"__main__", "__mp_main__"}:
    try:
        port = find_available_port()
        logger.info(f"Lancement de l'application sur le port {port}")
        ui.run(port=port, reload=False)
    except KeyboardInterrupt:
        logger.info("Application arrêtée par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        show_error_dialog(e)