"""
Composants UI réutilisables pour l'interface NiceGUI.
"""

from nicegui import ui, events
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Palette de couleurs
PRIMARY_COLOR = '#2563eb'  # Bleu
DARK_BG = '#1e293b'        # Gris foncé
SUCCESS_COLOR = '#22c55e'  # Vert
ERROR_COLOR = '#ef4444'    # Rouge
CARD_SHADOW = 'shadow-lg'
CARD_RADIUS = 'rounded-xl'

# Placeholder pour le logo et l'avatar
LOGO_PATH = '/static/logo.png'
AVATAR_PATH = '/static/avatar.png'

# Sidebar/navigation
class Sidebar:
    def __init__(self):
        with ui.left_drawer().classes('bg-slate-800 text-white'):
            ui.image(LOGO_PATH).classes('w-24 mx-auto my-4')
            ui.label('Bitcoin Scalper').classes('text-xl font-bold text-center mb-4')
            ui.separator()
            ui.link('Dashboard', '/').classes('block py-2 px-4 hover:bg-slate-700 rounded')
            ui.link('Configuration', '/config').classes('block py-2 px-4 hover:bg-slate-700 rounded')
            ui.link('Aide', '/help').classes('block py-2 px-4 hover:bg-slate-700 rounded')
            ui.separator()
            with ui.row().classes('justify-center mt-8'):
                ui.image(AVATAR_PATH).classes('w-12 h-12 rounded-full border-2 border-white')
                ui.label('Utilisateur').classes('ml-2')

# Toast notifications
class Notifier:
    @staticmethod
    def toast(message, type_='info'):
        color = {'info': PRIMARY_COLOR, 'success': SUCCESS_COLOR, 'error': ERROR_COLOR}.get(type_, PRIMARY_COLOR)
        ui.notify(message, color=color, position='top-right', timeout=2500)

class PriceChart:
    def __init__(self):
        self.fig = go.Figure()
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        self.chart = ui.plotly(self.fig).classes('w-full h-96')
        self.chart.props('dark')
        # Ajout d'un switch candle/line et zoom
        with ui.row().classes('justify-end mb-2'):
            self.chart_type = ui.toggle(['Bougies', 'Ligne'], value='Bougies')
            self.zoom_slider = ui.slider(min=10, max=200, value=100, step=10, label='Zoom')
        self.data = None

    def update(self, df: pd.DataFrame):
        self.data = df
        self.fig.data = []
        if self.chart_type.value == 'Bougies':
            self.fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Prix',
                increasing_line_color=SUCCESS_COLOR, decreasing_line_color=ERROR_COLOR
            ))
        else:
            self.fig.add_trace(go.Scatter(
                x=df.index, y=df['close'], mode='lines', name='Prix',
                line=dict(color=PRIMARY_COLOR, width=2)
            ))
        self.fig.update_layout(
            title='Bitcoin Price', yaxis_title='Price (USD)', xaxis_title='Time', template='plotly_dark',
            plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG
        )
        self.chart.update()

class LogConsole:
    def __init__(self):
        self.console = ui.textarea().classes('w-full h-32 bg-slate-900 text-white').props('readonly')
        self.logs = []

    def add_log(self, message: str, level='info'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        color = {'info': 'white', 'success': SUCCESS_COLOR, 'error': ERROR_COLOR, 'warning': '#facc15'}.get(level, 'white')
        self.logs.append(f"<span style='color:{color}'>[{timestamp}] {message}</span>")
        self.console.set_value('\n'.join(self.logs[-100:]))
        self.console.props('scrollable')

class ControlPanel:
    def __init__(self, on_start=None, on_stop=None, on_reset=None):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Contrôles').classes('text-xl font-bold mb-4')
            with ui.row().classes('w-full justify-between'):
                ui.button('Démarrer', on_click=lambda: self.confirm(on_start, 'Démarrer le bot ?', 'success'), icon='play_arrow', color=SUCCESS_COLOR)
                ui.button('Arrêter', on_click=lambda: self.confirm(on_stop, 'Arrêter le bot ?', 'error'), icon='stop', color=ERROR_COLOR)
                ui.button('Réinitialiser', on_click=lambda: self.confirm(on_reset, 'Réinitialiser le bot ?', 'info'), icon='refresh', color=PRIMARY_COLOR)
            ui.label('Stratégie').classes('mt-4')
            self.strategy_select = ui.select(
                options=['MACD', 'RSI', 'Bollinger Bands'],
                label='Stratégie', value='MACD', with_input=False
            ).classes('w-full mt-2')
            # Description dynamique sous le select
            self.strategy_desc = ui.label('').classes('text-sm mt-1')
            def update_desc(e=None):
                descs = {
                    'MACD': 'Moving Average Convergence Divergence',
                    'RSI': 'Relative Strength Index',
                    'Bollinger Bands': 'Bandes de volatilité'
                }
                self.strategy_desc.set_text(descs.get(self.strategy_select.value, ''))
            self.strategy_select.on('update:model-value', update_desc)
            update_desc()

    def confirm(self, callback, message, type_):
        def on_confirm():
            Notifier.toast(f'Action confirmée : {message}', type_)
            if callback:
                callback()
        ui.dialog(message, on_confirm=on_confirm)

class StatsPanel:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Statistiques').classes('text-xl font-bold mb-4')
            with ui.row().classes('w-full justify-between'):
                with ui.card().classes(f'w-1/3 bg-slate-700 {CARD_RADIUS}'):
                    ui.label('PnL').classes('text-lg font-bold')
                    self.pnl_label = ui.label('0.00').classes('text-2xl')
                    self.pnl_spark = ui.sparkline([0], color=SUCCESS_COLOR)
                with ui.card().classes(f'w-1/3 bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Positions').classes('text-lg font-bold')
                    self.positions_label = ui.label('0').classes('text-2xl')
                with ui.card().classes(f'w-1/3 bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Trades').classes('text-lg font-bold')
                    self.trades_label = ui.label('0').classes('text-2xl')

    def update(self, pnl: float, positions: int, trades: int, pnl_history=None):
        self.pnl_label.set_text(f"{pnl:.2f}")
        self.positions_label.set_text(str(positions))
        self.trades_label.set_text(str(trades))
        if pnl_history:
            self.pnl_spark.set_value(pnl_history)

# Internationalisation (français/anglais)
LANG = 'fr'
TRANSLATIONS = {
    'fr': {
        'Statut': 'Statut',
        'Dernière mise à jour': 'Dernière mise à jour',
        'Démarrer': 'Démarrer',
        'Arrêter': 'Arrêter',
        'Réinitialiser': 'Réinitialiser',
        'Stratégie': 'Stratégie',
        'Statistiques': 'Statistiques',
        'Positions': 'Positions',
        'Trades': 'Trades',
        'Contrôles': 'Contrôles',
        'Utilisateur': 'Utilisateur',
    },
    'en': {
        'Statut': 'Status',
        'Dernière mise à jour': 'Last update',
        'Démarrer': 'Start',
        'Arrêter': 'Stop',
        'Réinitialiser': 'Reset',
        'Stratégie': 'Strategy',
        'Statistiques': 'Statistics',
        'Positions': 'Positions',
        'Trades': 'Trades',
        'Contrôles': 'Controls',
        'Utilisateur': 'User',
    }
}
def t(key):
    return TRANSLATIONS.get(LANG, {}).get(key, key) 