"""
Composants d'analyse et de performance pour le bot de trading.
"""

from nicegui import ui
import plotly.graph_objects as go
from .components import CARD_SHADOW, CARD_RADIUS, PRIMARY_COLOR, SUCCESS_COLOR, ERROR_COLOR

class PerformanceMetrics:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Métriques de Performance').classes('text-xl font-bold mb-4')
            with ui.grid(columns=3).classes('gap-4'):
                # PnL Cumulé
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('PnL Cumulé').classes('text-sm')
                    self.total_pnl = ui.label('0.00 €').classes('text-xl')
                # Drawdown Max
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Drawdown Max').classes('text-sm')
                    self.max_drawdown = ui.label('0.00%').classes('text-xl')
                # Win Rate
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Win Rate').classes('text-sm')
                    self.win_rate = ui.label('0.00%').classes('text-xl')

    def update(self, total_pnl: float, max_drawdown: float, win_rate: float):
        self.total_pnl.set_text(f"{total_pnl:.2f} €")
        self.max_drawdown.set_text(f"{max_drawdown:.2f}%")
        self.win_rate.set_text(f"{win_rate:.2f}%")

class BenchmarkComparison:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Comparaison avec BTC Hold').classes('text-xl font-bold mb-4')
            self.fig = go.Figure()
            self.fig.update_layout(
                template='plotly_dark',
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
            self.chart = ui.plotly(self.fig).classes('w-full h-64')

    def update(self, bot_returns: list, btc_returns: list, dates: list):
        self.fig.data = []
        self.fig.add_trace(go.Scatter(
            x=dates, y=bot_returns,
            name='Bot Performance',
            line=dict(color=PRIMARY_COLOR)
        ))
        self.fig.add_trace(go.Scatter(
            x=dates, y=btc_returns,
            name='BTC Hold',
            line=dict(color='gray')
        ))
        self.chart.update()

class RiskMetrics:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Métriques de Risque').classes('text-xl font-bold mb-4')
            with ui.grid(columns=3).classes('gap-4'):
                # Sharpe Ratio
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Sharpe Ratio').classes('text-sm')
                    self.sharpe = ui.label('0.00').classes('text-xl')
                # Sortino Ratio
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Sortino Ratio').classes('text-sm')
                    self.sortino = ui.label('0.00').classes('text-xl')
                # Ratio Gain/Perte
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Ratio G/P').classes('text-sm')
                    self.profit_loss = ui.label('0.00').classes('text-xl')

    def update(self, sharpe: float, sortino: float, profit_loss: float):
        self.sharpe.set_text(f"{sharpe:.2f}")
        self.sortino.set_text(f"{sortino:.2f}")
        self.profit_loss.set_text(f"{profit_loss:.2f}")

class TradeAnalysis:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Analyse des Trades').classes('text-xl font-bold mb-4')
            with ui.tabs().classes('w-full') as tabs:
                with ui.tab('Distribution'):
                    self.distribution_chart = ui.plotly(go.Figure()).classes('w-full h-64')
                with ui.tab('Horaires'):
                    self.time_chart = ui.plotly(go.Figure()).classes('w-full h-64')
                with ui.tab('Corrélations'):
                    self.correlation_chart = ui.plotly(go.Figure()).classes('w-full h-64') 