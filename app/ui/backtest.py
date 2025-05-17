"""
Composants pour les backtests et les modèles de trading.
"""

from nicegui import ui
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from .components import CARD_SHADOW, CARD_RADIUS, PRIMARY_COLOR, SUCCESS_COLOR, ERROR_COLOR

class BacktestResults:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Résultats des Backtests').classes('text-xl font-bold mb-4')
            with ui.tabs().classes('w-full') as tabs:
                with ui.tab('Performance'):
                    self.performance_chart = ui.plotly(go.Figure()).classes('w-full h-64')
                with ui.tab('Drawdown'):
                    self.drawdown_chart = ui.plotly(go.Figure()).classes('w-full h-64')
                with ui.tab('Trades'):
                    self.trades_chart = ui.plotly(go.Figure()).classes('w-full h-64')

    def update(self, results: dict):
        # Performance
        fig = go.Figure()
        for name, data in results['performance'].items():
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['equity'],
                name=name,
                line=dict(color=data['color'])
            ))
        fig.update_layout(template='plotly_dark', showlegend=True)
        self.performance_chart.update(fig)

        # Drawdown
        fig = go.Figure()
        for name, data in results['drawdown'].items():
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['drawdown'],
                name=name,
                line=dict(color=data['color'])
            ))
        fig.update_layout(template='plotly_dark', showlegend=True)
        self.drawdown_chart.update(fig)

        # Trades
        fig = go.Figure()
        for name, data in results['trades'].items():
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['pnl'],
                name=name,
                mode='markers',
                marker=dict(
                    color=data['colors'],
                    size=8
                )
            ))
        fig.update_layout(template='plotly_dark', showlegend=True)
        self.trades_chart.update(fig)

class ModelComparison:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Comparaison des Modèles').classes('text-xl font-bold mb-4')
            self.table = ui.table({
                'columns': [
                    {'name': 'model', 'label': 'Modèle', 'field': 'model'},
                    {'name': 'sharpe', 'label': 'Sharpe', 'field': 'sharpe'},
                    {'name': 'sortino', 'label': 'Sortino', 'field': 'sortino'},
                    {'name': 'win_rate', 'label': 'Win Rate', 'field': 'win_rate'},
                    {'name': 'max_dd', 'label': 'Max DD', 'field': 'max_dd'},
                    {'name': 'profit_factor', 'label': 'Profit Factor', 'field': 'profit_factor'}
                ],
                'rows': []
            }).classes('w-full')

    def update(self, models: list):
        self.table.rows = models

class ParameterHeatmap:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Heatmap des Paramètres').classes('text-xl font-bold mb-4')
            self.heatmap = ui.plotly(go.Figure()).classes('w-full h-96')

    def update(self, data: pd.DataFrame, x_param: str, y_param: str, metric: str):
        fig = px.imshow(
            data,
            x=x_param,
            y=y_param,
            color=metric,
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        self.heatmap.update(fig)

class ModelParameters:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Paramètres du Modèle').classes('text-xl font-bold mb-4')
            with ui.tabs().classes('w-full') as tabs:
                with ui.tab('Stratégie'):
                    self.strategy_params = ui.column().classes('w-full')
                with ui.tab('Risque'):
                    self.risk_params = ui.column().classes('w-full')
                with ui.tab('Technique'):
                    self.tech_params = ui.column().classes('w-full') 