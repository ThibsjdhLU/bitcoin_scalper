"""
Composants UI réutilisables pour l'interface NiceGUI.
"""

from nicegui import ui
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

class PriceChart:
    def __init__(self):
        # Create initial empty figure
        self.fig = go.Figure()
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        
        # Create Plotly chart with the figure
        self.chart = ui.plotly(self.fig).classes('w-full h-96')
        
    def update(self, df: pd.DataFrame):
        """Met à jour le graphique avec les nouvelles données"""
        # Add new data to the figure
        self.fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Prix'
        ))
        
        # Mise en forme
        self.fig.update_layout(
            title='Bitcoin Price',
            yaxis_title='Price (USD)',
            xaxis_title='Time',
            template='plotly_dark'
        )
        
        # Update the chart display
        self.chart.update()

class LogConsole:
    def __init__(self):
        self.console = ui.textarea().classes('w-full h-32').props('readonly')
        self.logs = []
        
    def add_log(self, message: str):
        """Ajoute un message aux logs"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        self.console.set_value('\n'.join(self.logs[-100:]))  # Garder les 100 derniers logs

class ControlPanel:
    def __init__(self, on_start=None, on_stop=None, on_reset=None):
        with ui.card().classes('w-full p-4'):
            ui.label('Contrôles').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('w-full justify-between'):
                self.start_btn = ui.button('Démarrer', on_click=on_start)
                self.stop_btn = ui.button('Arrêter', on_click=on_stop)
                self.reset_btn = ui.button('Réinitialiser', on_click=on_reset)
            
            self.strategy_select = ui.select(
                ['MACD', 'RSI', 'Bollinger Bands'],
                label='Stratégie',
                value='MACD'
            ).classes('w-full mt-4')

class StatsPanel:
    def __init__(self):
        with ui.card().classes('w-full p-4'):
            ui.label('Statistiques').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('w-full justify-between'):
                with ui.card().classes('w-1/3'):
                    ui.label('PnL').classes('text-lg font-bold')
                    self.pnl_label = ui.label('0.00')
                
                with ui.card().classes('w-1/3'):
                    ui.label('Positions').classes('text-lg font-bold')
                    self.positions_label = ui.label('0')
                
                with ui.card().classes('w-1/3'):
                    ui.label('Trades').classes('text-lg font-bold')
                    self.trades_label = ui.label('0')
    
    def update(self, pnl: float, positions: int, trades: int):
        """Met à jour les statistiques"""
        self.pnl_label.set_text(f"{pnl:.2f}")
        self.positions_label.set_text(str(positions))
        self.trades_label.set_text(str(trades)) 