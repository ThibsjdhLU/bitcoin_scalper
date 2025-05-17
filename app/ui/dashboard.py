"""
Composants du dashboard principal pour le bot de trading.
"""

from nicegui import ui
from datetime import datetime
import pandas as pd
from .components import CARD_SHADOW, CARD_RADIUS, PRIMARY_COLOR, SUCCESS_COLOR, ERROR_COLOR

class BotStatus:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            with ui.row().classes('items-center justify-between'):
                ui.label('État du Bot').classes('text-xl font-bold')
                self.status = ui.label('Inactif').classes('text-lg')
            with ui.row().classes('mt-4'):
                self.uptime = ui.label('Uptime: 0h 0m').classes('text-sm')
                self.last_update = ui.label('Dernière mise à jour: -').classes('text-sm')

    def update(self, is_running: bool):
        """Met à jour l'état du bot."""
        try:
            status_text = "Actif" if is_running else "Inactif"
            status_color = SUCCESS_COLOR if is_running else ERROR_COLOR
            self.status.set_text(status_text)
            self.status.classes(f'text-lg text-{status_color}')
            self.last_update.set_text(f'Dernière mise à jour: {datetime.now().strftime("%H:%M:%S")}')
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du statut du bot: {e}")

class PerformanceMetrics:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Performance').classes('text-xl font-bold mb-4')
            with ui.grid(columns=3).classes('gap-4'):
                # Journalier
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Journalier').classes('text-sm')
                    self.daily_pnl = ui.label('0.00 €').classes('text-xl')
                    self.daily_pct = ui.label('0.00%').classes('text-sm')
                # Hebdomadaire
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Hebdomadaire').classes('text-sm')
                    self.weekly_pnl = ui.label('0.00 €').classes('text-xl')
                    self.weekly_pct = ui.label('0.00%').classes('text-sm')
                # Mensuel
                with ui.card().classes(f'bg-slate-700 {CARD_RADIUS}'):
                    ui.label('Mensuel').classes('text-sm')
                    self.monthly_pnl = ui.label('0.00 €').classes('text-xl')
                    self.monthly_pct = ui.label('0.00%').classes('text-sm')

    def update(self, metrics: dict):
        """Met à jour les métriques de performance."""
        try:
            daily_pnl, daily_pct = metrics.get('daily', (0.0, 0.0))
            weekly_pnl, weekly_pct = metrics.get('weekly', (0.0, 0.0))
            monthly_pnl, monthly_pct = metrics.get('monthly', (0.0, 0.0))
            
            self.daily_pnl.set_text(f"{daily_pnl:.2f} €")
            self.daily_pct.set_text(f"{daily_pct:.2f}%")
            self.weekly_pnl.set_text(f"{weekly_pnl:.2f} €")
            self.weekly_pct.set_text(f"{weekly_pct:.2f}%")
            self.monthly_pnl.set_text(f"{monthly_pnl:.2f} €")
            self.monthly_pct.set_text(f"{monthly_pct:.2f}%")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques: {e}")

class RecentTrades:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            with ui.row().classes('justify-between items-center mb-4'):
                ui.label('Derniers Trades').classes('text-xl font-bold')
                ui.button('Voir tout', icon='list').classes('bg-blue-500')
            self.table = ui.table({
                'columns': [
                    {'name': 'time', 'label': 'Heure', 'field': 'time'},
                    {'name': 'pair', 'label': 'Paire', 'field': 'pair'},
                    {'name': 'type', 'label': 'Type', 'field': 'type'},
                    {'name': 'price', 'label': 'Prix', 'field': 'price'},
                    {'name': 'pnl', 'label': 'PnL', 'field': 'pnl'}
                ],
                'rows': []
            }).classes('w-full')

    def update(self, trades: list):
        """Met à jour la liste des trades récents."""
        try:
            self.table.rows = trades
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des trades: {e}")

class EmergencyStop:
    def __init__(self, on_stop=None):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Arrêt d\'Urgence').classes('text-xl font-bold mb-4')
            ui.button('STOP BOT', on_click=on_stop, icon='warning').classes('w-full bg-red-500 text-white text-xl py-4') 