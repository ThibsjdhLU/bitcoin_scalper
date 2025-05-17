"""
Composants pour les paramètres et la configuration du bot.
"""

from nicegui import ui
from .components import CARD_SHADOW, CARD_RADIUS, PRIMARY_COLOR, SUCCESS_COLOR, ERROR_COLOR

class TradingSettings:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Paramètres de Trading').classes('text-xl font-bold mb-4')
            
            # Paires de trading
            with ui.row().classes('w-full gap-4'):
                with ui.column().classes('w-1/2'):
                    ui.label('Paires de Trading').classes('text-lg font-semibold mb-2')
                    self.pairs = ui.select(
                        ['BTCUSD', 'ETHUSD', 'XRPUSD'],
                        multiple=True,
                        with_input=True
                    ).classes('w-full')
                
                # Timeframes
                with ui.column().classes('w-1/2'):
                    ui.label('Timeframes').classes('text-lg font-semibold mb-2')
                    self.timeframes = ui.select(
                        ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
                        multiple=True
                    ).classes('w-full')

            # Paramètres de risque
            with ui.row().classes('w-full gap-4 mt-4'):
                with ui.column().classes('w-1/3'):
                    ui.label('Risque par Trade (%)').classes('text-lg font-semibold mb-2')
                    self.risk_per_trade = ui.number(
                        value=1.0,
                        min=0.1,
                        max=5.0,
                        step=0.1
                    ).classes('w-full')
                
                with ui.column().classes('w-1/3'):
                    ui.label('Stop Loss (pips)').classes('text-lg font-semibold mb-2')
                    self.stop_loss = ui.number(
                        value=50,
                        min=10,
                        max=200,
                        step=10
                    ).classes('w-full')
                
                with ui.column().classes('w-1/3'):
                    ui.label('Take Profit (pips)').classes('text-lg font-semibold mb-2')
                    self.take_profit = ui.number(
                        value=100,
                        min=20,
                        max=400,
                        step=20
                    ).classes('w-full')

    def get_settings(self) -> dict:
        return {
            'pairs': self.pairs.value,
            'timeframes': self.timeframes.value,
            'risk_per_trade': self.risk_per_trade.value,
            'stop_loss': self.stop_loss.value,
            'take_profit': self.take_profit.value
        }

class NotificationSettings:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Notifications').classes('text-xl font-bold mb-4')
            
            # Types de notifications
            with ui.column().classes('w-full gap-4'):
                self.trade_notifications = ui.checkbox('Notifications de Trades').classes('text-lg')
                self.error_notifications = ui.checkbox('Notifications d\'Erreurs').classes('text-lg')
                self.performance_notifications = ui.checkbox('Rapports de Performance').classes('text-lg')
                
                # Fréquence des rapports
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.label('Fréquence des Rapports').classes('text-lg font-semibold')
                    self.report_frequency = ui.select(
                        ['Quotidien', 'Hebdomadaire', 'Mensuel'],
                        value='Quotidien'
                    ).classes('w-full')

    def get_settings(self) -> dict:
        return {
            'trade_notifications': self.trade_notifications.value,
            'error_notifications': self.error_notifications.value,
            'performance_notifications': self.performance_notifications.value,
            'report_frequency': self.report_frequency.value
        }

class APISettings:
    def __init__(self):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Configuration API').classes('text-xl font-bold mb-4')
            
            # MT5
            with ui.column().classes('w-full gap-4'):
                ui.label('MetaTrader 5').classes('text-lg font-semibold')
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('w-1/2'):
                        ui.label('Login').classes('text-sm')
                        self.mt5_login = ui.input(type='number').classes('w-full')
                    with ui.column().classes('w-1/2'):
                        ui.label('Mot de passe').classes('text-sm')
                        self.mt5_password = ui.input(type='password').classes('w-full')
                
                # Telegram
                ui.label('Telegram').classes('text-lg font-semibold mt-4')
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('w-1/2'):
                        ui.label('Token Bot').classes('text-sm')
                        self.telegram_token = ui.input(type='password').classes('w-full')
                    with ui.column().classes('w-1/2'):
                        ui.label('Chat ID').classes('text-sm')
                        self.telegram_chat_id = ui.input().classes('w-full')

    def get_settings(self) -> dict:
        return {
            'mt5': {
                'login': self.mt5_login.value,
                'password': self.mt5_password.value
            },
            'telegram': {
                'token': self.telegram_token.value,
                'chat_id': self.telegram_chat_id.value
            }
        } 