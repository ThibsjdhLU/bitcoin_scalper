class ControlPanel:
    def __init__(self, on_start=None, on_stop=None, on_reset=None):
        with ui.card().classes(f'w-full p-4 bg-slate-800 text-white {CARD_SHADOW} {CARD_RADIUS}'):
            ui.label('Contrôles').classes('text-xl font-bold mb-4')
            with ui.row().classes('w-full justify-between'):
                ui.button('Démarrer', on_click=lambda: self.confirm(on_start, 'Démarrer le bot ?', 'success'), icon='play_arrow').classes('bg-green-500')
                ui.button('Arrêter', on_click=lambda: self.confirm(on_stop, 'Arrêter le bot ?', 'error'), icon='stop').classes('bg-red-500')
                ui.button('Réinitialiser', on_click=lambda: self.confirm(on_reset, 'Réinitialiser le bot ?', 'info'), icon='refresh').classes('bg-blue-500')
            ui.label('Stratégie').classes('mt-4')
            self.strategy_select = ui.select(
                options=['MACD', 'RSI', 'Bollinger Bands'],
                label='Stratégie',
                value='MACD',
                with_input=False
            ).classes('w-full mt-2')
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
        with ui.dialog() as dialog, ui.card():
            ui.label(message)
            with ui.row().classes('w-full justify-end'):
                ui.button('Annuler', on_click=dialog.close).classes('bg-gray-500')
                ui.button('Confirmer', on_click=lambda: [Notifier.toast(f'Action confirmée : {message}', type_), callback(), dialog.close()]).classes('bg-blue-500') 