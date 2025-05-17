"""
Service de trading pour NiceGUI.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import MetaTrader5 as mt5
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

class TradingService:
    def __init__(self, app_state):
        self.app_state = app_state
        self._connected = False
        self._current_strategy = None
        self._positions: List[Dict[str, Any]] = []
        self._trades_history: List[Dict[str, Any]] = []
        
    async def connect(self) -> bool:
        """Connexion à MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error("Échec de l'initialisation MT5")
                return False
            
            self._connected = True
            logger.info("Connexion MT5 réussie")
            return True
            
        except Exception as e:
            logger.error(f"Erreur de connexion MT5: {e}")
            return False
    
    async def disconnect(self):
        """Déconnexion de MetaTrader 5"""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Déconnexion MT5")
    
    async def update_market_data(self):
        """Met à jour les données de marché"""
        if not self._connected:
            return
        
        try:
            # Récupérer les données OHLC
            rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M1, 0, 1000)
            if rates is None:
                logger.error("Échec de récupération des données")
                return
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Mettre à jour l'état
            self.app_state.set('market_data', df)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données: {e}")
    
    async def update_positions(self):
        """Met à jour les positions ouvertes"""
        if not self._connected:
            return
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                self._positions = []
            else:
                self._positions = [
                    {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'price': pos.price_open,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit
                    }
                    for pos in positions
                ]
            
            # Calculer le PnL total
            total_pnl = sum(pos['profit'] for pos in self._positions)
            self.app_state.set('pnl', total_pnl)
            self.app_state.set('positions', len(self._positions))
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des positions: {e}")
    
    async def execute_trade(self, symbol: str, type: str, volume: float, 
                          price: Optional[float] = None, sl: Optional[float] = None, 
                          tp: Optional[float] = None) -> bool:
        """Exécute un ordre de trading"""
        if not self._connected:
            return False
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price if price else mt5.symbol_info_tick(symbol).ask,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Échec de l'exécution de l'ordre: {result.comment}")
                return False
            
            logger.info(f"Ordre exécuté: {type} {volume} {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre: {e}")
            return False
    
    async def update_trades_history(self):
        """Met à jour l'historique des trades"""
        if not self._connected:
            return
        
        try:
            from_date = datetime.now() - pd.Timedelta(days=7)
            history = mt5.history_deals_get(from_date)
            
            if history is None:
                self._trades_history = []
            else:
                self._trades_history = [
                    {
                        'ticket': deal.ticket,
                        'symbol': deal.symbol,
                        'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                        'volume': deal.volume,
                        'price': deal.price,
                        'profit': deal.profit,
                        'time': pd.to_datetime(deal.time, unit='s')
                    }
                    for deal in history
                ]
            
            self.app_state.set('trades', len(self._trades_history))
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'historique: {e}")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Retourne les positions actuelles"""
        return self._positions
    
    def get_trades_history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique des trades"""
        return self._trades_history 