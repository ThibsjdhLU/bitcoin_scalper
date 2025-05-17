"""
MetaTrader 5 connector module.
Handles all interactions with the MT5 terminal.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

# Définition des constantes MT5
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

class TimeFrame(Enum):
    """Timeframe enumeration for MT5."""
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1

class MT5Connector:
    """
    MetaTrader 5 connector class.
    Handles connection and data retrieval from MT5 terminal.
    """
    
    def __init__(self, 
                 login: int = None,
                 password: str = None,
                 server: str = None,
                 path: str = None):
        """
        Initialize MT5 connector.
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            path: Path to MT5 terminal executable
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.logger = logging.getLogger(__name__)
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize(path=self.path):
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if all([self.login, self.password, self.server]):
                if not mt5.login(login=self.login,
                               password=self.password,
                               server=self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            self.connected = True
            self.logger.info("Successfully connected to MT5")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5 terminal."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return {}
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return {}
            
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'currency': account_info.currency
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.
        
        Args:
            symbol: Symbol name
            
        Returns:
            dict: Symbol information
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return {}
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info: {mt5.last_error()}")
                return {}
            
            return {
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_contract_size': symbol_info.trade_contract_size,
                'point': symbol_info.point
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return {}
    
    def get_rates(self, 
                  symbol: str,
                  timeframe: TimeFrame,
                  start_pos: int = 0,
                  count: int = 1000) -> pd.DataFrame:
        """
        Get historical rates for a symbol.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe from TimeFrame enum
            start_pos: Start position
            count: Number of bars to get
            
        Returns:
            pd.DataFrame: Historical rates
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return pd.DataFrame()
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, start_pos, count)
            if rates is None:
                self.logger.error(f"Failed to get rates: {mt5.last_error()}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates: {e}")
            return pd.DataFrame()
    
    def place_order(self,
                   symbol: str,
                   order_type: int,
                   volume: float,
                   price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   comment: str = "") -> int:
        """
        Place a trading order.
        
        Args:
            symbol: Symbol name
            order_type: Order type (mt5.ORDER_TYPE_*)
            volume: Order volume
            price: Order price (for pending orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            int: Order ticket number if successful, 0 otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return 0
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "comment": comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if price is not None:
                request["price"] = price
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return 0
            
            return result.order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return 0
    
    def close_position(self, ticket: int) -> bool:
        """
        Close an open position.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return False
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None:
                self.logger.error(f"Position {ticket} not found")
                return False
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position[0].symbol,
                "volume": position[0].volume,
                "type": mt5.ORDER_TYPE_SELL if position[0].type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to close position: {result.comment}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def modify_position(self,
                       ticket: int,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """
        Modify an open position.
        
        Args:
            ticket: Position ticket number
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return False
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None:
                self.logger.error(f"Position {ticket} not found")
                return False
            
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "symbol": position[0].symbol,
                "position": ticket,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to modify position: {result.comment}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return False 