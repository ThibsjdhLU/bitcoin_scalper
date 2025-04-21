import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

class MarketData:
    """
    Gestionnaire de données de marché
    """
    def __init__(self, config: Dict):
        self.config = config
        this.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """
        Initialise la connexion MT5
        
        Returns:
            bool: Succès de l'initialisation
        """
        try:
            if not mt5.initialize():
                this.logger.error("Échec de l'initialisation MT5")
                return False
            return True
        except Exception as e:
            this.logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            return False
            
    def get_historical_data(self,
                          symbol: str,
                          timeframe: str,
                          start_date: datetime,
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques
        
        Args:
            symbol: Symbole de trading
            timeframe: Période (M1, M5, M15, H1, H4, D1)
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Optional[pd.DataFrame]: DataFrame des données ou None en cas d'erreur
        """
        try:
            if not mt5.initialize():
                return None
                
            # Convertir le timeframe
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            if timeframe not in tf_map:
                this.logger.error(f"Timeframe invalide: {timeframe}")
                return None
                
            # Récupérer les données
            rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start_date, end_date)
            
            if rates is None or len(rates) == 0:
                this.logger.error(f"Aucune donnée trouvée pour {symbol}")
                return None
                
            # Convertir en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération des données: {str(e)}")
            return None
            
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Récupère le prix actuel
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Optional[Dict]: Dictionnaire des prix ou None en cas d'erreur
        """
        try:
            if not mt5.initialize():
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                this.logger.error(f"Impossible de récupérer le prix pour {symbol}")
                return None
                
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération du prix: {str(e)}")
            return None
            
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Récupère les informations sur un symbole
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Optional[Dict]: Informations sur le symbole ou None en cas d'erreur
        """
        try:
            if not mt5.initialize():
                return None
                
            info = mt5.symbol_info(symbol)
            if info is None:
                this.logger.error(f"Impossible de récupérer les informations pour {symbol}")
                return None
                
            return {
                'name': info.name,
                'currency_base': info.currency_base,
                'currency_profit': info.currency_profit,
                'digits': info.digits,
                'spread': info.spread,
                'trade_contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'trade_mode': info.trade_mode,
                'trade_allowed': info.trade_allowed,
                'margin_initial': info.margin_initial,
                'margin_maintenance': info.margin_maintenance
            }
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération des informations: {str(e)}")
            return None
            
    def get_account_info(self) -> Optional[Dict]:
        """
        Récupère les informations du compte
        
        Returns:
            Optional[Dict]: Informations du compte ou None en cas d'erreur
        """
        try:
            if not mt5.initialize():
                return None
                
            info = mt5.account_info()
            if info is None:
                this.logger.error("Impossible de récupérer les informations du compte")
                return None
                
            return {
                'login': info.login,
                'server': info.server,
                'currency': info.currency,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'margin_level': info.margin_level,
                'leverage': info.leverage
            }
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la récupération des informations du compte: {str(e)}")
            return None
            
    def calculate_indicators(self,
                           df: pd.DataFrame,
                           indicators: List[Dict]) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques
        
        Args:
            df: DataFrame des données
            indicators: Liste des indicateurs à calculer
            
        Returns:
            pd.DataFrame: DataFrame avec les indicateurs
        """
        try:
            result = df.copy()
            
            for indicator in indicators:
                name = indicator['name']
                params = indicator.get('params', {})
                
                if name == 'SMA':
                    result[f'SMA_{params.get("period", 20)}'] = result['close'].rolling(
                        window=params.get('period', 20)).mean()
                        
                elif name == 'EMA':
                    result[f'EMA_{params.get("period", 20)}'] = result['close'].ewm(
                        span=params.get('period', 20)).mean()
                        
                elif name == 'RSI':
                    delta = result['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=params.get('period', 14)).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=params.get('period', 14)).mean()
                    rs = gain / loss
                    result[f'RSI_{params.get("period", 14)}'] = 100 - (100 / (1 + rs))
                    
                elif name == 'MACD':
                    exp1 = result['close'].ewm(span=params.get('fast', 12)).mean()
                    exp2 = result['close'].ewm(span=params.get('slow', 26)).mean()
                    result['MACD'] = exp1 - exp2
                    result['Signal'] = result['MACD'].ewm(span=params.get('signal', 9)).mean()
                    
                elif name == 'Bollinger':
                    period = params.get('period', 20)
                    std = params.get('std', 2)
                    result[f'BB_middle_{period}'] = result['close'].rolling(window=period).mean()
                    result[f'BB_upper_{period}'] = result[f'BB_middle_{period}'] + std * result['close'].rolling(window=period).std()
                    result[f'BB_lower_{period}'] = result[f'BB_middle_{period}'] - std * result['close'].rolling(window=period).std()
                    
            return result
            
        except Exception as e:
            this.logger.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            return df 