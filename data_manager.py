import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

class DataManager:
    """
    Gestionnaire des données de marché
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de données
        
        Args:
            config (dict): Configuration du gestionnaire
        """
        self.config = config
        
        # Paramètres de données
        self.symbols = config.get('symbols', ['BTC/USDT'])
        self.timeframe = config.get('timeframe', '1m')
        self.max_history = config.get('max_history', 1000)  # Nombre max de candles
        self.update_interval = config.get('update_interval', 1)  # Intervalle de mise à jour en secondes
        
        # Stockage des données
        self.data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Initialisation des DataFrames
        for symbol in self.symbols:
            self.data[symbol] = pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            self.last_update[symbol] = datetime.min
        
        logger.info("Gestionnaire de données initialisé")
    
    def update_data(self, symbol: str, new_data: List[MarketData]) -> None:
        """
        Met à jour les données pour un symbole
        
        Args:
            symbol (str): Symbole de trading
            new_data (List[MarketData]): Nouvelles données
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbole non supporté: {symbol}")
            return
        
        # Conversion en DataFrame
        df_new = pd.DataFrame([vars(d) for d in new_data])
        
        # Fusion avec les données existantes
        self.data[symbol] = pd.concat([self.data[symbol], df_new])
        
        # Suppression des doublons
        self.data[symbol] = self.data[symbol].drop_duplicates(subset=['timestamp'])
        
        # Tri par timestamp
        self.data[symbol] = self.data[symbol].sort_values('timestamp')
        
        # Limitation de l'historique
        if len(self.data[symbol]) > self.max_history:
            self.data[symbol] = self.data[symbol].tail(self.max_history)
        
        # Mise à jour du timestamp
        self.last_update[symbol] = datetime.now()
        
        logger.info(f"Données mises à jour pour {symbol}: {len(new_data)} nouvelles candles")
    
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Récupère les n dernières candles pour un symbole
        
        Args:
            symbol (str): Symbole de trading
            n (int): Nombre de candles à récupérer
            
        Returns:
            pd.DataFrame: Dernières candles
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbole non supporté: {symbol}")
            return pd.DataFrame()
        
        return self.data[symbol].tail(n)
    
    def get_data_range(self, symbol: str, start_time: datetime,
                      end_time: datetime) -> pd.DataFrame:
        """
        Récupère les données dans une plage de temps
        
        Args:
            symbol (str): Symbole de trading
            start_time (datetime): Début de la plage
            end_time (datetime): Fin de la plage
            
        Returns:
            pd.DataFrame: Données dans la plage
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbole non supporté: {symbol}")
            return pd.DataFrame()
        
        mask = (self.data[symbol]['timestamp'] >= start_time) & \
               (self.data[symbol]['timestamp'] <= end_time)
        return self.data[symbol][mask]
    
    def calculate_indicators(self, symbol: str, indicators: List[str]) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques demandés
        
        Args:
            symbol (str): Symbole de trading
            indicators (List[str]): Liste des indicateurs à calculer
            
        Returns:
            pd.DataFrame: DataFrame avec les indicateurs
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbole non supporté: {symbol}")
            return pd.DataFrame()
        
        df = self.data[symbol].copy()
        
        for indicator in indicators:
            if indicator == 'SMA':
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['SMA_50'] = df['close'].rolling(window=50).mean()
                df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            elif indicator == 'EMA':
                df['EMA_12'] = df['close'].ewm(span=12).mean()
                df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            elif indicator == 'RSI':
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            elif indicator == 'MACD':
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            elif indicator == 'BB':
                df['BB_Middle'] = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (std * 2)
            
            elif indicator == 'ATR':
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def get_market_state(self, symbol: str) -> Dict:
        """
        Récupère l'état actuel du marché
        
        Args:
            symbol (str): Symbole de trading
            
        Returns:
            Dict: État du marché
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbole non supporté: {symbol}")
            return {}
        
        latest = self.get_latest_data(symbol, 1)
        if latest.empty:
            return {}
        
        # Calcul des variations
        prev_close = self.data[symbol]['close'].iloc[-2] if len(self.data[symbol]) > 1 else latest['close'].iloc[0]
        price_change = latest['close'].iloc[0] - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        # Volume moyen
        avg_volume = self.data[symbol]['volume'].rolling(window=20).mean().iloc[-1]
        volume_change = (latest['volume'].iloc[0] / avg_volume - 1) * 100
        
        return {
            'symbol': symbol,
            'timestamp': latest['timestamp'].iloc[0],
            'price': latest['close'].iloc[0],
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': latest['volume'].iloc[0],
            'volume_change_pct': volume_change,
            'high': latest['high'].iloc[0],
            'low': latest['low'].iloc[0],
            'volatility': self._calculate_volatility(symbol)
        }
    
    def _calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """
        Calcule la volatilité sur une fenêtre
        
        Args:
            symbol (str): Symbole de trading
            window (int): Fenêtre de calcul
            
        Returns:
            float: Volatilité calculée
        """
        if symbol not in self.symbols:
            return 0.0
        
        returns = self.data[symbol]['close'].pct_change()
        return returns.rolling(window=window).std().iloc[-1] * 100  # En pourcentage
    
    def needs_update(self, symbol: str) -> bool:
        """
        Vérifie si les données doivent être mises à jour
        
        Args:
            symbol (str): Symbole de trading
            
        Returns:
            bool: True si mise à jour nécessaire
        """
        if symbol not in self.symbols:
            return False
        
        time_since_update = (datetime.now() - self.last_update[symbol]).total_seconds()
        return time_since_update >= self.update_interval 