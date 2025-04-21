import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolume

class TechnicalIndicators:
    """
    Gestionnaire d'indicateurs techniques
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_moving_averages(self,
                                data: pd.DataFrame,
                                windows: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calcule les moyennes mobiles
        
        Args:
            data: DataFrame avec les données OHLCV
            windows: Liste des périodes
            
        Returns:
            Dict: Dictionnaire des moyennes mobiles
        """
        try:
            ma_dict = {}
            
            for window in windows:
                # SMA
                sma = SMAIndicator(close=data['Close'], window=window)
                ma_dict[f'SMA_{window}'] = sma.sma_indicator()
                
                # EMA
                ema = EMAIndicator(close=data['Close'], window=window)
                ma_dict[f'EMA_{window}'] = ema.ema_indicator()
                
            return ma_dict
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des moyennes mobiles: {str(e)}")
            raise
            
    def calculate_macd(self,
                      data: pd.DataFrame,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calcule le MACD
        
        Args:
            data: DataFrame avec les données OHLCV
            fast_period: Période rapide
            slow_period: Période lente
            signal_period: Période du signal
            
        Returns:
            Dict: Composantes du MACD
        """
        try:
            macd = MACD(
                close=data['Close'],
                window_slow=slow_period,
                window_fast=fast_period,
                window_sign=signal_period
            )
            
            return {
                'macd': macd.macd(),
                'signal': macd.macd_signal(),
                'histogram': macd.macd_diff()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du MACD: {str(e)}")
            raise
            
    def calculate_rsi(self,
                     data: pd.DataFrame,
                     period: int = 14) -> pd.Series:
        """
        Calcule le RSI
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période du RSI
            
        Returns:
            Series: Valeurs du RSI
        """
        try:
            rsi = RSIIndicator(close=data['Close'], window=period)
            return rsi.rsi()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du RSI: {str(e)}")
            raise
            
    def calculate_stochastic(self,
                           data: pd.DataFrame,
                           k_period: int = 14,
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calcule l'oscillateur stochastique
        
        Args:
            data: DataFrame avec les données OHLCV
            k_period: Période de %K
            d_period: Période de %D
            
        Returns:
            Dict: Composantes du stochastique
        """
        try:
            stoch = StochasticOscillator(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=k_period,
                smooth_window=d_period
            )
            
            return {
                'k': stoch.stoch(),
                'd': stoch.stoch_signal()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du stochastique: {str(e)}")
            raise
            
    def calculate_bollinger_bands(self,
                                data: pd.DataFrame,
                                period: int = 20,
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calcule les bandes de Bollinger
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période
            std_dev: Nombre d'écarts-types
            
        Returns:
            Dict: Composantes des bandes de Bollinger
        """
        try:
            bb = BollingerBands(
                close=data['Close'],
                window=period,
                window_dev=std_dev
            )
            
            return {
                'upper': bb.bollinger_hband(),
                'middle': bb.bollinger_mavg(),
                'lower': bb.bollinger_lband()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des bandes de Bollinger: {str(e)}")
            raise
            
    def calculate_atr(self,
                     data: pd.DataFrame,
                     period: int = 14) -> pd.Series:
        """
        Calcule l'ATR
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période de l'ATR
            
        Returns:
            Series: Valeurs de l'ATR
        """
        try:
            atr = AverageTrueRange(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=period
            )
            return atr.average_true_range()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de l'ATR: {str(e)}")
            raise
            
    def calculate_vwap(self,
                      data: pd.DataFrame,
                      period: int = 14) -> pd.Series:
        """
        Calcule le VWAP
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période du VWAP
            
        Returns:
            Series: Valeurs du VWAP
        """
        try:
            vwap = VolumeWeightedAveragePrice(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                volume=data['Volume'],
                window=period
            )
            return vwap.volume_weighted_average_price()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du VWAP: {str(e)}")
            raise
            
    def calculate_obv(self,
                     data: pd.DataFrame) -> pd.Series:
        """
        Calcule l'OBV
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Series: Valeurs de l'OBV
        """
        try:
            obv = OnBalanceVolume(
                close=data['Close'],
                volume=data['Volume']
            )
            return obv.on_balance_volume()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de l'OBV: {str(e)}")
            raise
            
    def calculate_all_indicators(self,
                               data: pd.DataFrame,
                               config: Dict = None) -> Dict[str, pd.Series]:
        """
        Calcule tous les indicateurs techniques
        
        Args:
            data: DataFrame avec les données OHLCV
            config: Configuration des indicateurs
            
        Returns:
            Dict: Tous les indicateurs calculés
        """
        try:
            if config is None:
                config = self.config
                
            indicators = {}
            
            # Moyennes mobiles
            ma_dict = self.calculate_moving_averages(
                data,
                windows=config.get('ma_windows', [20, 50, 200])
            )
            indicators.update(ma_dict)
            
            # MACD
            macd_dict = self.calculate_macd(
                data,
                fast_period=config.get('macd_fast', 12),
                slow_period=config.get('macd_slow', 26),
                signal_period=config.get('macd_signal', 9)
            )
            indicators.update(macd_dict)
            
            # RSI
            indicators['RSI'] = self.calculate_rsi(
                data,
                period=config.get('rsi_period', 14)
            )
            
            # Stochastique
            stoch_dict = self.calculate_stochastic(
                data,
                k_period=config.get('stoch_k', 14),
                d_period=config.get('stoch_d', 3)
            )
            indicators.update(stoch_dict)
            
            # Bandes de Bollinger
            bb_dict = self.calculate_bollinger_bands(
                data,
                period=config.get('bb_period', 20),
                std_dev=config.get('bb_std', 2.0)
            )
            indicators.update(bb_dict)
            
            # ATR
            indicators['ATR'] = self.calculate_atr(
                data,
                period=config.get('atr_period', 14)
            )
            
            # VWAP
            indicators['VWAP'] = self.calculate_vwap(
                data,
                period=config.get('vwap_period', 14)
            )
            
            # OBV
            indicators['OBV'] = self.calculate_obv(data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de tous les indicateurs: {str(e)}")
            raise 