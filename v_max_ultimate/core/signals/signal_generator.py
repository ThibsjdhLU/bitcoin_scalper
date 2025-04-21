import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from ..indicators.technical_indicators import TechnicalIndicators

class SignalGenerator:
    """
    Générateur de signaux de trading
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators(config)
        
    def generate_macd_signal(self,
                           data: pd.DataFrame,
                           fast_period: int = 12,
                           slow_period: int = 26,
                           signal_period: int = 9) -> pd.Series:
        """
        Génère un signal basé sur le MACD
        
        Args:
            data: DataFrame avec les données OHLCV
            fast_period: Période rapide
            slow_period: Période lente
            signal_period: Période du signal
            
        Returns:
            Series: Signal (1: achat, -1: vente, 0: neutre)
        """
        try:
            macd_dict = self.indicators.calculate_macd(
                data,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period
            )
            
            signal = pd.Series(0, index=data.index)
            
            # Signal d'achat : MACD croise la ligne de signal vers le haut
            signal[macd_dict['macd'] > macd_dict['signal']] = 1
            
            # Signal de vente : MACD croise la ligne de signal vers le bas
            signal[macd_dict['macd'] < macd_dict['signal']] = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal MACD: {str(e)}")
            raise
            
    def generate_rsi_signal(self,
                          data: pd.DataFrame,
                          period: int = 14,
                          overbought: float = 70,
                          oversold: float = 30) -> pd.Series:
        """
        Génère un signal basé sur le RSI
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période du RSI
            overbought: Niveau de surachat
            oversold: Niveau de survente
            
        Returns:
            Series: Signal (1: achat, -1: vente, 0: neutre)
        """
        try:
            rsi = self.indicators.calculate_rsi(data, period=period)
            
            signal = pd.Series(0, index=data.index)
            
            # Signal d'achat : RSI en zone de survente
            signal[rsi < oversold] = 1
            
            # Signal de vente : RSI en zone de surachat
            signal[rsi > overbought] = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal RSI: {str(e)}")
            raise
            
    def generate_stochastic_signal(self,
                                 data: pd.DataFrame,
                                 k_period: int = 14,
                                 d_period: int = 3,
                                 overbought: float = 80,
                                 oversold: float = 20) -> pd.Series:
        """
        Génère un signal basé sur le stochastique
        
        Args:
            data: DataFrame avec les données OHLCV
            k_period: Période de %K
            d_period: Période de %D
            overbought: Niveau de surachat
            oversold: Niveau de survente
            
        Returns:
            Series: Signal (1: achat, -1: vente, 0: neutre)
        """
        try:
            stoch_dict = self.indicators.calculate_stochastic(
                data,
                k_period=k_period,
                d_period=d_period
            )
            
            signal = pd.Series(0, index=data.index)
            
            # Signal d'achat : %K croise %D vers le haut en zone de survente
            signal[(stoch_dict['k'] > stoch_dict['d']) & (stoch_dict['k'] < oversold)] = 1
            
            # Signal de vente : %K croise %D vers le bas en zone de surachat
            signal[(stoch_dict['k'] < stoch_dict['d']) & (stoch_dict['k'] > overbought)] = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal stochastique: {str(e)}")
            raise
            
    def generate_bollinger_signal(self,
                                data: pd.DataFrame,
                                period: int = 20,
                                std_dev: float = 2.0) -> pd.Series:
        """
        Génère un signal basé sur les bandes de Bollinger
        
        Args:
            data: DataFrame avec les données OHLCV
            period: Période
            std_dev: Nombre d'écarts-types
            
        Returns:
            Series: Signal (1: achat, -1: vente, 0: neutre)
        """
        try:
            bb_dict = self.indicators.calculate_bollinger_bands(
                data,
                period=period,
                std_dev=std_dev
            )
            
            signal = pd.Series(0, index=data.index)
            
            # Signal d'achat : Prix touche la bande inférieure
            signal[data['Close'] <= bb_dict['lower']] = 1
            
            # Signal de vente : Prix touche la bande supérieure
            signal[data['Close'] >= bb_dict['upper']] = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal Bollinger: {str(e)}")
            raise
            
    def generate_ma_crossover_signal(self,
                                   data: pd.DataFrame,
                                   fast_period: int = 20,
                                   slow_period: int = 50) -> pd.Series:
        """
        Génère un signal basé sur le croisement de moyennes mobiles
        
        Args:
            data: DataFrame avec les données OHLCV
            fast_period: Période rapide
            slow_period: Période lente
            
        Returns:
            Series: Signal (1: achat, -1: vente, 0: neutre)
        """
        try:
            ma_dict = self.indicators.calculate_moving_averages(
                data,
                windows=[fast_period, slow_period]
            )
            
            signal = pd.Series(0, index=data.index)
            
            # Signal d'achat : MA rapide croise MA lente vers le haut
            signal[ma_dict[f'SMA_{fast_period}'] > ma_dict[f'SMA_{slow_period}']] = 1
            
            # Signal de vente : MA rapide croise MA lente vers le bas
            signal[ma_dict[f'SMA_{fast_period}'] < ma_dict[f'SMA_{slow_period}']] = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal MA: {str(e)}")
            raise
            
    def generate_combined_signal(self,
                               data: pd.DataFrame,
                               config: Dict = None) -> pd.Series:
        """
        Génère un signal combiné de tous les indicateurs
        
        Args:
            data: DataFrame avec les données OHLCV
            config: Configuration des signaux
            
        Returns:
            Series: Signal combiné (1: achat, -1: vente, 0: neutre)
        """
        try:
            if config is None:
                config = self.config
                
            # Générer les signaux individuels
            macd_signal = this.generate_macd_signal(
                data,
                fast_period=config.get('macd_fast', 12),
                slow_period=config.get('macd_slow', 26),
                signal_period=config.get('macd_signal', 9)
            )
            
            rsi_signal = this.generate_rsi_signal(
                data,
                period=config.get('rsi_period', 14),
                overbought=config.get('rsi_overbought', 70),
                oversold=config.get('rsi_oversold', 30)
            )
            
            stoch_signal = this.generate_stochastic_signal(
                data,
                k_period=config.get('stoch_k', 14),
                d_period=config.get('stoch_d', 3),
                overbought=config.get('stoch_overbought', 80),
                oversold=config.get('stoch_oversold', 20)
            )
            
            bb_signal = this.generate_bollinger_signal(
                data,
                period=config.get('bb_period', 20),
                std_dev=config.get('bb_std', 2.0)
            )
            
            ma_signal = this.generate_ma_crossover_signal(
                data,
                fast_period=config.get('ma_fast', 20),
                slow_period=config.get('ma_slow', 50)
            )
            
            # Combiner les signaux avec pondération
            combined_signal = pd.Series(0, index=data.index)
            
            weights = {
                'macd': config.get('macd_weight', 0.2),
                'rsi': config.get('rsi_weight', 0.2),
                'stoch': config.get('stoch_weight', 0.2),
                'bb': config.get('bb_weight', 0.2),
                'ma': config.get('ma_weight', 0.2)
            }
            
            combined_signal = (
                macd_signal * weights['macd'] +
                rsi_signal * weights['rsi'] +
                stoch_signal * weights['stoch'] +
                bb_signal * weights['bb'] +
                ma_signal * weights['ma']
            )
            
            # Seuil de décision
            threshold = config.get('signal_threshold', 0.5)
            
            final_signal = pd.Series(0, index=data.index)
            final_signal[combined_signal > threshold] = 1
            final_signal[combined_signal < -threshold] = -1
            
            return final_signal
            
        except Exception as e:
            this.logger.error(f"Erreur lors de la génération du signal combiné: {str(e)}")
            raise 