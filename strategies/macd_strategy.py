"""
Stratégie de trading basée sur le Moving Average Convergence Divergence (MACD).
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from core.data_fetcher import DataFetcher, TimeFrame
from core.order_executor import OrderExecutor, OrderSide
from .base_strategy import BaseStrategy, Signal, SignalType
from utils.indicators import calculate_macd, calculate_ema, calculate_atr

class MACDStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le MACD.
    
    La stratégie utilise les conditions suivantes :
    - Signal d'achat :
        * Croisement haussier du MACD avec sa ligne de signal
        * Histogramme devient positif
        * Divergence haussière (prix fait des plus bas mais MACD fait des plus hauts)
    
    - Signal de vente :
        * Croisement baissier du MACD avec sa ligne de signal
        * Histogramme devient négatif
        * Divergence baissière (prix fait des plus hauts mais MACD fait des plus bas)
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        order_executor: OrderExecutor,
        symbols: List[str],
        timeframe: TimeFrame,
        params: Optional[Dict] = None
    ):
        """
        Initialise la stratégie MACD.
        
        Args:
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            params: Paramètres de la stratégie (optionnel)
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'trend_ema_period': 200,
            'min_histogram_change': 0.0001,
            'divergence_lookback': 10,
            'atr_period': 14,
            'take_profit_atr_multiplier': 2.0
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(
            name="MACD Strategy",
            description="Stratégie basée sur le MACD avec détection de divergences",
            data_fetcher=data_fetcher,
            order_executor=order_executor,
            params=default_params,
            symbols=symbols,
            timeframe=timeframe
        )
        
    def _validate_params(self) -> None:
        """
        Vérifie que les paramètres sont valides.
        
        Raises:
            ValueError: Si les paramètres sont invalides
        """
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError(
                "La période rapide doit être inférieure à la période lente"
            )
            
        if self.params['signal_period'] >= self.params['slow_period']:
            raise ValueError(
                "La période du signal doit être inférieure à la période lente"
            )
            
        if self.params['min_histogram_change'] <= 0:
            raise ValueError(
                "Le changement minimum de l'histogramme doit être positif"
            )
            
        if self.params['divergence_lookback'] < 5:
            raise ValueError(
                "Le lookback pour les divergences doit être >= 5"
            )
            
        if self.params['atr_period'] < 1:
            raise ValueError(
                "La période de l'ATR doit être >= 1"
            )
            
        if self.params['take_profit_atr_multiplier'] <= 0:
            raise ValueError(
                "Le multiplicateur ATR pour le take profit doit être > 0"
            )
            
    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.
        
        Returns:
            int: Nombre de bougies requises
        """
        return max(
            self.params['slow_period'],
            self.params['trend_ema_period'],
            self.params['divergence_lookback'],
            self.params['atr_period']
        ) * 2
        
    def should_enter(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Tuple[bool, Optional[Signal]]:
        """
        Détermine si on doit entrer en position.
        
        Args:
            symbol: Symbole à analyser
            data: Données OHLCV
            
        Returns:
            Tuple[bool, Optional[Signal]]: (Doit entrer, Signal)
        """
        buy_signals, sell_signals = self.calculate_signals(data)
        
        if buy_signals.iloc[-1]:
            metadata = self.generate_trade_metadata(data, -1, 'buy')
            signal = Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                strength=metadata['signal_strength'],
                metadata=metadata
            )
            return True, signal
            
        elif sell_signals.iloc[-1]:
            metadata = self.generate_trade_metadata(data, -1, 'sell')
            signal = Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                strength=metadata['signal_strength'],
                metadata=metadata
            )
            return True, signal
            
        return False, None
        
    def should_exit(
        self,
        symbol: str,
        data: pd.DataFrame,
        position_side: OrderSide
    ) -> Tuple[bool, Optional[Signal]]:
        """
        Détermine si on doit sortir d'une position.
        
        Args:
            symbol: Symbole à analyser
            data: Données OHLCV
            position_side: Sens de la position actuelle
            
        Returns:
            Tuple[bool, Optional[Signal]]: (Doit sortir, Signal)
        """
        buy_signals, sell_signals = self.calculate_signals(data)
        
        if position_side == OrderSide.BUY and sell_signals.iloc[-1]:
            metadata = self.generate_trade_metadata(data, -1, 'sell')
            signal = Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                strength=metadata['signal_strength'],
                metadata=metadata
            )
            return True, signal
            
        elif position_side == OrderSide.SELL and buy_signals.iloc[-1]:
            metadata = self.generate_trade_metadata(data, -1, 'buy')
            signal = Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                strength=metadata['signal_strength'],
                metadata=metadata
            )
            return True, signal
            
        return False, None
        
    def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> List[Signal]:
        """
        Génère les signaux de trading pour un symbole.
        
        Args:
            symbol: Symbole à analyser
            data: Données OHLCV
            
        Returns:
            List[Signal]: Liste des signaux générés
        """
        signals = []
        buy_signals, sell_signals = self.calculate_signals(data)
        
        # Générer les signaux d'achat
        for i in range(len(data)):
            if buy_signals.iloc[i]:
                metadata = self.generate_trade_metadata(data, i, 'buy')
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=data.index[i],
                    price=data['close'].iloc[i],
                    strength=metadata['signal_strength'],
                    metadata=metadata
                )
                signals.append(signal)
                
            elif sell_signals.iloc[i]:
                metadata = self.generate_trade_metadata(data, i, 'sell')
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=data.index[i],
                    price=data['close'].iloc[i],
                    strength=metadata['signal_strength'],
                    metadata=metadata
                )
                signals.append(signal)
                
        return signals
    
    def _detect_divergence(
        self,
        price: pd.Series,
        macd: pd.Series,
        lookback: int
    ) -> Tuple[bool, bool]:
        """
        Détecte les divergences entre le prix et le MACD.
        
        Args:
            price: Série des prix
            macd: Série du MACD
            lookback: Nombre de bougies à analyser
            
        Returns:
            Tuple[bool, bool]: (Divergence haussière, Divergence baissière)
        """
        # Extraire les segments à analyser
        price_segment = price[-lookback:]
        macd_segment = macd[-lookback:]
        
        # Trouver les extremums
        price_min = price_segment.min()
        price_max = price_segment.max()
        macd_min = macd_segment.min()
        macd_max = macd_segment.max()
        
        # Calculer les tendances
        price_trend = price_segment.iloc[-1] - price_segment.iloc[0]
        macd_trend = macd_segment.iloc[-1] - macd_segment.iloc[0]
        
        # Détecter les divergences
        bullish_div = price_trend < 0 and macd_trend > 0  # Prix baisse mais MACD monte
        bearish_div = price_trend > 0 and macd_trend < 0  # Prix monte mais MACD baisse
        
        return bullish_div, bearish_div
    
    def calculate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule les signaux d'achat et de vente.
        
        Args:
            data: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Tuple[pd.Series, pd.Series]: Signaux d'achat et de vente
        """
        # Calcul des indicateurs
        macd, signal, histogram = calculate_macd(
            data['close'],
            fast_period=self.params['fast_period'],
            slow_period=self.params['slow_period'],
            signal_period=self.params['signal_period']
        )
        
        trend_ema = calculate_ema(data['close'], self.params['trend_ema_period'])
        
        # Détecter les croisements
        crossover = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        crossunder = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        
        # Détecter les changements d'histogramme
        histogram_bullish = (
            (histogram > 0) &
            ((histogram - histogram.shift(1)) > self.params['min_histogram_change'])
        )
        histogram_bearish = (
            (histogram < 0) &
            ((histogram - histogram.shift(1)) < -self.params['min_histogram_change'])
        )
        
        # Détecter les divergences
        bullish_divs = pd.Series(False, index=data.index)
        bearish_divs = pd.Series(False, index=data.index)
        
        for i in range(self.params['divergence_lookback'], len(data)):
            bull_div, bear_div = self._detect_divergence(
                data['close'].iloc[i-self.params['divergence_lookback']:i+1],
                macd.iloc[i-self.params['divergence_lookback']:i+1],
                self.params['divergence_lookback']
            )
            bullish_divs.iloc[i] = bull_div
            bearish_divs.iloc[i] = bear_div
        
        # Combiner les signaux
        buy_signals = (
            (crossover | histogram_bullish | bullish_divs) &
            (data['close'] > trend_ema)  # Filtre de tendance
        )
        
        sell_signals = (
            (crossunder | histogram_bearish | bearish_divs) &
            (data['close'] < trend_ema)  # Filtre de tendance
        )
        
        return buy_signals, sell_signals
    
    def generate_trade_metadata(
        self,
        data: pd.DataFrame,
        index: int,
        signal_type: str
    ) -> Dict:
        """
        Génère les métadonnées pour un signal de trading.
        
        Args:
            data: DataFrame avec les données de marché
            index: Index du signal dans le DataFrame
            signal_type: Type de signal ('buy' ou 'sell')
            
        Returns:
            Dict: Métadonnées du trade
        """
        macd, signal, histogram = calculate_macd(
            data['close'],
            fast_period=self.params['fast_period'],
            slow_period=self.params['slow_period'],
            signal_period=self.params['signal_period']
        )
        
        trend_ema = calculate_ema(data['close'], self.params['trend_ema_period'])
        
        current_price = data['close'].iloc[index]
        current_macd = macd.iloc[index]
        current_signal = signal.iloc[index]
        current_histogram = histogram.iloc[index]
        
        # Calculer la force du signal
        if signal_type == 'buy':
            signal_strength = (
                (current_macd - current_signal) / abs(current_signal)
                if current_signal != 0 else 0
            )
        else:  # signal_type == 'sell'
            signal_strength = (
                (current_signal - current_macd) / abs(current_signal)
                if current_signal != 0 else 0
            )
            
        return {
            'signal_type': signal_type,
            'price': current_price,
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'signal_strength': signal_strength,
            'trend_ema': trend_ema.iloc[index]
        }
    
    def calculate_stop_loss(
        self,
        data: pd.DataFrame,
        index: int,
        signal_type: str
    ) -> float:
        """
        Calcule le niveau de stop loss pour un trade.
        
        Args:
            data: DataFrame avec les données de marché
            index: Index du signal dans le DataFrame
            signal_type: Type de signal ('buy' ou 'sell')
            
        Returns:
            float: Niveau de stop loss
        """
        trend_ema = calculate_ema(data['close'], self.params['trend_ema_period'])
        current_price = data['close'].iloc[index]
        
        if signal_type == 'buy':
            # Stop loss sous le plus bas récent ou sous l'EMA
            stop_level = min(
                data['low'].iloc[max(0, index-self.params['slow_period']):index+1].min(),
                trend_ema.iloc[index]
            )
        else:  # signal_type == 'sell'
            # Stop loss au-dessus du plus haut récent ou au-dessus de l'EMA
            stop_level = max(
                data['high'].iloc[max(0, index-self.params['slow_period']):index+1].max(),
                trend_ema.iloc[index]
            )
            
        return stop_level
    
    def calculate_take_profit(
        self,
        data: pd.DataFrame,
        index: int,
        signal_type: str
    ) -> float:
        """
        Calcule le niveau de take profit pour un trade.
        
        Args:
            data: DataFrame avec les données de marché
            index: Index du signal dans le DataFrame
            signal_type: Type de signal ('buy' ou 'sell')
            
        Returns:
            float: Niveau de take profit
        """
        current_price = data['close'].iloc[index]
        atr = calculate_atr(
            data['high'],
            data['low'],
            data['close'],
            self.params['atr_period']
        )
        current_atr = atr.iloc[index]
        
        if signal_type == 'buy':
            # Take profit à N ATR au-dessus du prix d'entrée
            take_profit = current_price + (self.params['take_profit_atr_multiplier'] * current_atr)
        else:  # signal_type == 'sell'
            # Take profit à N ATR en-dessous du prix d'entrée
            take_profit = current_price - (self.params['take_profit_atr_multiplier'] * current_atr)
            
        return float(take_profit)  # Convertir en float pour éviter les NaN 