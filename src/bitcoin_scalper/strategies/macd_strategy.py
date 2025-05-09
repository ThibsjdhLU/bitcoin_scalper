"""
Stratégie de trading basée sur le Moving Average Convergence Divergence (MACD).
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..core.data_fetcher import DataFetcher, TimeFrame
from ..core.order_executor import OrderExecutor, OrderSide
from ..utils.indicators import calculate_atr, calculate_ema, calculate_macd
from ..utils.logger import format_boolean, get_logger

from .base_strategy import BaseStrategy, Signal, SignalType


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
        params: Optional[Dict] = None,
        is_optimizing: bool = False,
    ):
        """
        Initialise la stratégie MACD.

        Args:
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            params: Paramètres de la stratégie (optionnel)
            is_optimizing: Si True, désactive les logs pendant l'optimisation
        """
        default_params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "trend_ema_period": 200,
            "min_histogram_change": 0.0001,
            "divergence_lookback": 10,
            "atr_period": 14,
            "take_profit_atr_multiplier": 2.0,
            "analysis_delay": 1.0,  # Délai en secondes entre chaque analyse
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
            timeframe=timeframe,
        )

        self.is_optimizing = is_optimizing
        self._initialized = False
        self._last_check_time = 0
        self._last_signal_time = 0

    def _validate_params(self) -> None:
        """
        Vérifie que les paramètres sont valides.

        Raises:
            ValueError: Si les paramètres sont invalides
        """
        if self.params["fast_period"] >= self.params["slow_period"]:
            raise ValueError(
                "La période rapide doit être inférieure à la période lente"
            )

        if self.params["signal_period"] >= self.params["slow_period"]:
            raise ValueError(
                "La période du signal doit être inférieure à la période lente"
            )

        if self.params["min_histogram_change"] <= 0:
            raise ValueError("Le changement minimum de l'histogramme doit être positif")

        if self.params["divergence_lookback"] < 5:
            raise ValueError("Le lookback pour les divergences doit être >= 5")

        if self.params["atr_period"] < 1:
            raise ValueError("La période de l'ATR doit être >= 1")

        if self.params["take_profit_atr_multiplier"] <= 0:
            raise ValueError("Le multiplicateur ATR pour le take profit doit être > 0")

    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.

        Returns:
            int: Nombre de bougies requises
        """
        return (
            max(
                self.params["slow_period"],
                self.params["trend_ema_period"],
                self.params["divergence_lookback"],
                self.params["atr_period"],
            )
            * 2
        )

    def should_enter(
        self, symbol: str, data: pd.DataFrame
    ) -> Tuple[bool, Optional[Signal]]:
        """
        Détermine si on doit entrer en position.

        Args:
            symbol: Symbole à analyser
            data: Données OHLCV

        Returns:
            Tuple[bool, Optional[Signal]]: (Doit entrer, Signal)
        """
        try:
            buy_signals, sell_signals = self.calculate_signals(data)

            # Vérifier si nous avons un signal d'achat ou de vente
            if buy_signals.iloc[-1]:
                metadata = self.generate_trade_metadata(data, -1, "buy")
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=data["close"].iloc[-1],
                    strength=metadata["signal_strength"],
                    metadata=metadata,
                )
                if not self.is_optimizing:
                    logger.info(
                        f"Ordre ACHAT {symbol} - Prix: {data['close'].iloc[-1]:.2f}, Force: {metadata['signal_strength']:.2f}"
                    )
                return True, signal

            elif sell_signals.iloc[-1]:
                metadata = self.generate_trade_metadata(data, -1, "sell")
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=data["close"].iloc[-1],
                    strength=metadata["signal_strength"],
                    metadata=metadata,
                )
                if not self.is_optimizing:
                    logger.info(
                        f"Ordre VENTE {symbol} - Prix: {data['close'].iloc[-1]:.2f}, Force: {metadata['signal_strength']:.2f}"
                    )
                return True, signal

            return False, None

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification des signaux d'entrée: {str(e)}"
            )
            return False, None

    def should_exit(
        self, symbol: str, data: pd.DataFrame, position_side: OrderSide
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
            metadata = self.generate_trade_metadata(data, -1, "sell")
            signal = Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data["close"].iloc[-1],
                strength=metadata["signal_strength"],
                metadata=metadata,
            )
            return True, signal

        elif position_side == OrderSide.SELL and buy_signals.iloc[-1]:
            metadata = self.generate_trade_metadata(data, -1, "buy")
            signal = Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(),
                price=data["close"].iloc[-1],
                strength=metadata["signal_strength"],
                metadata=metadata,
            )
            return True, signal

        return False, None

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> pd.Series:
        """
        Génère les signaux de trading pour un symbole donné.

        Args:
            symbol: Symbole à analyser
            data: Données OHLCV

        Returns:
            pd.Series: Série de signaux (1 pour achat, -1 pour vente, 0 sinon)
        """
        signals = pd.Series(0, index=data.index)
        buy_signals, sell_signals = self.calculate_signals(data)

        # Convertir les signaux booléens en entiers de manière sûre
        signals = signals.astype("int64")
        buy_signals = buy_signals.astype("int64")
        sell_signals = sell_signals.astype("int64")

        # Générer les signaux
        signals[buy_signals == 1] = 1
        signals[sell_signals == 1] = -1

        return signals

    def _detect_divergence(
        self, price: pd.Series, macd: pd.Series, lookback: int
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
        """Calcule les signaux d'achat et de vente."""
        try:
            # Vérifier le délai d'analyse de manière non-bloquante
            current_time = time.time()
            if hasattr(self, "_last_check_time"):
                delay = self.params.get("analysis_delay", 1.0)
                elapsed = current_time - self._last_check_time
                if elapsed < delay:
                    # Au lieu de bloquer avec time.sleep, on retourne des signaux neutres
                    return pd.Series(False, index=data.index), pd.Series(
                        False, index=data.index
                    )
            self._last_check_time = current_time

            # Calculer le MACD
            macd, signal, hist = calculate_macd(
                data["close"],
                self.params["fast_period"],
                self.params["slow_period"],
                self.params["signal_period"],
            )

            # Calculer l'EMA de tendance
            trend_ema = calculate_ema(data["close"], self.params["trend_ema_period"])

            # Détecter les divergences
            bullish_div, bearish_div = self._detect_divergence(
                data["close"], macd, self.params["divergence_lookback"]
            )

            # Conditions d'achat
            crossover = pd.Series(
                (macd > signal) & (macd.shift(1) <= signal.shift(1)), index=data.index
            )
            histogram_bullish = pd.Series(
                (hist > 0) & (hist > hist.shift(1)), index=data.index
            )
            price_above_ema = pd.Series(data["close"] > trend_ema, index=data.index)

            # Conditions de vente
            crossunder = pd.Series(
                (macd < signal) & (macd.shift(1) >= signal.shift(1)), index=data.index
            )
            histogram_bearish = pd.Series(
                (hist < 0) & (hist < hist.shift(1)), index=data.index
            )
            price_below_ema = pd.Series(data["close"] < trend_ema, index=data.index)

            # Générer les signaux
            buy_signals = pd.Series(
                crossover
                | (histogram_bullish & price_above_ema)
                | (bullish_div & price_above_ema),
                index=data.index,
            )
            sell_signals = pd.Series(
                crossunder
                | (histogram_bearish & price_below_ema)
                | (bearish_div & price_below_ema),
                index=data.index,
            )

            # Log uniquement si pas en optimisation et s'il y a un signal
            if not self.is_optimizing and (
                buy_signals.iloc[-1] or sell_signals.iloc[-1]
            ):
                logger.info(
                    f"Analyse MACD - Prix: {data['close'].iloc[-1]:.2f}, MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}"
                )

                if buy_signals.iloc[-1]:
                    logger.info(
                        f"Signal ACHAT - Conditions: Croisement haussier={crossover.iloc[-1]}, Histogramme haussier={histogram_bullish.iloc[-1]}, Divergence haussière={bullish_div}, Prix > EMA={price_above_ema.iloc[-1]}"
                    )
                elif sell_signals.iloc[-1]:
                    logger.info(
                        f"Signal VENTE - Conditions: Croisement baissier={crossunder.iloc[-1]}, Histogramme baissier={histogram_bearish.iloc[-1]}, Divergence baissière={bearish_div}, Prix < EMA={price_below_ema.iloc[-1]}"
                    )

            return buy_signals, sell_signals

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {data.index.name}: {str(e)}")
            return pd.Series(False, index=data.index), pd.Series(
                False, index=data.index
            )

    def generate_trade_metadata(
        self, data: pd.DataFrame, index: int, signal_type: str
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
        # S'assurer que les données sont des Series pandas
        close_prices = pd.Series(data["close"].values, index=data.index)

        macd, signal, histogram = calculate_macd(
            close_prices,
            fast_period=self.params["fast_period"],
            slow_period=self.params["slow_period"],
            signal_period=self.params["signal_period"],
        )

        trend_ema = calculate_ema(close_prices, self.params["trend_ema_period"])

        current_price = data["close"].iloc[index]
        current_macd = macd.iloc[index]
        current_signal = signal.iloc[index]
        current_histogram = histogram.iloc[index]

        # Calculer la force du signal
        if signal_type == "buy":
            signal_strength = (
                (current_macd - current_signal) / abs(current_signal)
                if current_signal != 0
                else 0
            )
        else:  # signal_type == 'sell'
            signal_strength = (
                (current_signal - current_macd) / abs(current_signal)
                if current_signal != 0
                else 0
            )

        return {
            "signal_type": signal_type,
            "price": current_price,
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_histogram,
            "signal_strength": signal_strength,
            "trend_ema": trend_ema.iloc[index],
        }

    def calculate_stop_loss(
        self, data: pd.DataFrame, index: int, signal_type: str
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
        current_price = data["close"].iloc[index]
        atr = calculate_atr(
            data["high"], data["low"], data["close"], self.params["atr_period"]
        ).iloc[index]

        if signal_type == "buy":
            # Stop loss à 1 ATR sous le prix actuel
            stop_level = current_price - atr
        else:  # signal_type == 'sell'
            # Stop loss à 1 ATR au-dessus du prix actuel
            stop_level = current_price + atr

        return float(stop_level)

    def calculate_take_profit(
        self, data: pd.DataFrame, index: int, signal_type: str
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
        current_price = data["close"].iloc[index]
        atr = calculate_atr(
            data["high"], data["low"], data["close"], self.params["atr_period"]
        ).iloc[index]

        if signal_type == "buy":
            # Take profit à 2 ATR au-dessus du prix actuel
            take_profit = current_price + (2 * atr)
        else:  # signal_type == 'sell'
            # Take profit à 2 ATR en-dessous du prix actuel
            take_profit = current_price - (2 * atr)

        return float(take_profit)  # Convertir en float pour éviter les NaN

    def _generate_signals_impl(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Implémente la logique de la stratégie MACD.

        Args:
            data: DataFrame avec les données OHLCV
            signals: Série des signaux à modifier
        """
        try:
            # Calculer le MACD
            macd, signal_line, histogram = calculate_macd(
                data["close"],
                self.params["fast_period"],
                self.params["slow_period"],
                self.params["signal_period"],
            )

            # Calculer la tendance avec l'EMA
            trend_ema = calculate_ema(data["close"], self.params["trend_ema_period"])

            # Calculer la variation de l'histogramme
            hist_change = histogram - histogram.shift(1)

            # Générer les signaux
            # Signal d'achat: MACD croise au-dessus de la ligne de signal
            # et la tendance est haussière
            buy_signal = pd.Series(
                (macd > signal_line)
                & (macd.shift(1) <= signal_line.shift(1))
                & (data["close"] > trend_ema)
                & (hist_change > self.params["min_histogram_change"]),
                index=data.index,
            )

            # Signal de vente: MACD croise en-dessous de la ligne de signal
            # et la tendance est baissière
            sell_signal = pd.Series(
                (macd < signal_line)
                & (macd.shift(1) >= signal_line.shift(1))
                & (data["close"] < trend_ema)
                & (abs(hist_change) > self.params["min_histogram_change"]),
                index=data.index,
            )

            # Mettre à jour les signaux
            signals[buy_signal] = 1
            signals[sell_signal] = -1

        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux MACD: {str(e)}")
