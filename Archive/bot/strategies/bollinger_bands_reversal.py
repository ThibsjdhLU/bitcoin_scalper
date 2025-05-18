"""
Stratégie de trading basée sur les retournements de tendance avec les Bandes de Bollinger.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.data_fetcher import DataFetcher, TimeFrame
from ..core.order_executor import OrderExecutor, OrderSide
from ..utils.indicators import calculate_bollinger_bands, calculate_rsi

from .base_strategy import BaseStrategy, Signal, SignalType


class BollingerBandsReversalStrategy(BaseStrategy):
    """
    Stratégie de trading utilisant les Bandes de Bollinger pour détecter les retournements de tendance.

    La stratégie utilise les conditions suivantes :
    - Signal d'achat :
        * Le prix touche ou dépasse la bande inférieure
        * Le RSI est en zone de survente (<30)
        * Le prix commence à remonter (bougie verte)

    - Signal de vente :
        * Le prix touche ou dépasse la bande supérieure
        * Le RSI est en zone de surachat (>70)
        * Le prix commence à baisser (bougie rouge)
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        order_executor: OrderExecutor,
        symbols: List[str],
        timeframe: TimeFrame,
        params: Optional[Dict] = None,
    ):
        """
        Initialise la stratégie.

        Args:
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            params: Paramètres de la stratégie (optionnel)
        """
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "min_reversal_pct": 0.5,
        }

        if params:
            default_params.update(params)

        super().__init__(
            name="Bollinger Bands Reversal",
            description="Stratégie basée sur les retournements de tendance avec les Bandes de Bollinger",
            data_fetcher=data_fetcher,
            order_executor=order_executor,
            params=default_params,
            symbols=symbols,
            timeframe=timeframe,
        )

    def _validate_params(self) -> None:
        """
        Vérifie que les paramètres sont valides.

        Raises:
            ValueError: Si les paramètres sont invalides
        """
        if self.params["bb_period"] < 2:
            raise ValueError("La période des Bandes de Bollinger doit être >= 2")

        if self.params["bb_std"] <= 0:
            raise ValueError("L'écart-type des Bandes de Bollinger doit être > 0")

        if not 0 < self.params["rsi_oversold"] < self.params["rsi_overbought"] < 100:
            raise ValueError(
                "Les seuils RSI doivent respecter: 0 < oversold < overbought < 100"
            )

        if self.params["min_reversal_pct"] <= 0:
            raise ValueError("Le pourcentage minimum de retournement doit être > 0")

    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.

        Returns:
            int: Nombre de bougies requises
        """
        return max(self.params["bb_period"], self.params["rsi_period"]) * 2

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
        # Calculer les indicateurs
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            data["close"], self.params["bb_period"], self.params["bb_std"]
        )
        rsi = calculate_rsi(data["close"], self.params["rsi_period"])

        current_price = data["close"].iloc[-1]
        current_rsi = rsi.iloc[-1]

        # Vérifier les conditions d'entrée
        price_below_lower = current_price <= bb_lower.iloc[-1]
        price_above_upper = current_price >= bb_upper.iloc[-1]
        rsi_oversold = current_rsi < self.params["rsi_oversold"]
        rsi_overbought = current_rsi > self.params["rsi_overbought"]

        # Calculer le pourcentage de retournement
        if price_below_lower and rsi_oversold:
            reversal_pct = (current_price - bb_lower.iloc[-1]) / bb_lower.iloc[-1]
            if reversal_pct >= self.params["min_reversal_pct"]:
                metadata = {
                    "bb_lower": bb_lower.iloc[-1],
                    "bb_middle": bb_middle.iloc[-1],
                    "rsi": current_rsi,
                    "reversal_pct": reversal_pct,
                }
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=reversal_pct,
                    metadata=metadata,
                )
                return True, signal

        elif price_above_upper and rsi_overbought:
            reversal_pct = (bb_upper.iloc[-1] - current_price) / bb_upper.iloc[-1]
            if reversal_pct >= self.params["min_reversal_pct"]:
                metadata = {
                    "bb_upper": bb_upper.iloc[-1],
                    "bb_middle": bb_middle.iloc[-1],
                    "rsi": current_rsi,
                    "reversal_pct": reversal_pct,
                }
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=reversal_pct,
                    metadata=metadata,
                )
                return True, signal

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
        # Calculer les indicateurs
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            data["close"], self.params["bb_period"], self.params["bb_std"]
        )
        rsi = calculate_rsi(data["close"], self.params["rsi_period"])

        current_price = data["close"].iloc[-1]
        current_rsi = rsi.iloc[-1]

        if position_side == OrderSide.BUY:
            # Sortir d'une position longue si le prix atteint la bande moyenne
            if current_price >= bb_middle.iloc[-1]:
                metadata = {
                    "bb_middle": bb_middle.iloc[-1],
                    "rsi": current_rsi,
                    "exit_reason": "middle_band",
                }
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=1.0,
                    metadata=metadata,
                )
                return True, signal

        elif position_side == OrderSide.SELL:
            # Sortir d'une position courte si le prix atteint la bande moyenne
            if current_price <= bb_middle.iloc[-1]:
                metadata = {
                    "bb_middle": bb_middle.iloc[-1],
                    "rsi": current_rsi,
                    "exit_reason": "middle_band",
                }
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=1.0,
                    metadata=metadata,
                )
                return True, signal

        return False, None

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Génère les signaux de trading pour un symbole.

        Args:
            symbol: Symbole à analyser
            data: Données OHLCV

        Returns:
            List[Signal]: Liste des signaux générés
        """
        signals = []

        # Calculer les indicateurs
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            data["close"], self.params["bb_period"], self.params["bb_std"]
        )
        rsi = calculate_rsi(data["close"], self.params["rsi_period"])

        # Générer les signaux pour chaque bougie
        for i in range(len(data)):
            price = data["close"].iloc[i]
            current_rsi = rsi.iloc[i]

            # Vérifier les conditions d'entrée
            price_below_lower = price <= bb_lower.iloc[i]
            price_above_upper = price >= bb_upper.iloc[i]
            rsi_oversold = current_rsi < self.params["rsi_oversold"]
            rsi_overbought = current_rsi > self.params["rsi_overbought"]

            if price_below_lower and rsi_oversold:
                reversal_pct = (price - bb_lower.iloc[i]) / bb_lower.iloc[i]
                if reversal_pct >= self.params["min_reversal_pct"]:
                    metadata = {
                        "bb_lower": bb_lower.iloc[i],
                        "bb_middle": bb_middle.iloc[i],
                        "rsi": current_rsi,
                        "reversal_pct": reversal_pct,
                    }
                    signal = Signal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=data.index[i],
                        price=price,
                        strength=reversal_pct,
                        metadata=metadata,
                    )
                    signals.append(signal)

            elif price_above_upper and rsi_overbought:
                reversal_pct = (bb_upper.iloc[i] - price) / bb_upper.iloc[i]
                if reversal_pct >= self.params["min_reversal_pct"]:
                    metadata = {
                        "bb_upper": bb_upper.iloc[i],
                        "bb_middle": bb_middle.iloc[i],
                        "rsi": current_rsi,
                        "reversal_pct": reversal_pct,
                    }
                    signal = Signal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        timestamp=data.index[i],
                        price=price,
                        strength=reversal_pct,
                        metadata=metadata,
                    )
                    signals.append(signal)

        return signals

    def calculate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule les signaux d'achat et de vente.

        Args:
            data: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Tuple[pd.Series, pd.Series]: Signaux d'achat et de vente
        """
        try:
            # Calcul des indicateurs
            upper_band, middle_band, lower_band = calculate_bollinger_bands(
                data["close"],
                period=self.params["bb_period"],
                num_std=self.params["bb_std"],
            )

            rsi = calculate_rsi(data["close"], period=self.params["rsi_period"])

            # Calcul des retournements de prix
            price_change_pct = ((data["close"] - data["open"]) / data["open"]) * 100
            is_bullish = pd.Series(
                price_change_pct > self.params["min_reversal_pct"], index=data.index
            )
            is_bearish = pd.Series(
                price_change_pct < -self.params["min_reversal_pct"], index=data.index
            )

            # Signaux d'achat
            buy_signals = pd.Series(
                (data["low"] <= lower_band)
                & (  # Prix touche/dépasse la bande inférieure
                    rsi < self.params["rsi_oversold"]
                )
                & is_bullish,  # RSI en survente  # Retournement haussier
                index=data.index,
            )

            # Signaux de vente
            sell_signals = pd.Series(
                (data["high"] >= upper_band)
                & (  # Prix touche/dépasse la bande supérieure
                    rsi > self.params["rsi_overbought"]
                )
                & is_bearish,  # RSI en surachat  # Retournement baissier
                index=data.index,
            )

            # Log uniquement si pas en optimisation et s'il y a un signal
            if not self.is_optimizing and (
                buy_signals.iloc[-1] or sell_signals.iloc[-1]
            ):
                logger.info(
                    f"Analyse BB - Prix: {data['close'].iloc[-1]:.2f}, RSI: {rsi.iloc[-1]:.2f}"
                )

                if buy_signals.iloc[-1]:
                    logger.info(
                        f"Signal ACHAT - Conditions: Prix < BB_inf={data['low'].iloc[-1] <= lower_band.iloc[-1]}, RSI survendu={rsi.iloc[-1] < self.params['rsi_oversold']}, Retournement haussier={is_bullish.iloc[-1]}"
                    )
                elif sell_signals.iloc[-1]:
                    logger.info(
                        f"Signal VENTE - Conditions: Prix > BB_sup={data['high'].iloc[-1] >= upper_band.iloc[-1]}, RSI suracheté={rsi.iloc[-1] > self.params['rsi_overbought']}, Retournement baissier={is_bearish.iloc[-1]}"
                    )

            return buy_signals, sell_signals

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse BB: {str(e)}")
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
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            data["close"],
            period=self.params["bb_period"],
            num_std=self.params["bb_std"],
        )

        rsi = calculate_rsi(data["close"], period=self.params["rsi_period"])

        current_price = data["close"].iloc[index]
        current_rsi = rsi.iloc[index]

        if signal_type == "buy":
            band_distance = (
                (current_price - lower_band.iloc[index]) / current_price * 100
            )
            signal_strength = (
                (self.params["rsi_oversold"] - current_rsi)
                / self.params["rsi_oversold"]
                * 100
            )
        else:  # signal_type == 'sell'
            band_distance = (
                (upper_band.iloc[index] - current_price) / current_price * 100
            )
            signal_strength = (
                (current_rsi - self.params["rsi_overbought"])
                / (100 - self.params["rsi_overbought"])
                * 100
            )

        return {
            "signal_type": signal_type,
            "price": current_price,
            "rsi": current_rsi,
            "band_distance": band_distance,
            "signal_strength": signal_strength,
            "bb_upper": upper_band.iloc[index],
            "bb_middle": middle_band.iloc[index],
            "bb_lower": lower_band.iloc[index],
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
        _, middle_band, _ = calculate_bollinger_bands(
            data["close"],
            period=self.params["bb_period"],
            num_std=self.params["bb_std"],
        )

        current_price = data["close"].iloc[index]

        if signal_type == "buy":
            # Stop loss sous le plus bas récent ou la bande moyenne
            stop_level = min(
                data["low"]
                .iloc[max(0, index - self.params["bb_period"]) : index + 1]
                .min(),
                middle_band.iloc[index],
            )
        else:  # signal_type == 'sell'
            # Stop loss au-dessus du plus haut récent ou la bande moyenne
            stop_level = max(
                data["high"]
                .iloc[max(0, index - self.params["bb_period"]) : index + 1]
                .max(),
                middle_band.iloc[index],
            )

        return stop_level

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
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            data["close"],
            period=self.params["bb_period"],
            num_std=self.params["bb_std"],
        )

        current_price = data["close"].iloc[index]

        if signal_type == "buy":
            # Take profit à la bande supérieure
            take_profit = upper_band.iloc[index]
        else:  # signal_type == 'sell'
            # Take profit à la bande inférieure
            take_profit = lower_band.iloc[index]

        return take_profit

    def _generate_signals_impl(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Implémente la logique de la stratégie Bollinger Bands.

        Args:
            data: DataFrame avec les données OHLCV
            signals: Série des signaux à modifier
        """
        try:
            # Calculer les bandes de Bollinger
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                data["close"], self.params["bb_period"], self.params["bb_std"]
            )

            # Calculer le RSI
            rsi = calculate_rsi(data["close"], self.params["rsi_period"])

            # Calculer la variation du prix
            price_change = data["close"].pct_change()

            # Générer les signaux
            # Signal d'achat: Prix touche la bande inférieure et rebondit
            # avec un RSI en zone de survente
            buy_signal = (
                (data["close"] <= bb_lower)
                & (price_change > self.params["min_reversal_pct"])
                & (rsi < 30)
            )

            # Signal de vente: Prix touche la bande supérieure et rebondit
            # avec un RSI en zone de surachat
            sell_signal = (
                (data["close"] >= bb_upper)
                & (price_change < -self.params["min_reversal_pct"])
                & (rsi > 70)
            )

            # Mettre à jour les signaux
            signals[buy_signal] = 1
            signals[sell_signal] = -1

        except Exception as e:
            logger.error(
                f"Erreur lors de la génération des signaux Bollinger Bands: {str(e)}"
            )
