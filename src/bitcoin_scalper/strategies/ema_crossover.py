"""
Stratégie de trading basée sur le croisement des moyennes mobiles exponentielles (EMA).
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..core.data_fetcher import DataFetcher, TimeFrame
from ..core.order_executor import OrderExecutor, OrderSide
from ..strategies.base_strategy import BaseStrategy, Signal, SignalType
from ..utils.indicators import calculate_ema


class EMACrossoverStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le croisement des EMA.

    Cette stratégie génère des signaux d'achat lorsque l'EMA rapide croise au-dessus
    de l'EMA lente, et des signaux de vente lorsque l'EMA rapide croise en dessous.

    Attributes:
        fast_period (int): Période de l'EMA rapide
        slow_period (int): Période de l'EMA lente
        min_crossover_strength (float): Force minimale du croisement pour générer un signal
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
        Initialise la stratégie EMA Crossover.

        Args:
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            params: Paramètres de la stratégie (optionnel)
        """
        default_params = {
            "fast_period": 9,
            "slow_period": 21,
            "min_crossover_strength": 0.0001,
        }

        if params:
            default_params.update(params)

        super().__init__(
            name="EMA Crossover",
            description="Stratégie basée sur le croisement des moyennes mobiles exponentielles",
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
        if self.params["fast_period"] >= self.params["slow_period"]:
            raise ValueError(
                "La période rapide doit être inférieure à la période lente"
            )

        if self.params["min_crossover_strength"] <= 0:
            raise ValueError("La force minimale du croisement doit être positive")

    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.

        Returns:
            int: Nombre de bougies requises
        """
        return self.params["slow_period"] * 2

    def _calculate_emas(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule les EMA rapide et lente.

        Args:
            data: Données OHLCV

        Returns:
            Tuple[pd.Series, pd.Series]: (EMA rapide, EMA lente)
        """
        fast_ema = calculate_ema(data["close"], self.params["fast_period"])
        slow_ema = calculate_ema(data["close"], self.params["slow_period"])

        return fast_ema, slow_ema

    def _calculate_crossover_strength(
        self, fast_ema: pd.Series, slow_ema: pd.Series
    ) -> float:
        """
        Calcule la force du croisement.

        Args:
            fast_ema: EMA rapide
            slow_ema: EMA lente

        Returns:
            float: Force du croisement
        """
        return abs(fast_ema.iloc[-1] - slow_ema.iloc[-1]) / slow_ema.iloc[-1]

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
        fast_ema, slow_ema = self._calculate_emas(data)

        # Vérifier le croisement
        current_cross = fast_ema.iloc[-1] > slow_ema.iloc[-1]
        previous_cross = fast_ema.iloc[-2] > slow_ema.iloc[-2]

        if current_cross and not previous_cross:
            # Croisement à la hausse
            strength = self._calculate_crossover_strength(fast_ema, slow_ema)

            if strength >= self.params["min_crossover_strength"]:
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=data["close"].iloc[-1],
                    strength=strength,
                    metadata={
                        "fast_ema": fast_ema.iloc[-1],
                        "slow_ema": slow_ema.iloc[-1],
                        "crossover_strength": strength,
                    },
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
        fast_ema, slow_ema = self._calculate_emas(data)

        # Vérifier le croisement
        current_cross = fast_ema.iloc[-1] > slow_ema.iloc[-1]
        previous_cross = fast_ema.iloc[-2] > slow_ema.iloc[-2]

        if position_side == OrderSide.BUY and not current_cross and previous_cross:
            # Croisement à la baisse pour une position longue
            strength = self._calculate_crossover_strength(fast_ema, slow_ema)

            if strength >= self.params["min_crossover_strength"]:
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=data["close"].iloc[-1],
                    strength=strength,
                    metadata={
                        "fast_ema": fast_ema.iloc[-1],
                        "slow_ema": slow_ema.iloc[-1],
                        "crossover_strength": strength,
                    },
                )
                return True, signal

        elif position_side == OrderSide.SELL and current_cross and not previous_cross:
            # Croisement à la hausse pour une position courte
            strength = self._calculate_crossover_strength(fast_ema, slow_ema)

            if strength >= self.params["min_crossover_strength"]:
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=data["close"].iloc[-1],
                    strength=strength,
                    metadata={
                        "fast_ema": fast_ema.iloc[-1],
                        "slow_ema": slow_ema.iloc[-1],
                        "crossover_strength": strength,
                    },
                )
                return True, signal

        return False, None

    def _generate_signals_impl(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Implémente la logique de la stratégie EMA Crossover.

        Args:
            data: DataFrame avec les données OHLCV
            signals: Série des signaux à modifier
        """
        try:
            # Calculer les EMAs
            fast_ema = calculate_ema(data["close"], self.params["fast_period"])
            slow_ema = calculate_ema(data["close"], self.params["slow_period"])

            # Calculer la différence entre les EMAs
            ema_diff = fast_ema - slow_ema
            prev_ema_diff = ema_diff.shift(1)

            # Générer les signaux
            # Signal d'achat: EMA rapide croise au-dessus de l'EMA lente
            buy_signal = (
                (ema_diff > 0)
                & (prev_ema_diff <= 0)
                & (ema_diff > self.params["min_crossover_strength"])
            )

            # Signal de vente: EMA rapide croise en-dessous de l'EMA lente
            sell_signal = (
                (ema_diff < 0)
                & (prev_ema_diff >= 0)
                & (abs(ema_diff) > self.params["min_crossover_strength"])
            )

            # Mettre à jour les signaux
            signals[buy_signal] = 1
            signals[sell_signal] = -1

        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux EMA: {str(e)}")
