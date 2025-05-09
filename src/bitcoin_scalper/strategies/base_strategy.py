"""
Module de base pour les stratégies de trading.
Définit l'interface commune et les fonctionnalités de base pour toutes les stratégies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from ..core.data_fetcher import DataFetcher, TimeFrame
from ..core.order_executor import OrderExecutor, OrderSide, OrderType


class SignalType(Enum):
    """Types de signaux possibles."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Représente un signal de trading.

    Attributes:
        type (SignalType): Type de signal
        symbol (str): Symbole concerné
        timestamp (datetime): Horodatage du signal
        price (float): Prix au moment du signal
        strength (float): Force du signal (0-1)
        metadata (Dict): Métadonnées additionnelles
    """

    type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    strength: float
    metadata: Dict


class BaseStrategy(ABC):
    """
    Classe abstraite définissant l'interface commune pour toutes les stratégies.

    Attributes:
        name (str): Nom de la stratégie
        description (str): Description de la stratégie
        data_fetcher (DataFetcher): Instance pour récupérer les données
        order_executor (OrderExecutor): Instance pour exécuter les ordres
        params (Dict): Paramètres de la stratégie
        symbols (List[str]): Liste des symboles à trader
        timeframe (TimeFrame): Timeframe utilisé
    """

    # Flag de classe pour suivre l'initialisation
    _initialized_strategies = set()

    def __init__(
        self,
        name: str,
        description: str,
        data_fetcher: DataFetcher,
        order_executor: OrderExecutor,
        params: Dict,
        symbols: List[str],
        timeframe: TimeFrame,
        is_optimizing: bool = False,
    ):
        """
        Initialise la stratégie de base.

        Args:
            name: Nom de la stratégie
            description: Description de la stratégie
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            params: Paramètres de la stratégie
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            is_optimizing: Si True, désactive les logs pendant l'optimisation
        """
        self.name = name
        self.description = description
        self.data_fetcher = data_fetcher
        self.order_executor = order_executor
        self.params = params
        self.symbols = symbols
        self.timeframe = timeframe
        self.is_optimizing = is_optimizing

        # Log d'initialisation unique par type de stratégie
        strategy_key = f"{name}_{len(symbols)}"
        if not self.is_optimizing and strategy_key not in self._initialized_strategies:
            logger.info(f"Stratégie {name} initialisée avec {len(symbols)} symboles")
            self._initialized_strategies.add(strategy_key)

        # Vérifier les paramètres requis
        self._validate_params()

        # Initialiser le volume
        self.volume = self.params.get("volume", 0.01)

    def _validate_params(self) -> None:
        """
        Vérifie que tous les paramètres requis sont présents.
        À surcharger dans les classes filles.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> pd.Series:
        """
        Génère les signaux de trading.

        Args:
            symbol: Symbole à analyser
            data: DataFrame avec les données OHLCV

        Returns:
            pd.Series: Série des signaux (1: achat, -1: vente, 0: neutre)
        """
        try:
            # S'assurer que les données sont un DataFrame pandas
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Vérifier que les colonnes nécessaires sont présentes
            required_columns = ["open", "high", "low", "close"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(
                    f"Données manquantes. Colonnes requises: {required_columns}"
                )

            # Générer les signaux
            signals = pd.Series(0, index=data.index)

            # Implémentation spécifique à la stratégie
            self._generate_signals_impl(data, signals)

            return signals

        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {str(e)}")
            return pd.Series(0, index=data.index)

    def _generate_signals_impl(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Implémentation spécifique de la génération des signaux.
        À surcharger par les classes dérivées.

        Args:
            data: DataFrame avec les données OHLCV
            signals: Série des signaux à modifier
        """
        raise NotImplementedError(
            "La méthode _generate_signals_impl doit être implémentée"
        )

    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.

        Returns:
            int: Nombre de bougies requises
        """
        return 100  # Valeur par défaut

    def update_params(self, new_params: Dict) -> None:
        """
        Met à jour les paramètres de la stratégie.

        Args:
            new_params: Nouveaux paramètres
        """
        self.params.update(new_params)
        self._validate_params()
        logger.info(f"Paramètres de {self.name} mis à jour")

    def get_current_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données actuelles pour un symbole.

        Args:
            symbol: Symbole à récupérer

        Returns:
            Optional[pd.DataFrame]: Données OHLCV ou None en cas d'erreur
        """
        try:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(
                minutes=self.get_required_data() * self.timeframe.value
            )

            data = self.data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=self.timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if data is None or len(data) < self.get_required_data():
                logger.warning(
                    f"Données insuffisantes pour {symbol}: "
                    f"{len(data) if data is not None else 0} bougies"
                )
                return None

            # Convertir les données en DataFrame avec le bon format
            if isinstance(data, list):
                data = pd.DataFrame(data)
                data.columns = [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ]
                data.set_index("time", inplace=True)
                data.index = pd.to_datetime(data.index, unit="s")

            return data

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {str(e)}")
            return None

    def analyze(
        self, symbol: str, data: pd.DataFrame
    ) -> Tuple[bool, bool, List[Signal]]:
        """
        Analyse un symbole et retourne les signaux de trading.

        Args:
            symbol: Symbole à analyser
            data: DataFrame avec les données OHLCV

        Returns:
            Tuple[bool, bool, List[Signal]]: (Doit entrer, Doit sortir, Liste des signaux)
        """
        try:
            # Vérifier si on doit entrer en position
            should_enter, enter_signal = self.should_enter(symbol, data)

            # Vérifier si on doit sortir d'une position
            should_exit, exit_signal = self.should_exit(symbol, data, OrderSide.BUY)

            # Créer la liste des signaux
            signals = []
            if enter_signal is not None:
                signals.append(enter_signal)
            if exit_signal is not None:
                signals.append(exit_signal)

            return should_enter, should_exit, signals

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {symbol}: {str(e)}")
            return False, False, []

    def get_volume(self) -> float:
        """
        Retourne le volume à utiliser pour les ordres.

        Returns:
            float: Volume à utiliser
        """
        # Utiliser le volume de la stratégie ou le volume par défaut
        volume = self.params.get("volume", 0.01)

        # S'assurer que le volume ne dépasse pas le maximum autorisé
        max_volume = self.params.get("max_position_size", 0.01)
        return min(volume, max_volume)
