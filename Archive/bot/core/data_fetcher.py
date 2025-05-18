"""
Module de récupération des données OHLCV depuis MetaTrader 5.
Gère la récupération en temps réel et historique des données.
"""
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger
import numpy as np
import logging

from ..core.mt5_connector import MT5Connector


class TimeFrame(Enum):
    """Timeframes supportés."""

    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1


class DataFetcher:
    """
    Gère la récupération des données OHLCV depuis MT5.

    Attributes:
        connector (MT5Connector): Instance du connecteur MT5
        config_path (str): Chemin vers le fichier de configuration
    """

    def __init__(
        self, connector: MT5Connector, config_path: str = "config/config.json"
    ):
        """
        Initialise le récupérateur de données.

        Args:
            connector: Instance du connecteur MT5
            config_path: Chemin vers le fichier de configuration
        """
        self.connector = connector
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques OHLCV.

        Args:
            symbol: Symbole à récupérer
            timeframe: Timeframe des données
            start_date: Date de début
            end_date: Date de fin (optionnel, défaut: maintenant)

        Returns:
            Optional[pd.DataFrame]: DataFrame avec les données OHLCV ou None en cas d'erreur
        """
        if not self.connector.connected:
            logger.error("Non connecté à MT5")
            return None

        if end_date is None:
            end_date = datetime.now()

        try:
            # Récupérer les données
            rates = mt5.copy_rates_range(symbol, timeframe.value, start_date, end_date)

            if rates is None or len(rates) == 0:
                logger.error(f"Aucune donnée trouvée pour {symbol}")
                return None

            # Convertir en DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Renommer les colonnes
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "tick_volume": "Volume",
                },
                inplace=True,
            )

            logger.info(
                f"Données historiques récupérées pour {symbol}: {len(df)} bougies"
            )
            return df

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {str(e)}")
            return None

    def get_latest_candle(self, symbol: str, timeframe: TimeFrame) -> Optional[Dict]:
        """
        Récupère la dernière bougie.

        Args:
            symbol: Symbole à récupérer
            timeframe: Timeframe des données

        Returns:
            Optional[Dict]: Dernière bougie ou None en cas d'erreur
        """
        if not self.connector.connected:
            logger.error("Non connecté à MT5")
            return None

        try:
            # Récupérer la dernière bougie
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, 0, 1)

            if rates is None or len(rates) == 0:
                logger.error(f"Aucune donnée trouvée pour {symbol}")
                return None

            # Convertir en dictionnaire
            candle = rates[0]
            return {
                "time": datetime.fromtimestamp(candle["time"]),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["tick_volume"],
            }

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération de la dernière bougie: {str(e)}"
            )
            return None

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Récupère le prix actuel (bid/ask).

        Args:
            symbol: Symbole à récupérer

        Returns:
            Optional[Dict[str, float]]: Prix actuel ou None en cas d'erreur
        """
        if not self.connector.connected:
            logger.error("Non connecté à MT5")
            return None

        try:
            # Récupérer les informations du symbole
            symbol_info = mt5.symbol_info_tick(symbol)

            if symbol_info is None:
                logger.error(f"Symbole non trouvé: {symbol}")
                return None

            return {
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "last": symbol_info.last,
                "volume": symbol_info.volume,
            }

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix: {str(e)}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Récupère les informations détaillées d'un symbole.

        Args:
            symbol: Symbole à récupérer

        Returns:
            Optional[Dict]: Informations du symbole ou None en cas d'erreur
        """
        if not self.connector.connected:
            logger.error("Non connecté à MT5")
            return None

        try:
            # Récupérer les informations du symbole
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                logger.error(f"Symbole non trouvé: {symbol}")
                return None

            return {
                "name": symbol_info.name,
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "point": symbol_info.point,
                "digits": symbol_info.digits,
                "spread": symbol_info.spread,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
                "swap_long": symbol_info.swap_long,
                "swap_short": symbol_info.swap_short,
                "margin_initial": symbol_info.margin_initial,
                "margin_maintenance": symbol_info.margin_maintenance,
            }

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des informations du symbole: {str(e)}"
            )
            return None
