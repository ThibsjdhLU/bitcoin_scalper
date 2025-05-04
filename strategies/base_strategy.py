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

from core.data_fetcher import DataFetcher, TimeFrame
from core.order_executor import OrderExecutor, OrderSide, OrderType

class SignalType(Enum):
    """Types de signaux possibles."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class Signal:
    """Représente un signal de trading."""
    type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    strength: float  # Entre 0 et 1
    metadata: Dict  # Informations supplémentaires (ex: indicateurs)

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
    
    def __init__(
        self,
        name: str,
        description: str,
        data_fetcher: DataFetcher,
        order_executor: OrderExecutor,
        params: Dict,
        symbols: List[str],
        timeframe: TimeFrame
    ):
        """
        Initialise la stratégie.
        
        Args:
            name: Nom de la stratégie
            description: Description de la stratégie
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            params: Paramètres de la stratégie
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
        """
        self.name = name
        self.description = description
        self.data_fetcher = data_fetcher
        self.order_executor = order_executor
        self.params = params
        self.symbols = symbols
        self.timeframe = timeframe
        
        # Vérifier les paramètres requis
        self._validate_params()
        
        logger.info(f"Stratégie {name} initialisée avec {len(symbols)} symboles")
    
    def _validate_params(self) -> None:
        """
        Vérifie que tous les paramètres requis sont présents.
        À surcharger dans les classes filles.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
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
                end_date=end_date
            )
            
            if data is None or len(data) < self.get_required_data():
                logger.warning(
                    f"Données insuffisantes pour {symbol}: "
                    f"{len(data) if data is not None else 0} bougies"
                )
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {str(e)}")
            return None
    
    def analyze(self, symbol: str) -> Optional[Tuple[bool, bool, List[Signal]]]:
        """
        Analyse un symbole et génère les signaux.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Optional[Tuple[bool, bool, List[Signal]]]: (Entrée, Sortie, Signaux)
        """
        data = self.get_current_data(symbol)
        if data is None:
            return None
        
        signals = self.generate_signals(symbol, data)
        should_enter, enter_signal = self.should_enter(symbol, data)
        
        # Pour l'instant, on suppose qu'il n'y a pas de position ouverte
        should_exit, exit_signal = self.should_exit(symbol, data, OrderSide.BUY)
        
        return should_enter, should_exit, signals 