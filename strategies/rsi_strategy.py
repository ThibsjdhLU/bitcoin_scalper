"""
Stratégie de trading basée sur le Relative Strength Index (RSI).
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.data_fetcher import DataFetcher, TimeFrame
from core.order_executor import OrderExecutor, OrderSide
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.indicators import calculate_rsi, calculate_ema

class RSIStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le RSI.
    
    Cette stratégie génère des signaux d'achat lorsque le RSI est en survente
    et des signaux de vente lorsque le RSI est en surachat. Elle utilise également
    une EMA comme filtre de tendance.
    
    Attributes:
        rsi_period (int): Période du RSI
        overbought_threshold (float): Seuil de surachat
        oversold_threshold (float): Seuil de survente
        trend_ema_period (int): Période de l'EMA pour le filtre de tendance
        exit_rsi_threshold (float): Seuil de sortie (distance du niveau neutre)
        min_bounce_strength (float): Force minimale du rebond pour générer un signal
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
        Initialise la stratégie RSI.
        
        Args:
            data_fetcher: Instance pour récupérer les données
            order_executor: Instance pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe utilisé
            params: Paramètres de la stratégie (optionnel)
        """
        default_params = {
            'rsi_period': 14,
            'overbought_threshold': 70,
            'oversold_threshold': 30,
            'trend_ema_period': 200,  # EMA longue pour le filtre de tendance
            'exit_rsi_threshold': 5,  # Distance du niveau 50 pour sortir
            'min_bounce_strength': 0.001  # Force minimale du rebond
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(
            name="RSI Strategy",
            description="Stratégie basée sur le RSI avec filtre de tendance EMA",
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
        if not 0 < self.params['oversold_threshold'] < self.params['overbought_threshold'] < 100:
            raise ValueError(
                "Les seuils doivent respecter: 0 < oversold < overbought < 100"
            )
        
        if self.params['rsi_period'] < 2:
            raise ValueError("La période du RSI doit être >= 2")
        
        if self.params['trend_ema_period'] < 1:
            raise ValueError("La période de l'EMA doit être >= 1")
        
        if self.params['exit_rsi_threshold'] < 0:
            raise ValueError("Le seuil de sortie doit être >= 0")
        
        if self.params['min_bounce_strength'] <= 0:
            raise ValueError("La force minimale du rebond doit être > 0")
    
    def get_required_data(self) -> int:
        """
        Retourne le nombre de bougies nécessaires pour calculer les indicateurs.
        
        Returns:
            int: Nombre de bougies requises
        """
        return max(self.params['rsi_period'], self.params['trend_ema_period']) * 2
    
    def _calculate_indicators(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule le RSI et l'EMA de tendance.
        
        Args:
            data: Données OHLCV
            
        Returns:
            Tuple[pd.Series, pd.Series]: (RSI, EMA)
        """
        rsi = calculate_rsi(data['close'], self.params['rsi_period'])
        trend_ema = calculate_ema(data['close'], self.params['trend_ema_period'])
        
        return rsi, trend_ema
    
    def _calculate_bounce_strength(
        self,
        rsi: pd.Series,
        price: pd.Series
    ) -> float:
        """
        Calcule la force du rebond.
        
        Args:
            rsi: Série du RSI
            price: Série des prix
            
        Returns:
            float: Force du rebond
        """
        # Force basée sur la variation du RSI et du prix
        rsi_change = abs(rsi.iloc[-1] - rsi.iloc[-2]) / 50  # Normalisé par rapport à la plage du RSI
        price_change = abs(price.iloc[-1] - price.iloc[-2]) / price.iloc[-2]
        
        return (rsi_change + price_change) / 2
    
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
        rsi, trend_ema = self._calculate_indicators(data)
        current_price = data['close'].iloc[-1]
        
        # Calculer la force du rebond
        bounce_strength = self._calculate_bounce_strength(rsi, data['close'])
        
        # Vérifier les conditions d'entrée
        is_oversold = rsi.iloc[-2] < self.params['oversold_threshold']
        is_overbought = rsi.iloc[-2] > self.params['overbought_threshold']
        rsi_bouncing_up = rsi.iloc[-1] > rsi.iloc[-2]
        rsi_bouncing_down = rsi.iloc[-1] < rsi.iloc[-2]
        
        # Vérifier la tendance
        above_ema = current_price > trend_ema.iloc[-1]
        
        if bounce_strength >= self.params['min_bounce_strength']:
            if is_oversold and rsi_bouncing_up and above_ema:
                # Signal d'achat
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=bounce_strength,
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'trend_ema': trend_ema.iloc[-1],
                        'bounce_strength': bounce_strength
                    }
                )
                return True, signal
            
            elif is_overbought and rsi_bouncing_down and not above_ema:
                # Signal de vente
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=bounce_strength,
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'trend_ema': trend_ema.iloc[-1],
                        'bounce_strength': bounce_strength
                    }
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
        rsi, _ = self._calculate_indicators(data)
        current_price = data['close'].iloc[-1]
        
        # Calculer la distance par rapport au niveau neutre (50)
        distance_from_neutral = abs(rsi.iloc[-1] - 50)
        
        if position_side == OrderSide.BUY:
            # Sortir d'une position longue
            if (rsi.iloc[-1] > self.params['overbought_threshold'] or
                distance_from_neutral < self.params['exit_rsi_threshold']):
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=distance_from_neutral / 50,  # Normalisé
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'exit_reason': 'overbought' if rsi.iloc[-1] > self.params['overbought_threshold'] else 'neutral'
                    }
                )
                return True, signal
        
        elif position_side == OrderSide.SELL:
            # Sortir d'une position courte
            if (rsi.iloc[-1] < self.params['oversold_threshold'] or
                distance_from_neutral < self.params['exit_rsi_threshold']):
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=distance_from_neutral / 50,  # Normalisé
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'exit_reason': 'oversold' if rsi.iloc[-1] < self.params['oversold_threshold'] else 'neutral'
                    }
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
        rsi, trend_ema = self._calculate_indicators(data)
        current_price = data['close'].iloc[-1]
        
        # Calculer la force du rebond
        bounce_strength = self._calculate_bounce_strength(rsi, data['close'])
        
        if bounce_strength >= self.params['min_bounce_strength']:
            # Vérifier les conditions de surachat/survente
            if rsi.iloc[-2] < self.params['oversold_threshold'] and rsi.iloc[-1] > rsi.iloc[-2]:
                # Signal d'achat sur rebond de survente
                signal = Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=bounce_strength,
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'trend_ema': trend_ema.iloc[-1],
                        'bounce_strength': bounce_strength,
                        'condition': 'oversold_bounce'
                    }
                )
                signals.append(signal)
            
            elif rsi.iloc[-2] > self.params['overbought_threshold'] and rsi.iloc[-1] < rsi.iloc[-2]:
                # Signal de vente sur rebond de surachat
                signal = Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    strength=bounce_strength,
                    metadata={
                        'rsi': rsi.iloc[-1],
                        'trend_ema': trend_ema.iloc[-1],
                        'bounce_strength': bounce_strength,
                        'condition': 'overbought_bounce'
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_signals_impl(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Implémente la logique de la stratégie RSI.
        
        Args:
            data: DataFrame avec les données OHLCV
            signals: Série des signaux à modifier
        """
        try:
            # Calculer le RSI
            rsi = calculate_rsi(data['close'], self.params['rsi_period'])
            
            # Calculer la tendance avec l'EMA
            trend_ema = calculate_ema(data['close'], self.params['trend_ema_period'])
            
            # Calculer la variation du RSI
            rsi_change = rsi - rsi.shift(1)
            
            # Générer les signaux
            # Signal d'achat: RSI sort de la zone de survente
            # et la tendance est haussière
            buy_signal = (
                (rsi < self.params['oversold_threshold']) &
                (rsi.shift(1) < self.params['oversold_threshold']) &
                (rsi_change > self.params['min_bounce_strength']) &
                (data['close'] > trend_ema)
            )
            
            # Signal de vente: RSI sort de la zone de surachat
            # et la tendance est baissière
            sell_signal = (
                (rsi > self.params['overbought_threshold']) &
                (rsi.shift(1) > self.params['overbought_threshold']) &
                (rsi_change < -self.params['min_bounce_strength']) &
                (data['close'] < trend_ema)
            )
            
            # Mettre à jour les signaux
            signals[buy_signal] = 1
            signals[sell_signal] = -1
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux RSI: {str(e)}") 