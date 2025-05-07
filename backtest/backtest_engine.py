"""
Module de backtesting pour tester les stratégies de trading.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from core.mt5_connector import MT5Connector
from core.order_executor import OrderExecutor, OrderType, OrderSide

class BacktestEngine:
    """
    Moteur de backtesting pour tester les stratégies de trading.
    
    Attributes:
        connector (MT5Connector): Instance du connecteur MT5
        order_executor (OrderExecutor): Instance de l'exécuteur d'ordres
        data (pd.DataFrame): Données historiques pour le backtest
        initial_balance (float): Solde initial
        current_balance (float): Solde actuel
        positions (List[Dict]): Liste des positions ouvertes
        trades (List[Dict]): Historique des trades
    """
    
    def __init__(
        self,
        connector: MT5Connector,
        order_executor: OrderExecutor,
        initial_balance: float = 10000.0
    ):
        """
        Initialise le moteur de backtesting.
        
        Args:
            connector: Instance du connecteur MT5
            order_executor: Instance de l'exécuteur d'ordres
            initial_balance: Solde initial pour le backtest
        """
        self.connector = connector
        self.order_executor = order_executor
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = []
        self.trades = []
        self.data = pd.DataFrame()
        
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        Charge les données historiques pour le backtest.
        
        Args:
            symbol: Symbole à trader
            timeframe: Timeframe (ex: '1m', '5m', '1h')
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            bool: True si le chargement est réussi, False sinon
        """
        try:
            # Convertir le timeframe en format MT5
            tf_map = {
                '1m': 'M1',
                '5m': 'M5',
                '15m': 'M15',
                '30m': 'M30',
                '1h': 'H1',
                '4h': 'H4',
                '1d': 'D1'
            }
            mt5_tf = tf_map.get(timeframe)
            if not mt5_tf:
                raise ValueError(f"Timeframe non supporté: {timeframe}")
            
            # Récupérer les données
            rates = self.connector.get_rates(
                symbol=symbol,
                timeframe=mt5_tf,
                start_date=start_date,
                end_date=end_date
            )
            
            if not rates:
                logger.error("Aucune donnée récupérée")
                return False
            
            # Convertir en DataFrame
            self.data = pd.DataFrame(rates)
            self.data['time'] = pd.to_datetime(self.data['time'], unit='s')
            self.data.set_index('time', inplace=True)
            
            logger.info(f"Données chargées: {len(self.data)} barres")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return False
    
    def run_backtest(self, strategy) -> Dict:
        """
        Exécute le backtest avec une stratégie donnée.
        
        Args:
            strategy: Instance de la stratégie à tester
            
        Returns:
            Dict: Résultats du backtest
        """
        if self.data.empty:
            logger.error("Aucune donnée chargée")
            return {}
        
        results = {
            'trades': [],
            'balance': [self.initial_balance],
            'equity': [self.initial_balance],
            'drawdown': [0.0],
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        try:
            # Parcourir les données
            for i in range(len(self.data)):
                current_bar = self.data.iloc[i]
                
                # Obtenir le signal de la stratégie
                signal = strategy.generate_signal(current_bar)
                
                if signal:
                    # Exécuter l'ordre
                    success, order_id = self._execute_signal(signal, current_bar)
                    
                    if success:
                        # Mettre à jour les résultats
                        self._update_results(results, signal, current_bar)
            
            # Calculer les métriques finales
            self._calculate_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {str(e)}")
            return {}
    
    def _execute_signal(
        self,
        signal: Dict,
        current_bar: pd.Series
    ) -> Tuple[bool, Optional[int]]:
        """
        Exécute un signal de trading.
        
        Args:
            signal: Signal de trading
            current_bar: Barre de prix actuelle
            
        Returns:
            Tuple[bool, Optional[int]]: (Succès, ID de l'ordre)
        """
        try:
            # Déterminer le type d'ordre
            if signal['type'] == 'MARKET':
                return self.order_executor.execute_market_order(
                    symbol=signal['symbol'],
                    volume=signal['volume'],
                    side=signal['side'],
                    sl=signal.get('sl'),
                    tp=signal.get('tp')
                )
            elif signal['type'] == 'LIMIT':
                return self.order_executor.execute_limit_order(
                    symbol=signal['symbol'],
                    volume=signal['volume'],
                    side=signal['side'],
                    price=signal['price'],
                    sl=signal.get('sl'),
                    tp=signal.get('tp')
                )
            elif signal['type'] == 'STOP':
                return self.order_executor.execute_stop_order(
                    symbol=signal['symbol'],
                    volume=signal['volume'],
                    side=signal['side'],
                    price=signal['price'],
                    sl=signal.get('sl'),
                    tp=signal.get('tp')
                )
            
            return False, None
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal: {str(e)}")
            return False, None
    
    def _update_results(
        self,
        results: Dict,
        signal: Dict,
        current_bar: pd.Series
    ) -> None:
        """
        Met à jour les résultats du backtest.
        
        Args:
            results: Dictionnaire des résultats
            signal: Signal de trading
            current_bar: Barre de prix actuelle
        """
        # Calculer le P&L
        entry_price = current_bar['close']
        exit_price = signal.get('tp', current_bar['close'])
        volume = signal['volume']
        
        if signal['side'] == OrderSide.BUY:
            pnl = (exit_price - entry_price) * volume
        else:
            pnl = (entry_price - exit_price) * volume
        
        # Mettre à jour le solde
        self.current_balance += pnl
        
        # Enregistrer le trade
        trade = {
            'entry_time': current_bar.name,
            'exit_time': current_bar.name,
            'symbol': signal['symbol'],
            'side': signal['side'].value,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'volume': volume,
            'pnl': pnl
        }
        
        results['trades'].append(trade)
        results['balance'].append(self.current_balance)
        results['equity'].append(self.current_balance)
        
        # Calculer le drawdown
        peak = max(results['equity'])
        drawdown = (peak - self.current_balance) / peak * 100
        results['drawdown'].append(drawdown)
    
    def _calculate_metrics(self, results: Dict) -> None:
        """
        Calcule les métriques finales du backtest.
        
        Args:
            results: Dictionnaire des résultats
        """
        if not results['trades']:
            return
        
        # Calculer le win rate
        winning_trades = sum(1 for t in results['trades'] if t['pnl'] > 0)
        total_trades = len(results['trades'])
        results['win_rate'] = winning_trades / total_trades * 100
        
        # Calculer le profit factor
        gross_profit = sum(t['pnl'] for t in results['trades'] if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in results['trades'] if t['pnl'] < 0))
        results['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Mettre à jour les compteurs
        results['total_trades'] = total_trades
        results['winning_trades'] = winning_trades
        results['losing_trades'] = total_trades - winning_trades 