"""
Service de backtesting pour tester les stratégies de trading.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BacktestService:
    def __init__(self):
        self.data_dir = Path("data")
        self.results = None
        
    def load_historical_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Charge les données historiques depuis un fichier CSV."""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Le fichier CSV doit contenir les colonnes: timestamp, open, high, low, close, volume")
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données historiques: {str(e)}")
            return None
            
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy_params: Dict,
                    initial_capital: float = 10000.0) -> Dict:
        """Exécute un backtest avec les paramètres donnés."""
        try:
            results = {
                'trades': [],
                'equity_curve': [],
                'metrics': {}
            }
            
            capital = initial_capital
            position = None
            entry_price = 0
            entry_time = None
            
            for i in range(len(data)):
                current_price = data.iloc[i]['close']
                current_time = data.index[i]
                
                # Logique de trading simplifiée (à adapter selon la stratégie)
                if position is None:  # Pas de position ouverte
                    if self._should_enter_long(data.iloc[i], strategy_params):
                        position = 'long'
                        entry_price = current_price
                        entry_time = current_time
                else:  # Position ouverte
                    if self._should_exit_long(data.iloc[i], strategy_params):
                        profit = (current_price - entry_price) / entry_price
                        capital *= (1 + profit)
                        
                        results['trades'].append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit,
                            'capital': capital
                        })
                        
                        position = None
                
                results['equity_curve'].append({
                    'timestamp': current_time,
                    'equity': capital
                })
            
            # Calcul des métriques
            results['metrics'] = self._calculate_metrics(results['trades'], initial_capital)
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {str(e)}")
            return None
            
    def _should_enter_long(self, candle: pd.Series, params: Dict) -> bool:
        """Détermine si on doit entrer en position longue."""
        # Exemple simple: entrée si le prix est au-dessus de la moyenne mobile
        if 'ma_period' in params:
            ma = candle['close'].rolling(window=params['ma_period']).mean()
            return candle['close'] > ma
        return False
        
    def _should_exit_long(self, candle: pd.Series, params: Dict) -> bool:
        """Détermine si on doit sortir d'une position longue."""
        # Exemple simple: sortie si le prix est en dessous de la moyenne mobile
        if 'ma_period' in params:
            ma = candle['close'].rolling(window=params['ma_period']).mean()
            return candle['close'] < ma
        return False
        
    def _calculate_metrics(self, trades: List[Dict], initial_capital: float) -> Dict:
        """Calcule les métriques de performance."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        profits = [t['profit'] for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calcul du drawdown maximum
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] * (1 + trade['profit']))
            
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calcul du ratio de Sharpe (simplifié)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
    def get_results(self) -> Optional[Dict]:
        """Retourne les résultats du dernier backtest."""
        return self.results 