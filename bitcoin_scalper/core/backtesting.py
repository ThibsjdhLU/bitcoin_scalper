import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional

class Backtester:
    """
    Backtesting vectorisé sur DataFrame de signaux/features.
    Simule l'exécution des trades, calcule les KPIs principaux.
    """
    def __init__(self, df: pd.DataFrame, signal_col: str = "signal", price_col: str = "close", fee: float = 0.0):
        self.df = df.copy()
        self.signal_col = signal_col
        self.price_col = price_col
        self.fee = fee
        self.trades = None
        self.kpis = None

    def run(self):
        """
        Exécute le backtest sur la base des signaux (1=achat, -1=vente, 0=flat).
        """
        df = self.df
        signals = df[self.signal_col].fillna(0).values
        prices = df[self.price_col].values
        positions = np.zeros_like(signals)
        # Génère la position (long/short/flat)
        for i in range(1, len(signals)):
            if signals[i] != 0:
                positions[i] = signals[i]
            else:
                positions[i] = positions[i-1]
        # Calcul des retours
        returns = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if positions[i-1] == 1:
                returns[i] = (prices[i] - prices[i-1]) / prices[i-1] - self.fee
            elif positions[i-1] == -1:
                returns[i] = (prices[i-1] - prices[i]) / prices[i-1] - self.fee
            else:
                returns[i] = 0
        equity_curve = np.cumprod(1 + returns)
        self.df["equity_curve"] = equity_curve
        self.df["returns"] = returns
        self.trades = self._extract_trades(positions, prices)
        self.kpis = self._compute_kpis(equity_curve, returns, self.trades)
        return self.df, self.trades, self.kpis

    def _extract_trades(self, positions, prices):
        # Détecte les changements de position pour extraire les trades
        trades = []
        pos = 0
        entry = None
        for i, p in enumerate(positions):
            if p != pos:
                if pos != 0 and entry is not None:
                    trades.append({
                        "side": "long" if pos == 1 else "short",
                        "entry": entry,
                        "exit": prices[i],
                        "pnl": (prices[i] - entry) if pos == 1 else (entry - prices[i])
                    })
                if p != 0:
                    entry = prices[i]
                else:
                    entry = None
                pos = p
        return trades

    def _compute_kpis(self, equity_curve, returns, trades):
        # Sharpe ratio
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252) if np.std(returns) > 0 else 0
        # Max drawdown
        roll_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = np.min(drawdown)
        # Profit factor
        gains = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [-t["pnl"] for t in trades if t["pnl"] < 0]
        profit_factor = (np.sum(gains) / np.sum(losses)) if np.sum(losses) > 0 else np.inf
        # Taux de réussite
        win_rate = (np.sum([t["pnl"] > 0 for t in trades]) / len(trades)) if trades else 0
        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "nb_trades": len(trades),
            "final_return": equity_curve[-1] - 1 if len(equity_curve) > 0 else 0
        } 