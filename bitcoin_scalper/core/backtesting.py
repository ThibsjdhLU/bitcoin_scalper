import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import time
from bitcoin_scalper.core.order_execution import execute_adaptive_trade

logger = logging.getLogger("bitcoin_scalper.backtesting")
logger.setLevel(logging.INFO)

class Backtester:
    """
    Backtest réaliste d'une stratégie ML : PnL, Sharpe, drawdown, frais, slippage, equity curve.
    Permet l'injection de coûts dynamiques, contraintes, et la comparaison à un benchmark.

    Args:
        df (pd.DataFrame): Données historiques avec signaux et prix.
        signal_col (str): Colonne des signaux (-1, 0, 1).
        price_col (str): Colonne des prix d'exécution.
        label_col (Optional[str]): Colonne des labels (pour reporting).
        model (Optional[Any]): Modèle ML (optionnel, si signaux à prédire).
        initial_capital (float): Capital initial.
        fee (float): Frais de transaction (proportionnel, si fee_fn non fourni).
        slippage (float): Slippage (proportionnel, si slippage_fn non fourni).
        min_trade_size (float): Taille minimale d'un trade.
        out_dir (str): Dossier de sortie des rapports.
        slippage_fn (Optional[Callable]): Fonction de slippage dynamique (price, volume, orderbook, timestamp) -> float.
        spread_series (Optional[Union[pd.Series, Callable]]): Série temporelle ou fonction fournissant le spread à chaque timestamp.
        fee_fn (Optional[Callable]): Fonction de calcul des frais (price, volume, timestamp) -> float.
        orderbook_series (Optional[pd.Series]): Série temporelle de snapshots orderbook (pour slippage avancé).
        latency_fn (Optional[Callable]): Fonction de latence (timestamp, trade_info) -> float (secondes).
        reject_fn (Optional[Callable]): Fonction de rejet d'ordre (timestamp, trade_info) -> bool.
        benchmarks (Optional[list]): Liste des benchmarks à comparer (ex: ['buy_and_hold', 'rsi2', ...]).
        scheduler (Any): Scheduler adaptatif.
        client (Any): Client pour compatibilité interface.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close",
        label_col: Optional[str] = None,
        model: Optional[Any] = None,
        initial_capital: float = 10000.0,
        fee: float = 0.0005,
        slippage: float = 0.0002,
        min_trade_size: float = 0.0,
        out_dir: str = "backtest_reports",
        slippage_fn: Optional[Any] = None,
        spread_series: Optional[Any] = None,
        fee_fn: Optional[Any] = None,
        orderbook_series: Optional[Any] = None,
        latency_fn: Optional[Any] = None,
        reject_fn: Optional[Any] = None,
        benchmarks: Optional[list] = None,
        scheduler: Any = None,
        client: Any = None
    ):
        self.df = df.copy()
        self.signal_col = signal_col
        self.price_col = price_col
        self.label_col = label_col
        self.model = model
        self.initial_capital = initial_capital
        self.fee = fee
        self.slippage = slippage
        self.min_trade_size = min_trade_size
        self.out_dir = out_dir
        self.slippage_fn = slippage_fn
        self.spread_series = spread_series
        self.fee_fn = fee_fn
        self.orderbook_series = orderbook_series
        self.latency_fn = latency_fn
        self.reject_fn = reject_fn
        self.benchmarks = benchmarks or []
        self.scheduler = scheduler
        self.client = client
        os.makedirs(self.out_dir, exist_ok=True)

    def _run_benchmark(self, name, df):
        """Calcule les KPIs d'un benchmark donné sur le DataFrame df."""
        if name == 'buy_and_hold':
            # Entrée au premier prix, sortie au dernier, position = 1
            entry = df[self.price_col].iloc[0]
            exit = df[self.price_col].iloc[-1]
            returns = (exit - entry) / entry
            kpis = {
                'final_return': returns,
                'sharpe': 0,  # non pertinent sur un seul trade
                'max_drawdown': 0,
                'nb_trades': 1
            }
            equity_curve = np.linspace(entry, exit, len(df))
            return {'kpis': kpis, 'equity_curve': equity_curve}
        elif name == 'rsi2':
            # Benchmark RSI2 : achat si RSI<30, vente si RSI>70
            if 'rsi' not in df.columns:
                return None
            capital = self.initial_capital
            position = 0
            equity_curve = []
            last_price = None
            for i, row in df.iterrows():
                price = row[self.price_col]
                rsi = row['rsi']
                if rsi < 30 and position == 0:
                    position = 1
                    last_price = price
                elif rsi > 70 and position == 1:
                    capital += price - last_price
                    position = 0
                equity_curve.append(capital)
            if position == 1:
                capital += df[self.price_col].iloc[-1] - last_price
            kpis = {
                'final_return': (capital / self.initial_capital) - 1,
                'sharpe': 0,
                'max_drawdown': float(np.max(np.maximum.accumulate(equity_curve) - equity_curve)),
                'nb_trades': 1
            }
            return {'kpis': kpis, 'equity_curve': equity_curve}
        # Ajouter d'autres benchmarks ici
        return None

    def run(self) -> tuple[pd.DataFrame, list, dict, dict]:
        """
        Exécute le backtest avec application des coûts dynamiques si fournis.
        Retourne :
            - out_df : DataFrame enrichi (equity, returns)
            - trades : liste des trades exécutés
            - kpis : dictionnaire des métriques post-trade (Sharpe, drawdown, profit factor, win rate, expectancy, max losing streak, etc.)
            - benchmarks_results : résultats des benchmarks (si demandés)
        """
        df = self.df.copy()
        if self.model is not None:
            features = [c for c in df.columns if c not in [self.signal_col, self.price_col, self.label_col]]
            df[self.signal_col] = self.model.predict(df[features])
        capital = self.initial_capital
        position = 0
        equity_curve = []
        returns = []
        trades = []
        rejected_orders = []
        last_price = None
        entry_price = None
        for i, row in df.iterrows():
            price = row[self.price_col]
            signal = row[self.signal_col]
            timestamp = row.name
            proba = None
            if self.model is not None and hasattr(self.model, "predict_proba"):
                features_row = row.drop([self.signal_col, self.price_col, self.label_col], errors="ignore").values.reshape(1, -1)
                proba = self.model.predict_proba(features_row)[0].max()
            # --- Exécution adaptative si scheduler fourni ---
            if self.scheduler is not None and proba is not None:
                res = execute_adaptive_trade(
                    scheduler=self.scheduler,
                    symbol="BTCUSD",  # TODO: rendre générique
                    signal=signal,
                    proba=proba,
                    client=self.client
                )
                if res["success"]:
                    trades.append(res)
                    # TODO: maj capital selon PnL réel
            else:
                # Exécution classique (simulation)
                if signal != 0:
                    if position != 0:
                        pnl = (price - last_price) * position - abs(position) * price * (self.fee + self.slippage)
                        capital += pnl
                        trades.append({"entry": last_price, "exit": price, "side": np.sign(position), "pnl": pnl, "date": i})
                    position = signal
                    last_price = price
            equity_curve.append(capital)
        if position != 0 and last_price is not None:
            price = df.iloc[-1][self.price_col]
            timestamp = df.index[-1]
            spread = 0.0
            if self.spread_series is not None:
                if callable(self.spread_series):
                    spread = self.spread_series(timestamp)
                elif hasattr(self.spread_series, 'get'):
                    spread = self.spread_series.get(timestamp, 0.0)
                elif isinstance(self.spread_series, pd.Series):
                    spread = self.spread_series.loc[timestamp] if timestamp in self.spread_series.index else 0.0
            exec_price = price + spread/2 if position > 0 else price - spread/2
            slippage_val = self.slippage
            if self.slippage_fn is not None:
                orderbook = None
                if self.orderbook_series is not None:
                    if hasattr(self.orderbook_series, 'get'):
                        orderbook = self.orderbook_series.get(timestamp, None)
                    elif isinstance(self.orderbook_series, pd.Series):
                        orderbook = self.orderbook_series.loc[timestamp] if timestamp in self.orderbook_series.index else None
                slippage_val = self.slippage_fn(price, 1.0, orderbook, timestamp)
            fee_val = self.fee
            if self.fee_fn is not None:
                fee_val = self.fee_fn(price, 1.0, timestamp)
            pnl = (exec_price - last_price) * position - abs(position) * exec_price * (fee_val + slippage_val)
            capital += pnl
            returns.append(pnl / self.initial_capital)
            trades.append({"entry": last_price, "exit": exec_price, "side": np.sign(position), "pnl": pnl, "date": df.index[-1]})
            equity_curve[-1] = capital
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)
        pnl = capital - self.initial_capital
        max_drawdown = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve))
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) > 1 else 0
        nb_trades = len(trades)
        win_trades = [t for t in trades if t["pnl"] > 0]
        loss_trades = [t for t in trades if t["pnl"] < 0]
        profit_factor = float(sum(t["pnl"] for t in win_trades) / abs(sum(t["pnl"] for t in loss_trades))) if loss_trades else float('inf')
        win_rate = float(len(win_trades) / nb_trades) if nb_trades > 0 else 0
        final_return = float((capital / self.initial_capital) - 1)
        expectancy = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0
        max_losing_streak = 0
        current_streak = 0
        for t in trades:
            if t["pnl"] < 0:
                current_streak += 1
                if current_streak > max_losing_streak:
                    max_losing_streak = current_streak
            else:
                current_streak = 0
        kpis = {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "nb_trades": nb_trades,
            "final_return": final_return,
            "final_capital": capital,
            "expectancy": expectancy,
            "max_losing_streak": max_losing_streak
        }
        out_df = df.copy()
        out_df["equity_curve"] = equity_curve / self.initial_capital
        out_df["returns"] = np.append([0], np.diff(equity_curve) / self.initial_capital)
        # --- Benchmarks ---
        benchmarks_results = {}
        for b in self.benchmarks:
            res = self._run_benchmark(b, df)
            if res is not None:
                benchmarks_results[b] = res
        # Reporting fichiers (inchangé)
        pd.DataFrame(trades).to_csv(os.path.join(self.out_dir, "trades.csv"), index=False)
        pd.DataFrame(rejected_orders).to_csv(os.path.join(self.out_dir, "rejected_orders.csv"), index=False)
        pd.DataFrame({"equity": equity_curve}, index=df.index).to_csv(os.path.join(self.out_dir, "equity_curve.csv"))
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(kpis, f, indent=2)
        with open(os.path.join(self.out_dir, "benchmarks.json"), "w") as f:
            json.dump({k: v['kpis'] for k, v in benchmarks_results.items()}, f, indent=2)
        plt.figure(figsize=(8,4))
        plt.plot(df.index, equity_curve)
        plt.title("Equity curve")
        plt.xlabel("Date")
        plt.ylabel("Capital")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "equity_curve.png"))
        plt.close()
        drawdown = np.maximum.accumulate(equity_curve) - equity_curve
        plt.figure(figsize=(8,4))
        plt.plot(df.index, drawdown)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "drawdown.png"))
        plt.close()
        if self.label_col is not None and self.label_col in df.columns:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(df[self.label_col], df[self.signal_col], labels=[-1,0,1])
            plt.figure(figsize=(4,4))
            plt.imshow(cm, cmap="Blues")
            plt.title("Confusion matrix (backtest)")
            plt.xlabel("Signal")
            plt.ylabel("Label")
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "confusion_backtest.png"))
            plt.close()
        logger.info(f"Rapport backtest exporté : {os.path.join(self.out_dir, 'backtest_report.json')}")
        if self.benchmarks:
            return out_df, trades, kpis, benchmarks_results
        else:
            return out_df, trades, kpis 