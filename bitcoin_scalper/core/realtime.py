import pandas as pd
import numpy as np
import logging
from typing import Any, Callable, Optional, Dict
import time
from bitcoin_scalper.core.order_execution import execute_adaptive_trade

logger = logging.getLogger("bitcoin_scalper.realtime")
logger.setLevel(logging.INFO)

class RealTimeExecutor:
    """
    Exécution temps réel ou simulation d'un modèle ML sur flux de données (live ou replay).
    Gère le portefeuille, les signaux, la latence, le reporting.
    """
    def __init__(
        self,
        model: Any,
        data_source: Callable[[], pd.DataFrame],
        signal_col: str = "signal",
        price_col: str = "close",
        initial_capital: float = 10000.0,
        fee: float = 0.0005,
        slippage: float = 0.0002,
        mode: str = "simulation",
        sleep_time: float = 1.0,
        out_dir: str = "realtime_reports",
        scheduler: Any = None,
        client: Any = None
    ):
        """
        :param model: modèle ML (doit avoir predict)
        :param data_source: fonction qui retourne un DataFrame (nouveaux ticks/bars)
        :param signal_col: colonne des signaux
        :param price_col: colonne des prix
        :param initial_capital: capital initial
        :param fee: frais de transaction
        :param slippage: slippage
        :param mode: "simulation" (replay) ou "live"
        :param sleep_time: temps d'attente entre deux ticks (simulation)
        :param out_dir: dossier de reporting
        :param scheduler: scheduler adaptatif
        :param client: client broker (pour live)
        """
        self.model = model
        self.data_source = data_source
        self.signal_col = signal_col
        self.price_col = price_col
        self.capital = initial_capital
        self.fee = fee
        self.slippage = slippage
        self.mode = mode
        self.sleep_time = sleep_time
        self.out_dir = out_dir
        self.position = 0
        self.last_price = None
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.scheduler = scheduler
        self.client = client
        self._setup_reporting()

    def _setup_reporting(self):
        import os
        os.makedirs(self.out_dir, exist_ok=True)

    def step(self, row: pd.Series):
        price = row[self.price_col]
        features = row.drop([self.signal_col, self.price_col], errors="ignore").values.reshape(1, -1)
        signal = self.model.predict(features)[0]
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0].max()
        # Exécution du trade via scheduler adaptatif si fourni
        if self.scheduler is not None and self.client is not None and proba is not None:
            res = execute_adaptive_trade(
                scheduler=self.scheduler,
                symbol="BTCUSD",  # TODO: rendre générique
                signal=signal,
                proba=proba,
                client=self.client
            )
            if res["success"]:
                self.trades.append(res)
                self.capital += 0  # TODO: maj capital selon PnL réel
        else:
            # Exécution classique (simulation)
            if signal != 0:
                if self.position != 0:
                    pnl = (price - self.last_price) * self.position - abs(self.position) * price * (self.fee + self.slippage)
                    self.capital += pnl
                    self.trades.append({"entry": self.last_price, "exit": price, "side": np.sign(self.position), "pnl": pnl, "date": row.name})
                self.position = signal
                self.last_price = price
        self.equity_curve.append(self.capital)
        self.timestamps.append(row.name)

    def run(self, max_steps: Optional[int] = None):
        """
        Lance l'exécution temps réel ou simulation.
        :param max_steps: nombre max de ticks à traiter (None = infini)
        """
        logger.info(f"Démarrage RealTimeExecutor mode={self.mode}")
        steps = 0
        while True:
            df = self.data_source()
            if df is None or df.empty:
                if self.mode == "simulation":
                    break
                else:
                    time.sleep(self.sleep_time)
                    continue
            for _, row in df.iterrows():
                self.step(row)
                steps += 1
                if max_steps and steps >= max_steps:
                    logger.info("Arrêt après max_steps")
                    self._finalize()
                    return
                if self.mode == "simulation":
                    time.sleep(self.sleep_time)
        self._finalize()

    def _finalize(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        pd.DataFrame(self.trades).to_csv(f"{self.out_dir}/trades.csv", index=False)
        pd.DataFrame({"equity": self.equity_curve}, index=self.timestamps).to_csv(f"{self.out_dir}/equity_curve.csv")
        plt.figure(figsize=(8,4))
        plt.plot(self.timestamps, self.equity_curve)
        plt.title("Equity curve (realtime)")
        plt.xlabel("Date")
        plt.ylabel("Capital")
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/equity_curve.png")
        plt.close()
        logger.info(f"Reporting realtime exporté dans {self.out_dir}") 