import pandas as pd
import numpy as np
import logging
from typing import Any, Callable, Optional, Dict, List
import time
from datetime import datetime, timezone
from bitcoin_scalper.core.order_execution import execute_adaptive_trade
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter
from bitcoin_scalper.core.risk_management import RiskManager

logger = logging.getLogger("bitcoin_scalper.realtime")
logger.setLevel(logging.INFO)

class RealTimeExecutor:
    """
    Exécution temps réel ou simulation d'un modèle ML sur flux de données (live ou replay).
    Gère le portefeuille, les signaux, la latence, le reporting.
    Intègre désormais la gestion dynamique des SL/TP et le filtrage par entropie.
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
        client: Any = None,
        risk_manager: Optional[RiskManager] = None,
        trade_filter: Optional[TradeDecisionFilter] = None,
        atr_col: str = "atr_14", # Colonne ATR pour stop dynamique
        max_latency_ms: int = 200, # Max latency in milliseconds
        kill_switch_threshold: int = 5 # Number of consecutive stale quotes to trigger kill switch
    ):
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
        self.risk_manager = risk_manager
        self.trade_filter = trade_filter
        self.atr_col = atr_col
        self.max_latency_ms = max_latency_ms
        self.kill_switch_threshold = kill_switch_threshold
        self.consecutive_latency_errors = 0
        self._setup_reporting()

        # Initialize defaults if not provided
        if self.trade_filter is None:
            self.trade_filter = TradeDecisionFilter()

        # RiskManager is required for dynamic stops, but might be optional if passed via scheduler or just for simulation
        # If client is provided but risk_manager is not, we can instantiate it if we have the client.
        if self.risk_manager is None and self.client is not None:
             self.risk_manager = RiskManager(self.client)

    def _setup_reporting(self):
        import os
        os.makedirs(self.out_dir, exist_ok=True)

    def check_latency(self, tick_timestamp: pd.Timestamp) -> bool:
        """
        Vérifie la latence entre l'heure du tick et l'heure actuelle.
        Retourne False si la latence est trop élevée (Abort Trade).
        Lève une Exception si le compteur d'erreurs dépasse le seuil (Kill Switch).
        """
        if self.mode == "simulation":
            return True

        # Timestamp en UTC
        now = datetime.now(timezone.utc)

        # Ensure tick_timestamp is timezone-aware and UTC
        if tick_timestamp.tzinfo is None:
             # Assume UTC if naive, but log warning if critical
             tick_timestamp = tick_timestamp.replace(tzinfo=timezone.utc)
        else:
             tick_timestamp = tick_timestamp.astimezone(timezone.utc)

        delta_ms = (now - tick_timestamp).total_seconds() * 1000

        if delta_ms > self.max_latency_ms:
            self.consecutive_latency_errors += 1
            logger.warning(f"⚠️ Stale Quote detected! Latency: {delta_ms:.2f}ms > {self.max_latency_ms}ms. (Error count: {self.consecutive_latency_errors}/{self.kill_switch_threshold})")

            if self.consecutive_latency_errors >= self.kill_switch_threshold:
                logger.critical("🚨 KILL SWITCH TRIGGERED: Too many consecutive stale quotes. Stopping bot.")
                raise RuntimeError("Kill Switch Triggered: System Latency Too High")

            return False # Abort trade
        else:
            if self.consecutive_latency_errors > 0:
                logger.info("Latency back to normal.")
            self.consecutive_latency_errors = 0
            return True

    def step(self, row: pd.Series):
        # 1. Latency Check (PRIORITAIRE)
        if not self.check_latency(row.name):
             return # Abort processing for this tick

        price = row[self.price_col]

        # 2. Feature Engineering / Preparation
        exclude_cols = [self.signal_col, self.price_col, "target", "label", "open", "high", "low", "volume", "tickvol"]
        features_row = row.drop(exclude_cols, errors="ignore")
        # Convert Series to DataFrame to use select_dtypes, then reshape
        features_row_numeric = features_row.to_frame().T.select_dtypes(include=[np.number])
        features = features_row_numeric.values

        # 3. Prédiction ML
        # Obtenir les probabilités [prob_sell, prob_buy] ou [prob_down, prob_neutral, prob_up]
        # On suppose un modèle qui retourne predict_proba
        signal = 0
        proba = 0.5
        probs = None

        if hasattr(self.model, "predict_proba"):
            probs_array = self.model.predict_proba(features)
            probs = np.array(probs_array[0]) # Distribution complète pour entropie
            proba = probs.max()    # Confiance max
            predicted_class = self.model.predict(features)[0]
            # Map prediction to signal (-1, 0, 1) depending on model encoding
            # Assuming standard encoding: 0=Sell, 1=Buy OR 0=Hold, 1=Buy etc.
            # Needs alignment with labeling strategy. Assuming -1, 0, 1 output from predict or mapped.
            # If classifier returns 0,1,2 we need mapping.
            # For this context, let's assume predict returns the signed signal directly or we infer it.
            # If binary (0, 1): 0=Sell?? No usually 0=Hold?
            # Let's rely on model.predict returning the actionable signal directly for now
            # OR map it if we knew the labeling.
            signal = predicted_class
        else:
            # Fallback if no proba (e.g. some regressors or simple models)
            signal = self.model.predict(features)[0]
            probs = [0.5, 0.5] # Dummy

        # 4. Filtre Entropie
        if self.trade_filter:
            should_trade, reason = self.trade_filter.filter(proba, probs)
            if not should_trade:
                # logger.info(f"Trade aborted by Entropy Filter: {reason}")
                return

        # 5. Calcul Risk
        atr = row.get(self.atr_col, row.get("atr", 0.0))
        if atr == 0.0:
             logger.warning("ATR is 0 or missing, cannot calculate dynamic stops.")
             return

        stop_loss_price = None
        take_profit_price = None

        if signal != 0:
            side = 'buy' if signal > 0 else 'sell'

            if self.risk_manager:
                stops = self.risk_manager.calculate_dynamic_stops(price, atr, side, proba)
                stop_loss_price = stops['sl']
                take_profit_price = stops['tp']
            else:
                # Fallback logic if no RiskManager provided (e.g. pure simulation without complex logic)
                k_sl = 1.5
                k_tp = 3.0
                if side == 'buy':
                    stop_loss_price = price - (atr * k_sl)
                    take_profit_price = price + (atr * k_tp)
                else:
                    stop_loss_price = price + (atr * k_sl)
                    take_profit_price = price - (atr * k_tp)

        # 6. Exécution
        # Exécution via Scheduler (Live ou Simulation avancée)
        if self.scheduler is not None:
            if signal != 0:
                # Calculate SL distance for scheduler
                stop_loss_dist = abs(price - stop_loss_price) if stop_loss_price else None

                res = execute_adaptive_trade(
                    scheduler=self.scheduler,
                    symbol="BTCUSD",
                    signal=signal,
                    proba=proba,
                    client=self.client,
                    stop_loss=stop_loss_dist,
                    probs=probs
                )
                if res["success"]:
                    self.trades.append(res)
        else:
            # Exécution classique (Simulation basique interne)
            # This block updates internal PnL for backtesting/simulation mode
            if signal != 0:
                # Close opposite
                if self.position != 0 and np.sign(self.position) != np.sign(signal):
                     pnl = (price - self.last_price) * self.position - abs(self.position) * price * (self.fee + self.slippage)
                     self.capital += pnl
                     self.trades.append({"entry": self.last_price, "exit": price, "side": np.sign(self.position), "pnl": pnl, "date": row.name, "reason": "signal_reversal"})
                     self.position = 0

                # Open
                if self.position == 0:
                    self.position = signal
                    self.last_price = price
                    self.current_sl = stop_loss_price
                    self.current_tp = take_profit_price

            # Manage open position (SL/TP)
            elif self.position != 0:
                # Check SL/TP hits
                hit_sl = (self.position > 0 and price <= self.current_sl) or (self.position < 0 and price >= self.current_sl)
                hit_tp = (self.position > 0 and price >= self.current_tp) or (self.position < 0 and price <= self.current_tp)

                if hit_sl:
                    pnl = (self.current_sl - self.last_price) * self.position - abs(self.position) * self.current_sl * (self.fee + self.slippage)
                    self.capital += pnl
                    self.trades.append({"entry": self.last_price, "exit": self.current_sl, "side": np.sign(self.position), "pnl": pnl, "date": row.name, "reason": "SL"})
                    self.position = 0
                elif hit_tp:
                    pnl = (self.current_tp - self.last_price) * self.position - abs(self.position) * self.current_tp * (self.fee + self.slippage)
                    self.capital += pnl
                    self.trades.append({"entry": self.last_price, "exit": self.current_tp, "side": np.sign(self.position), "pnl": pnl, "date": row.name, "reason": "TP"})
                    self.position = 0

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
            try:
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
            except RuntimeError as e:
                # Catch Kill Switch
                logger.critical(f"Bot arrêté : {e}")
                self._finalize()
                break
            except Exception as e:
                logger.error(f"Erreur inattendue : {e}", exc_info=True)
                # Should we stop? Safe mode?
                time.sleep(5)

        self._finalize()

    def _finalize(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        if self.out_dir:
            try:
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
            except Exception as e:
                logger.error(f"Erreur lors du reporting: {e}")
