import pandas as pd
import numpy as np
import logging
from typing import Any, Callable, Optional, Dict, List
import time
from datetime import datetime, timezone
from bitcoin_scalper.core.order_execution import execute_adaptive_trade

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
        atr_col: str = "atr_14", # Colonne ATR pour stop dynamique
        max_latency_ms: int = 200, # Max latency in milliseconds
        kill_switch_threshold: int = 5 # Number of consecutive stale quotes to trigger kill switch
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
        :param atr_col: Nom de la colonne ATR pour les stops dynamiques
        :param max_latency_ms: Latence maximale autorisée en ms (défaut 200ms)
        :param kill_switch_threshold: Nombre d'erreurs consécutives avant kill switch (défaut 5)
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
        self.atr_col = atr_col
        self.max_latency_ms = max_latency_ms
        self.kill_switch_threshold = kill_switch_threshold
        self.consecutive_latency_errors = 0
        self._setup_reporting()

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
        # 1. Check Latency
        if not self.check_latency(row.name):
             return # Abort processing for this tick

        price = row[self.price_col]

        # Préparation des features (exclusion des colonnes cibles/prix)
        exclude_cols = [self.signal_col, self.price_col, "target", "label", "open", "high", "low", "volume", "tickvol"]
        features_row = row.drop(exclude_cols, errors="ignore")
        # On garde les features numériques seulement
        features_row = features_row.select_dtypes(include=[np.number])
        features = features_row.values.reshape(1, -1)

        # Scaling if model expects it (handled inside model pipeline usually)

        # Prédiction
        signal = self.model.predict(features)[0]

        proba = None
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs_array = self.model.predict_proba(features)
            probs = probs_array[0] # Distribution complète pour entropie
            proba = probs.max()    # Confiance max

        # Récupération ATR pour stop dynamique
        atr = row.get(self.atr_col, row.get("atr", 0.0))

        # Calcul Stop Loss Dynamique (distance)
        stop_loss_dist = None
        if signal != 0:
            side = 'buy' if signal == 1 else 'sell'

            # Utilisation du RiskManager du scheduler s'il existe
            rm = None
            if self.scheduler is not None and hasattr(self.scheduler, 'risk_manager'):
                rm = self.scheduler.risk_manager

            # Sinon utilisation d'une logique locale si nécessaire (simu basique)

            if rm:
                dyn_stops = rm.calculate_dynamic_stops(price, atr, side, proba if proba else 0.5)
                # On convertit en distance pour le scheduler/sizing
                stop_loss_dist = abs(price - dyn_stops['sl'])
            else:
                # Fallback basic logic
                k_sl = 2.0 if (proba and proba > 0.8) else 1.5
                stop_loss_dist = atr * k_sl

        # Exécution via Scheduler (Live ou Simulation avancée)
        if self.scheduler is not None:
            # On passe probs et stop_loss_dist au scheduler via execute_adaptive_trade
            if signal != 0:
                res = execute_adaptive_trade(
                    scheduler=self.scheduler,
                    symbol="BTCUSD",
                    signal=signal,
                    proba=proba if proba else 0.5,
                    client=self.client,
                    stop_loss=stop_loss_dist,
                    probs=probs # Pour l'Entropie
                )
                if res["success"]:
                    self.trades.append(res)
                    # En simulation sans client réel, on ne maj pas self.capital ici car execute_adaptive_trade ne simule pas le fill
                    # Il faudrait simuler l'ordre.
                    # TODO: Améliorer la simu avec scheduler.
        else:
            # Exécution classique (Simulation basique interne)
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

                    # Store params for simulation
                    side = 'buy' if signal == 1 else 'sell'
                    self.current_sl_dist = stop_loss_dist
                    self.current_tp_dist = stop_loss_dist * 2.0 # Default R:R 2

                    self.current_sl = price - stop_loss_dist if side == 'buy' else price + stop_loss_dist
                    self.current_tp = price + self.current_tp_dist if side == 'buy' else price - self.current_tp_dist

            # Manage open position (SL/TP)
            elif self.position != 0:
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
