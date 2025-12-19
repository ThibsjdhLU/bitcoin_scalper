import argparse
import pandas as pd
import numpy as np
import logging
import time
import sys
import random
from datetime import timedelta, datetime, timezone
from bitcoin_scalper.core.realtime import RealTimeExecutor
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SimulationReplay")

class MockModel:
    """Mock model generating random signals and probabilities."""
    def predict(self, X):
        # 10% buy, 10% sell, 80% hold
        rand = random.random()
        if rand < 0.1: return [1]
        if rand < 0.2: return [-1]
        return [0]

    def predict_proba(self, X):
        # Generate varied probabilities to test entropy filter
        # Format: [[prob_sell, prob_neutral, prob_buy]] if 3 classes?
        # Or binary [[prob_0, prob_1]]?
        # Assuming RealTimeExecutor expects proba = probs.max()
        # Let's generate a distribution
        p1 = random.random()
        p2 = random.random() * (1 - p1)
        p3 = 1 - p1 - p2
        return np.array([[p1, p2, p3]])

class MockClient:
    """Mock MT5 Client for RiskManager."""
    def __init__(self):
        self.balance = 10000.0
        self.equity = 10000.0

    def _request(self, method, endpoint, params=None, data=None):
        if endpoint == "/account":
            return {"balance": self.balance, "equity": self.equity}
        if endpoint.startswith("/symbol/"):
            return {"tick_value": 1.0, "ask": 20000.0, "bid": 19999.0}
        return {}

class SimulationExecutor(RealTimeExecutor):
    """
    Subclass of RealTimeExecutor for Replay Simulation.
    Overrides latency check to simulate artificial latency.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.artificial_latency_ms = 0.0

    def check_latency(self, tick_timestamp: pd.Timestamp) -> bool:
        """
        Overrides check_latency to use injected artificial latency
        instead of real wall-clock time.
        """
        if self.artificial_latency_ms > self.max_latency_ms:
            self.consecutive_latency_errors += 1
            logger.warning(f"⚠️ [SIM] Latency Check: Artificially injected {self.artificial_latency_ms}ms > {self.max_latency_ms}ms. Warning!")

            if self.consecutive_latency_errors >= self.kill_switch_threshold:
                logger.critical("🚨 [SIM] KILL SWITCH TRIGGERED (Simulation): Too many consecutive stale quotes.")
                raise RuntimeError("Kill Switch Triggered (Simulated)")
            return False

        if self.consecutive_latency_errors > 0:
            logger.info("[SIM] Latency back to normal.")
        self.consecutive_latency_errors = 0
        return True

def run_simulation(csv_path: str):
    logger.info(f"Starting Simulation Replay with {csv_path}...")

    # Load Data
    try:
        df = pd.read_csv(csv_path)
        # Ensure timestamp is datetime and index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
        elif '<DATE>' in df.columns and '<TIME>' in df.columns:
             # MT5 format handling
             df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
             df.set_index('timestamp', inplace=True)
             # Rename columns for compatibility if needed (RealTimeExecutor expects specific names)
             df.rename(columns={'<CLOSE>': 'close', '<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<TICKVOL>': 'tickvol', '<VOL>': 'volume'}, inplace=True)

        # Ensure ATR exists (required for RiskManager)
        if 'atr_14' not in df.columns:
            # Quick calc ATR if missing
            logger.info("Calculating ATR...")
            high = df['high']
            low = df['low']
            close = df['close']
            tr = np.maximum(high - low, np.abs(high - close.shift(1)))
            tr = np.maximum(tr, np.abs(low - close.shift(1)))
            df['atr_14'] = tr.rolling(14).mean().bfill()

    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    # Setup Components
    model = MockModel()
    client = MockClient()
    risk_manager = RiskManager(client=client)
    trade_filter = TradeDecisionFilter()

    executor = SimulationExecutor(
        model=model,
        data_source=lambda: None, # Not used in manual step loop
        signal_col='signal', # Dummy
        price_col='close',
        risk_manager=risk_manager,
        trade_filter=trade_filter,
        client=client,
        mode="simulation",
        max_latency_ms=200,
        kill_switch_threshold=3
    )

    # Simulation Loop
    logger.info("--- Beginning Replay Loop ---")

    for i, (index, row) in enumerate(df.iterrows()):
        row.name = index # Ensure timestamp is attached

        # Inject artificial latency randomly
        # 5% chance of high latency
        is_laggy = random.random() < 0.05
        executor.artificial_latency_ms = random.randint(250, 500) if is_laggy else random.randint(10, 100)

        try:
            executor.step(row)
        except RuntimeError as e:
            logger.critical(f"Simulation stopped by Kill Switch: {e}")
            break

        # Limit output for sanity if file is huge
        # if i > 100: break

    logger.info("--- Simulation Complete ---")
    logger.info(f"Total Trades Simulated: {len(executor.trades)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dry-run simulation replay.")
    parser.add_argument("--csv", type=str, required=True, help="Path to historical CSV data.")
    args = parser.parse_args()

    run_simulation(args.csv)
