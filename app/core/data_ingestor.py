import threading
import time
import logging
from typing import Optional, List, Dict, Any
from bot.connectors.mt5_rest_client import MT5RestClient
from app.core.timescaledb_client import TimescaleDBClient
from app.core.data_cleaner import DataCleaner

logger = logging.getLogger("data_ingestor")

class DataIngestor:
    """
    Pipeline d'ingestion temps réel des données ticks et OHLCV depuis un serveur MT5 REST,
    avec nettoyage avancé (optionnel) et insertion batch dans TimescaleDB.
    Gère la robustesse réseau, le batching et la qualité des données.
    """
    def __init__(self, mt5_client: MT5RestClient, db_client: TimescaleDBClient, symbol: str = "BTCUSD", timeframe: str = "M1", batch_size: int = 100, poll_interval: float = 2.0, cleaner: Optional[DataCleaner] = None):
        self.mt5_client = mt5_client
        self.db_client = db_client
        self.symbol = symbol
        self.timeframe = timeframe
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.cleaner = cleaner
        self._stop_event = threading.Event()
        self._thread = None
        self._last_tick_time = None
        self._last_ohlcv_time = None

    def start(self):
        """Démarre l'ingestion en tâche de fond."""
        if self._thread and self._thread.is_alive():
            logger.warning("Ingestion déjà en cours.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Ingestion temps réel démarrée.")

    def stop(self):
        """Arrête l'ingestion."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        logger.info("Ingestion temps réel arrêtée.")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._ingest_ticks()
                self._ingest_ohlcv()
            except Exception as e:
                logger.error(f"Erreur ingestion : {e}")
            time.sleep(self.poll_interval)

    def _ingest_ticks(self):
        try:
            ticks = self.mt5_client.get_ticks(self.symbol, limit=self.batch_size)
            if self._last_tick_time:
                ticks = [t for t in ticks if t["timestamp"] > self._last_tick_time]
            if self.cleaner and ticks:
                ticks = self.cleaner.clean_ticks(ticks)
            if ticks:
                self.db_client.insert_ticks(ticks)
                self._last_tick_time = max(t["timestamp"] for t in ticks)
                logger.info(f"{len(ticks)} ticks ingérés (nettoyés)." if self.cleaner else f"{len(ticks)} ticks ingérés.")
        except Exception as e:
            logger.warning(f"Erreur ingestion ticks : {e}")

    def _ingest_ohlcv(self):
        try:
            ohlcv = self.mt5_client.get_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.batch_size)
            if self._last_ohlcv_time:
                ohlcv = [o for o in ohlcv if o["timestamp"] > self._last_ohlcv_time]
            if self.cleaner and ohlcv:
                ohlcv = self.cleaner.clean_ohlcv(ohlcv)
            if ohlcv:
                self.db_client.insert_ohlcv(ohlcv)
                self._last_ohlcv_time = max(o["timestamp"] for o in ohlcv)
                logger.info(f"{len(ohlcv)} OHLCV ingérés (nettoyés)." if self.cleaner else f"{len(ohlcv)} OHLCV ingérés.")
        except Exception as e:
            logger.warning(f"Erreur ingestion OHLCV : {e}")

"""
Exemple d'utilisation :
from app.core.data_cleaner import DataCleaner
mt5_client = MT5RestClient("https://mt5-server/api", api_key="cle")
db_client = TimescaleDBClient(...)
cleaner = DataCleaner()
ingestor = DataIngestor(mt5_client, db_client, cleaner=cleaner)
ingestor.start()
# ...
ingestor.stop()
""" 