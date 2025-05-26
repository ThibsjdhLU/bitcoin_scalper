import pytest
from unittest.mock import MagicMock, patch
from app.core.data_ingestor import DataIngestor
import time

def make_tick(ts):
    return {"symbol": "BTCUSD", "bid": 1, "ask": 2, "timestamp": ts, "volume": 0.1}

def make_ohlcv(ts):
    return {"symbol": "BTCUSD", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": ts, "timeframe": "M1"}

class DummyMT5:
    def __init__(self):
        self.ticks = []
        self.ohlcv = []
    def get_ticks(self, symbol, limit=100):
        return self.ticks
    def get_ohlcv(self, symbol, timeframe="M1", limit=100):
        return self.ohlcv

class DummyDB:
    def __init__(self):
        self.ticks = []
        self.ohlcv = []
    def insert_ticks(self, ticks):
        self.ticks.extend(ticks)
    def insert_ohlcv(self, ohlcv):
        self.ohlcv.extend(ohlcv)

def test_ingest_ticks_and_ohlcv():
    mt5 = DummyMT5()
    db = DummyDB()
    mt5.ticks = [make_tick("2024-06-01T12:00:00Z"), make_tick("2024-06-01T12:00:01Z")]
    mt5.ohlcv = [make_ohlcv("2024-06-01T12:00:00Z")]
    ingestor = DataIngestor(mt5, db, batch_size=10)
    ingestor._ingest_ticks()
    ingestor._ingest_ohlcv()
    assert len(db.ticks) == 2
    assert len(db.ohlcv) == 1
    # Test filtrage timestamp
    mt5.ticks = [make_tick("2024-06-01T12:00:01Z"), make_tick("2024-06-01T12:00:02Z")]
    ingestor._ingest_ticks()
    assert len(db.ticks) == 3  # Un seul tick nouveau

def test_ingest_error_handling():
    mt5 = DummyMT5()
    db = DummyDB()
    mt5.get_ticks = MagicMock(side_effect=Exception("fail"))
    mt5.get_ohlcv = MagicMock(side_effect=Exception("fail"))
    ingestor = DataIngestor(mt5, db)
    # Ne doit pas lever d'exception
    ingestor._ingest_ticks()
    ingestor._ingest_ohlcv()

def test_start_stop_thread():
    mt5 = DummyMT5()
    db = DummyDB()
    ingestor = DataIngestor(mt5, db, poll_interval=0.1)
    ingestor.start()
    time.sleep(0.2)
    ingestor.stop()
    assert not ingestor._thread.is_alive() 