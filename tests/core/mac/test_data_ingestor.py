import pytest
from unittest.mock import MagicMock, patch
from bitcoin_scalper.core.data_ingestor import DataIngestor
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

@pytest.fixture
def ingestor():
    mt5 = MagicMock()
    db = MagicMock()
    cleaner = MagicMock()
    return DataIngestor(mt5, db, symbol="BTCUSD", timeframe="1min", cleaner=cleaner)

def test_instanciation(ingestor):
    assert hasattr(ingestor, "start")
    assert hasattr(ingestor, "stop")
    assert hasattr(ingestor, "_run")

def test_start_stop(ingestor):
    with patch.object(ingestor, "_run", return_value=None):
        ingestor.start()
        ingestor.stop()
        assert not ingestor._thread.is_alive()

def test_ingest_ohlcv_success(ingestor):
    ingestor.mt5_client.get_ohlcv.return_value = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 1, "timestamp": 1}]
    ingestor.cleaner.clean_ohlcv.return_value = [{"close": 100, "timestamp": 1}]
    ingestor.db_client.insert_ohlcv.return_value = True
    ingestor._last_ohlcv_time = None
    ingestor._ingest_ohlcv()
    ingestor.mt5_client.get_ohlcv.assert_called()
    ingestor.cleaner.clean_ohlcv.assert_called()
    ingestor.db_client.insert_ohlcv.assert_called()

def test_ingest_ohlcv_mt5_error(ingestor):
    ingestor.mt5_client.get_ohlcv.side_effect = Exception("MT5 error")
    ingestor._ingest_ohlcv()
    ingestor.mt5_client.get_ohlcv.assert_called()

def test_ingest_ohlcv_db_error(ingestor):
    ingestor.mt5_client.get_ohlcv.return_value = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 1, "timestamp": 1}]
    ingestor.cleaner.clean_ohlcv.return_value = [{"close": 100, "timestamp": 1}]
    ingestor.db_client.insert_ohlcv.side_effect = Exception("DB error")
    ingestor._last_ohlcv_time = None
    ingestor._ingest_ohlcv()
    ingestor.db_client.insert_ohlcv.assert_called()

def test_ingest_ticks_success(ingestor):
    ingestor.mt5_client.get_ticks.return_value = [{"timestamp": 1, "symbol": "BTCUSD"}]
    ingestor.cleaner.clean_ticks.return_value = [{"timestamp": 1, "symbol": "BTCUSD"}]
    ingestor.db_client.insert_ticks.return_value = True
    ingestor._last_tick_time = None
    ingestor._ingest_ticks()
    ingestor.mt5_client.get_ticks.assert_called()
    ingestor.cleaner.clean_ticks.assert_called()
    ingestor.db_client.insert_ticks.assert_called()

def test_ingest_ticks_mt5_error(ingestor):
    ingestor.mt5_client.get_ticks.side_effect = Exception("MT5 error")
    ingestor._ingest_ticks()
    ingestor.mt5_client.get_ticks.assert_called()

def test_ingest_ticks_db_error(ingestor):
    ingestor.mt5_client.get_ticks.return_value = [{"timestamp": 1, "symbol": "BTCUSD"}]
    ingestor.cleaner.clean_ticks.return_value = [{"timestamp": 1, "symbol": "BTCUSD"}]
    ingestor.db_client.insert_ticks.side_effect = Exception("DB error")
    ingestor._last_tick_time = None
    ingestor._ingest_ticks()
    ingestor.db_client.insert_ticks.assert_called() 