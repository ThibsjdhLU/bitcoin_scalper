import pytest
from unittest.mock import patch, MagicMock
from bitcoin_scalper.core.timescaledb_client import TimescaleDBClient, TimescaleDBError

def get_db_params():
    return dict(host="localhost", port=5432, dbname="test", user="user", password="pw")

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
def test_connect_success(mock_connect):
    mock_connect.return_value = MagicMock(closed=0)
    db = TimescaleDBClient(**get_db_params())
    assert db.conn is not None
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect", side_effect=Exception("fail"))
def test_connect_fail(mock_connect):
    with pytest.raises(TimescaleDBError):
        TimescaleDBClient(**get_db_params())

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
def test_create_schema_success(mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.create_schema()
    assert cur.execute.called
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
def test_create_schema_fail(mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    cur.execute.side_effect = Exception("fail")
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    with pytest.raises(TimescaleDBError):
        db.create_schema()
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
@patch("bitcoin_scalper.core.timescaledb_client.execute_batch")
def test_insert_ticks_success(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.insert_ticks([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "bid": 1, "ask": 2, "volume": 0.1}])
    assert mock_batch.called
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
@patch("bitcoin_scalper.core.timescaledb_client.execute_batch", side_effect=Exception("fail"))
def test_insert_ticks_fail(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    with pytest.raises(TimescaleDBError):
        db.insert_ticks([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "bid": 1, "ask": 2, "volume": 0.1}])
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
@patch("bitcoin_scalper.core.timescaledb_client.execute_batch")
def test_insert_ohlcv_success(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.insert_ohlcv([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timeframe": "M1"}])
    assert mock_batch.called
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
@patch("bitcoin_scalper.core.timescaledb_client.execute_batch", side_effect=Exception("fail"))
def test_insert_ohlcv_fail(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    with pytest.raises(TimescaleDBError):
        db.insert_ohlcv([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timeframe": "M1"}])
    db.close()

@patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect")
def test_close(mock_connect):
    conn = MagicMock(closed=0)
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.close()
    assert conn.close.called

@pytest.fixture
def db():
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        yield TimescaleDBClient(host="localhost", port=5432, dbname="test", user="user", password="pass")

def test_instanciation(db):
    assert hasattr(db, "insert_ticks")
    assert hasattr(db, "insert_ohlcv")
    assert hasattr(db, "create_schema")

def test_insert_ticks_success(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value = MagicMock()
    ticks = [{"symbol": "BTCUSD", "bid": 1, "ask": 2, "timestamp": 1, "volume": 0.1}]
    with patch("bitcoin_scalper.core.timescaledb_client.execute_batch") as mock_batch:
        db.insert_ticks(ticks)
        mock_batch.assert_called()

def test_insert_ohlcv_success(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value = MagicMock()
    ohlcv = [{"symbol": "BTCUSD", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": 1, "timeframe": "M1"}]
    with patch("bitcoin_scalper.core.timescaledb_client.execute_batch") as mock_batch:
        db.insert_ohlcv(ohlcv)
        mock_batch.assert_called()

def test_create_schema_success(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value = MagicMock()
    db.create_schema()
    db.conn.cursor.return_value.execute.assert_called()

def test_insert_ticks_empty(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value = MagicMock()
    db.insert_ticks([])  # Ne doit pas lever d'exception
    db.conn.cursor.return_value.execute.assert_not_called()

def test_insert_ohlcv_empty(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value = MagicMock()
    db.insert_ohlcv([])  # Ne doit pas lever d'exception
    db.conn.cursor.return_value.execute.assert_not_called()

def test_insert_ticks_db_error(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value.execute.side_effect = Exception("DB error")
    with pytest.raises(Exception):
        db.insert_ticks([{"symbol": "BTCUSD", "bid": 1, "ask": 2, "timestamp": 1, "volume": 0.1}])

def test_insert_ohlcv_db_error(db):
    db.conn = MagicMock()
    db.conn.cursor.return_value.execute.side_effect = Exception("DB error")
    with pytest.raises(Exception):
        db.insert_ohlcv([{"symbol": "BTCUSD", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": 1, "timeframe": "M1"}])

def test_connect_fail():
    with patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect", side_effect=Exception("fail")):
        db = TimescaleDBClient(host="localhost", port=5432, dbname="test", user="user", password="pass")
        assert db.disabled is True

def test_create_schema_fail():
    with patch("bitcoin_scalper.core.timescaledb_client.psycopg2.connect") as mock_connect:
        conn = MagicMock(closed=0)
        cur = MagicMock()
        cur.execute.side_effect = Exception("fail")
        conn.cursor.return_value = cur
        mock_connect.return_value = conn
        db = TimescaleDBClient(host="localhost", port=5432, dbname="test", user="user", password="pass")
        db.create_schema()
        assert db.disabled is True 