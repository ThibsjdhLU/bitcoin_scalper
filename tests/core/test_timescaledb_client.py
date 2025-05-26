import pytest
from unittest.mock import patch, MagicMock
from app.core.timescaledb_client import TimescaleDBClient, TimescaleDBError

def get_db_params():
    return dict(host="localhost", port=5432, dbname="test", user="user", password="pw")

@patch("app.core.timescaledb_client.psycopg2.connect")
def test_connect_success(mock_connect):
    mock_connect.return_value = MagicMock(closed=0)
    db = TimescaleDBClient(**get_db_params())
    assert db.conn is not None
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect", side_effect=Exception("fail"))
def test_connect_fail(mock_connect):
    with pytest.raises(TimescaleDBError):
        TimescaleDBClient(**get_db_params())

@patch("app.core.timescaledb_client.psycopg2.connect")
def test_create_schema_success(mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.create_schema()
    assert cur.execute.called
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect")
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

@patch("app.core.timescaledb_client.psycopg2.connect")
@patch("app.core.timescaledb_client.execute_batch")
def test_insert_ticks_success(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.insert_ticks([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "bid": 1, "ask": 2, "volume": 0.1}])
    assert mock_batch.called
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect")
@patch("app.core.timescaledb_client.execute_batch", side_effect=Exception("fail"))
def test_insert_ticks_fail(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    with pytest.raises(TimescaleDBError):
        db.insert_ticks([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "bid": 1, "ask": 2, "volume": 0.1}])
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect")
@patch("app.core.timescaledb_client.execute_batch")
def test_insert_ohlcv_success(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.insert_ohlcv([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timeframe": "M1"}])
    assert mock_batch.called
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect")
@patch("app.core.timescaledb_client.execute_batch", side_effect=Exception("fail"))
def test_insert_ohlcv_fail(mock_batch, mock_connect):
    conn = MagicMock(closed=0)
    cur = MagicMock()
    conn.cursor.return_value = cur
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    with pytest.raises(TimescaleDBError):
        db.insert_ohlcv([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timeframe": "M1"}])
    db.close()

@patch("app.core.timescaledb_client.psycopg2.connect")
def test_close(mock_connect):
    conn = MagicMock(closed=0)
    mock_connect.return_value = conn
    db = TimescaleDBClient(**get_db_params())
    db.close()
    assert conn.close.called 