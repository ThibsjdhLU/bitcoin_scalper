import pytest
import psycopg2
from bitcoin_scalper.core.timescaledb_client import TimescaleDBClient
import os
import time

def get_test_db_params():
    return dict(
        host=os.environ.get('TSDB_HOST', 'localhost'),
        port=int(os.environ.get('TSDB_PORT', 5432)),
        dbname=os.environ.get('TSDB_DB', 'bitcoin_scalper_test'),
        user=os.environ.get('TSDB_USER', 'postgres'),
        password=os.environ.get('TSDB_PASS', 'postgres'),
        sslmode=os.environ.get('TSDB_SSL', 'prefer'),
    )

@pytest.mark.integration
@pytest.mark.mac
@pytest.mark.skipif(
    not os.environ.get('RUN_TSDB_TESTS', '0') == '1',
    reason='Test TimescaleDB désactivé (RUN_TSDB_TESTS != 1)'
)
def test_ohlcv_schema():
    params = get_test_db_params()
    db = TimescaleDBClient(**params)
    db.create_schema()
    cur = db.conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ohlcv';")
    columns = [row[0] for row in cur.fetchall()]
    assert 'timestamp' in columns, "La colonne 'timestamp' doit exister dans ohlcv."
    cur.execute("SELECT kcu.column_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name WHERE tc.table_name = 'ohlcv' AND tc.constraint_type = 'PRIMARY KEY';")
    pk_cols = [row[0] for row in cur.fetchall()]
    assert set(pk_cols) == {'symbol', 'timestamp'}, f"La clé primaire doit être (symbol, timestamp), trouvé : {pk_cols}"
    cur.close()
    db.close()

def test_insert_ohlcv_unix_timestamp():
    params = get_test_db_params()
    db = TimescaleDBClient(**params)
    db.create_schema()
    now = int(time.time())
    ohlcv = [{
        'symbol': 'TEST',
        'timestamp': now,
        'open': 1.0,
        'high': 2.0,
        'low': 0.5,
        'close': 1.5,
        'volume': 10.0,
        'timeframe': 'M1'
    }]
    try:
        db.insert_ohlcv(ohlcv)
    except Exception as e:
        pytest.fail(f"Insertion OHLCV avec timestamp Unix a échoué : {e}")
    db.close()

def test_insert_ohlcv_missing_key():
    params = get_test_db_params()
    db = TimescaleDBClient(**params)
    db.create_schema()
    ohlcv = [{
        'symbol': 'TEST',
        'timestamp': int(time.time()),
        'open': 1.0,
        'high': 2.0,
        'low': 0.5,
        'close': 1.5,
        'volume': 10.0
        # 'timeframe' manquant
    }]
    with pytest.raises(Exception):
        db.insert_ohlcv(ohlcv)
    db.close() 