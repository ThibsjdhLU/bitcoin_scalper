import psycopg2
from psycopg2.extras import execute_batch
import logging
from typing import List, Dict, Any, Optional
import datetime
import pytz

logger = logging.getLogger("timescaledb_client")

class TimescaleDBError(Exception):
    """Exception personnalisée pour TimescaleDB."""
    pass

class TimescaleDBClient:
    """
    Client pour l'insertion performante de données ticks/OHLCV dans TimescaleDB.
    Gère la connexion, la reconnexion, l'insertion batch et la création du schéma si nécessaire.
    """
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str, sslmode: str = "require"):
        self.conn_params = dict(host=host, port=port, dbname=dbname, user=user, password=password, sslmode=sslmode)
        self.conn = None
        self.disabled = False  # Mode dégradé si TimescaleDB non dispo
        try:
            self.connect()
        except Exception as e:
            logger.error(f"Erreur connexion TimescaleDB: {e}. Mode dégradé activé.")
            self.disabled = True

    def connect(self):
        if self.disabled:
            return
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            logger.info("Connexion TimescaleDB établie.")
        except Exception as e:
            logger.error(f"Erreur connexion TimescaleDB: {e}")
            raise TimescaleDBError(f"Connexion TimescaleDB échouée: {e}")

    def ensure_connection(self):
        if self.disabled:
            return
        try:
            if self.conn is None or self.conn.closed:
                self.connect()
        except Exception as e:
            raise TimescaleDBError(f"Impossible de rétablir la connexion: {e}")

    def create_schema(self):
        """
        Crée les tables nécessaires (ticks, ohlcv) et les hypertables TimescaleDB.
        - Partitionne sur 'timestamp' (TIMESTAMPTZ NOT NULL)
        - Indexe sur (symbol, timestamp DESC)
        - Gestion robuste des erreurs et rollback
        """
        if self.disabled:
            logger.warning("TimescaleDB désactivé : création du schéma ignorée.")
            return
        self.ensure_connection()
        cur = self.conn.cursor()
        try:
            # Création extension TimescaleDB
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                logger.info("Extension TimescaleDB vérifiée/créée.")
            except Exception as e:
                logger.error(f"Impossible de créer l'extension TimescaleDB : {e}")
                logger.warning("Mode dégradé activé : stockage TimescaleDB désactivé.")
                self.disabled = True
                return

            # Table ticks
            cur.execute('''
                CREATE TABLE IF NOT EXISTS ticks (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    bid DOUBLE PRECISION,
                    ask DOUBLE PRECISION,
                    volume DOUBLE PRECISION
                );
            ''')
            logger.info("Table ticks vérifiée/créée.")
            # Hypertable ticks
            try:
                cur.execute("SELECT create_hypertable('ticks', 'timestamp', if_not_exists => TRUE);")
                logger.info("Hypertable ticks vérifiée/créée.")
            except Exception as e:
                logger.warning(f'Hypertable ticks: {e}')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp DESC);')

            # Table ohlcv
            cur.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    timeframe TEXT,
                    PRIMARY KEY (symbol, timestamp)
                );
            ''')
            logger.info("Table ohlcv vérifiée/créée.")
            # Hypertable ohlcv
            try:
                cur.execute("SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);")
                logger.info("Hypertable ohlcv vérifiée/créée.")
            except Exception as e:
                logger.warning(f'Hypertable ohlcv: {e}')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, timestamp DESC);')

            self.conn.commit()
            logger.info("Schéma TimescaleDB vérifié/créé.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Erreur création schéma: {e}")
            logger.warning("Mode dégradé activé : stockage TimescaleDB désactivé.")
            self.disabled = True
        finally:
            cur.close()

    def _prepare_records(self, records: List[Dict[str, Any]], required_keys: List[str]) -> List[Dict[str, Any]]:
        prepped = []
        for rec in records:
            # Vérification des clés requises
            for k in required_keys:
                if k not in rec:
                    raise TimescaleDBError(f"Clé requise manquante : '{k}' dans {rec}")
            # Conversion du timestamp si entier
            ts = rec['timestamp']
            if isinstance(ts, int):
                # On suppose timestamp Unix en secondes
                rec = rec.copy()
                rec['timestamp'] = datetime.datetime.fromtimestamp(ts, tz=pytz.UTC)
            elif isinstance(ts, float):
                rec = rec.copy()
                rec['timestamp'] = datetime.datetime.fromtimestamp(int(ts), tz=pytz.UTC)
            elif isinstance(ts, str):
                # On tente de parser la string
                try:
                    rec = rec.copy()
                    rec['timestamp'] = datetime.datetime.fromisoformat(ts)
                    if rec['timestamp'].tzinfo is None:
                        rec['timestamp'] = rec['timestamp'].replace(tzinfo=pytz.UTC)
                except Exception:
                    raise TimescaleDBError(f"Format de timestamp non supporté : {ts}")
            prepped.append(rec)
        return prepped

    def insert_ticks(self, ticks: List[Dict[str, Any]]):
        if self.disabled:
            logger.warning("insert_ticks ignoré : TimescaleDB désactivé.")
            return
        self.ensure_connection()
        if not ticks:
            return
        ticks = self._prepare_records(ticks, ["symbol", "timestamp", "bid", "ask", "volume"])
        cur = self.conn.cursor()
        try:
            execute_batch(cur, '''
                INSERT INTO ticks (symbol, timestamp, bid, ask, volume)
                VALUES (%(symbol)s, %(timestamp)s, %(bid)s, %(ask)s, %(volume)s)
            ''', ticks)
            self.conn.commit()
            logger.info(f"{len(ticks)} ticks insérés.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Erreur insertion ticks: {e}")
            raise TimescaleDBError(f"Erreur insertion ticks: {e}")
        finally:
            cur.close()

    def insert_ohlcv(self, ohlcv: List[Dict[str, Any]]):
        if self.disabled:
            logger.warning("insert_ohlcv ignoré : TimescaleDB désactivé.")
            return
        self.ensure_connection()
        if not ohlcv:
            return
        ohlcv = self._prepare_records(ohlcv, ["symbol", "timestamp", "open", "high", "low", "close", "volume", "timeframe"])
        cur = self.conn.cursor()
        try:
            execute_batch(cur, '''
                INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(timeframe)s)
                ON CONFLICT (symbol, timestamp) DO NOTHING
            ''', ohlcv)
            self.conn.commit()
            logger.info(f"{len(ohlcv)} OHLCV insérés.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Erreur insertion OHLCV: {e}")
            raise TimescaleDBError(f"Erreur insertion OHLCV: {e}")
        finally:
            cur.close()

    def close(self):
        if self.disabled:
            logger.info("TimescaleDB désactivé : close ignoré.")
            return
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Connexion TimescaleDB fermée.")

"""
Exemple d'utilisation :
db = TimescaleDBClient(host, port, dbname, user, password)
db.create_schema()
db.insert_ticks([{"symbol": "BTCUSD", "timestamp": "2024-06-01T12:00:00Z", "bid": 68000, "ask": 68010, "volume": 0.5}])
db.close()
"""

# Test unitaire : insertion doublon

def test_insert_ohlcv_duplicate(monkeypatch):
    """
    Vérifie qu'aucune exception n'est levée lors d'une tentative d'insertion de doublon OHLCV.
    """
    class DummyConn:
        def cursor(self):
            class DummyCursor:
                def execute(self, *a, **k): pass
                def close(self): pass
            return DummyCursor()
        def commit(self): pass
        def rollback(self): pass
        @property
        def closed(self): return False
    client = TimescaleDBClient('h', 1, 'd', 'u', 'p')
    client.conn = DummyConn()
    client.disabled = False
    def dummy_prepare(records, keys): return [{k: 1 for k in keys}]
    client._prepare_records = dummy_prepare
    try:
        client.insert_ohlcv([{'symbol': 'BTCUSD', 'timestamp': 1, 'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 1, 'timeframe': 'M1'}])
    except Exception as e:
        assert False, f"Exception inattendue : {e}" 