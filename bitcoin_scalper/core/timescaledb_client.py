import psycopg2
from psycopg2.extras import execute_batch
import logging
from typing import List, Dict, Any, Optional

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
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            logger.info("Connexion TimescaleDB établie.")
        except Exception as e:
            logger.error(f"Erreur connexion TimescaleDB: {e}")
            raise TimescaleDBError(f"Connexion TimescaleDB échouée: {e}")

    def ensure_connection(self):
        try:
            if self.conn is None or self.conn.closed:
                self.connect()
        except Exception as e:
            raise TimescaleDBError(f"Impossible de rétablir la connexion: {e}")

    def create_schema(self):
        """Crée les tables nécessaires si elles n'existent pas et l'extension TimescaleDB si besoin."""
        self.ensure_connection()
        cur = self.conn.cursor()
        try:
            # Création extension TimescaleDB
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                logger.info("Extension TimescaleDB vérifiée/créée.")
            except Exception as e:
                logger.error(f"Impossible de créer l'extension TimescaleDB : {e}")
                raise TimescaleDBError("TimescaleDB n'est pas disponible sur cette base de données.\nDétail : " + str(e))

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
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    timeframe TEXT
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
            raise TimescaleDBError(f"Erreur création schéma: {e}")
        finally:
            cur.close()

    def insert_ticks(self, ticks: List[Dict[str, Any]]):
        """Insère une liste de ticks (dictionnaires) en batch."""
        self.ensure_connection()
        if not ticks:
            return
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
        """Insère une liste d'OHLCV (dictionnaires) en batch."""
        self.ensure_connection()
        if not ohlcv:
            return
        cur = self.conn.cursor()
        try:
            execute_batch(cur, '''
                INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(timeframe)s)
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