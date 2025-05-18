import MetaTrader5 as mt5
import logging
import time
from app.core.config import SecureConfig, ConfigError

class MT5ConnectionError(Exception):
    """Exception personnalisée pour la connexion MT5."""
    pass

class MT5Connector:
    """
    Gère la connexion à MetaTrader 5 avec reconnexion automatique et logs détaillés.
    """
    def __init__(self, config: SecureConfig, max_retries: int = 5, retry_delay: int = 3):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connected = False
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("MT5Connector")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def connect(self):
        retries = 0
        while retries < self.max_retries:
            try:
                login = int(self.config.get_encrypted('mt5_login'))
                password = self.config.get_encrypted('mt5_password')
                server = self.config.get('mt5_server')
                path = self.config.get('mt5_path')
                if not mt5.initialize(path):
                    raise MT5ConnectionError(f"Échec initialisation MT5: {mt5.last_error()}")
                authorized = mt5.login(login, password=password, server=server)
                if not authorized:
                    raise MT5ConnectionError(f"Échec login MT5: {mt5.last_error()}")
                self.connected = True
                self.logger.info("Connexion MT5 réussie.")
                return True
            except Exception as e:
                self.logger.error(f"Erreur connexion MT5: {e}")
                retries += 1
                time.sleep(self.retry_delay)
        self.logger.critical("Impossible de se connecter à MT5 après plusieurs tentatives.")
        self.connected = False
        return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.logger.info("Déconnexion MT5 effectuée.")
            self.connected = False

    def is_connected(self) -> bool:
        return self.connected and mt5.terminal_info() is not None 