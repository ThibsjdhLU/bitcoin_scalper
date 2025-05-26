import MetaTrader5 as mt5
import logging
import time
from app.core.config import SecureConfig, ConfigError

logger = logging.getLogger("mt5_connector")

class MT5ConnectorError(Exception):
    """Exception personnalisée pour le connecteur MT5."""
    pass

class MT5Connector:
    """
    Gère la connexion, reconnexion et déconnexion à MetaTrader5 de façon sécurisée et robuste.
    """
    def __init__(self, config: SecureConfig, max_retries: int = 3, retry_delay: float = 2.0):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connected = False

    def connect(self) -> bool:
        """Initialise la connexion à MT5 avec gestion des erreurs et reconnexion automatique."""
        for attempt in range(1, self.max_retries + 1):
            if mt5.initialize():
                login = self.config.get("mt5_login")
                password = self.config.get("mt5_password")
                server = self.config.get("mt5_server")
                if login is None or password is None or server is None:
                    raise MT5ConnectorError("Identifiants MT5 incomplets (login, password, server requis)")
                authorized = mt5.login(login=login, password=password, server=server)
                if authorized:
                    self.connected = True
                    logger.info("Connexion MT5 réussie.")
                    return True
                else:
                    logger.error(f"Échec login MT5 (tentative {attempt}): {mt5.last_error()}")
            else:
                logger.error(f"Échec initialisation MT5 (tentative {attempt}): {mt5.last_error()}")
            time.sleep(self.retry_delay)
        raise MT5ConnectorError("Impossible de se connecter à MetaTrader5 après plusieurs tentatives.")

    def disconnect(self):
        """Déconnecte proprement MT5."""
        mt5.shutdown()
        self.connected = False
        logger.info("Déconnexion MT5 effectuée.")

    def is_connected(self) -> bool:
        """Vérifie l'état de la connexion MT5."""
        return self.connected and mt5.terminal_info() is not None

    def ensure_connection(self):
        """Vérifie et rétablit la connexion si nécessaire."""
        if not self.is_connected():
            logger.warning("Connexion MT5 perdue, tentative de reconnexion...")
            self.connect()

"""
Exemple d'utilisation :
from app.core.config import SecureConfig
config = SecureConfig("/chemin/vers/config.enc", os.environ["CONFIG_AES_KEY"])
mt5c = MT5Connector(config)
mt5c.connect()
# ...
mt5c.disconnect()
""" 