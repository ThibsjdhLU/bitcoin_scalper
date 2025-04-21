import os
from dotenv import load_dotenv
from mt5_connector import MT5Connector
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mt5_connection():
    # Chargement des variables d'environnement
    load_dotenv()
    
    # Création du connecteur MT5
    mt5_connector = MT5Connector()
    
    # Test d'initialisation
    logger.info("Test d'initialisation MT5...")
    if not mt5_connector.initialize():
        logger.error(f"Erreur d'initialisation: {mt5_connector.get_last_error()}")
        return False
    
    # Récupération des informations de connexion
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    # Test de connexion
    logger.info(f"Test de connexion au compte {login} sur {server}...")
    if not mt5_connector.login(login, password, server):
        logger.error(f"Erreur de connexion: {mt5_connector.get_last_error()}")
        return False
    
    # Récupération des informations du compte
    account_info = mt5_connector.get_account_info()
    if account_info:
        logger.info("Informations du compte:")
        for key, value in account_info.items():
            logger.info(f"{key}: {value}")
    
    # Test de récupération des symboles disponibles
    logger.info("Test de récupération des symboles disponibles...")
    symbols = mt5_connector.get_available_symbols()
    if symbols:
        logger.info(f"Nombre de symboles disponibles: {len(symbols)}")
        logger.info("Premiers symboles:")
        for symbol in symbols[:5]:
            logger.info(symbol)
    
    # Fermeture de la connexion
    mt5_connector.shutdown()
    logger.info("Test terminé avec succès")
    return True

if __name__ == "__main__":
    test_mt5_connection() 