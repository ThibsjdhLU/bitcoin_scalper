#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour tester la connexion à MetaTrader 5.
"""

import logging
import os
from datetime import datetime
import sys
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MT5Diagnostic")

def check_module_installation():
    """Vérifie si le module MetaTrader5 est installé."""
    logger.info("Vérification de l'installation du module MetaTrader5...")
    
    try:
        import MetaTrader5 as mt5
        logger.info(f"Module MetaTrader5 trouvé, version: {mt5.__version__}")
        return True
    except ImportError:
        logger.error("Le module MetaTrader5 n'est pas installé.")
        logger.info("Veuillez l'installer avec: pip install MetaTrader5")
        return False

def check_mt5_terminal():
    """Vérifie si le terminal MetaTrader 5 est installé."""
    logger.info("Vérification de l'installation du terminal MetaTrader 5...")
    
    # Chemins possibles pour l'installation de MT5
    possible_paths = [
        "C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe",
        "C:\\Program Files (x86)\\Ava Trade MT5 Terminal\\terminal.exe",
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"MetaTrader 5 trouvé à l'emplacement: {path}")
            return True
    
    logger.error("Terminal MetaTrader 5 non trouvé sur le système.")
    logger.info("Veuillez télécharger et installer MetaTrader 5 depuis https://www.avatrade.fr/trading-platforms/metatrader-5")
    return False

def test_mt5_connection(login, password, server):
    """Teste la connexion à MetaTrader 5."""
    logger.info(f"Test de connexion à MT5 avec le serveur: {server}")
    
    try:
        import MetaTrader5 as mt5
        
        # Convertir login en entier si nécessaire
        try:
            login = int(login)
        except (ValueError, TypeError):
            logger.error(f"Login invalide: {login}. Le login doit être un nombre entier.")
            return False
        
        # Arrêter MT5 s'il est déjà en cours d'exécution
        if mt5.terminal_info():
            mt5.shutdown()
            time.sleep(1)
        
        # Initialiser MT5
        logger.info("Initialisation de MT5...")
        if not mt5.initialize():
            error_code = mt5.last_error()
            logger.error(f"Échec de l'initialisation MT5: Code {error_code}")
            return False
        
        logger.info(f"MetaTrader 5 initialisé, version: {mt5.version()}")
        
        # Tentative de connexion
        logger.info(f"Tentative de connexion au serveur {server}...")
        authorized = mt5.login(login=login, password=password, server=server)
        
        if not authorized:
            error_code = mt5.last_error()
            logger.error(f"Échec de l'authentification MT5: Code {error_code}")
            
            # Afficher des informations supplémentaires sur l'erreur
            if error_code == 10000:
                logger.error("Erreur de connexion à MT5: aucune erreur retournée")
            elif error_code == 10013:
                logger.error("Erreur de connexion à MT5: identifiants invalides")
            elif error_code == 10014:
                logger.error("Erreur de connexion à MT5: IP ou serveur non autorisé")
            elif error_code == 10015:
                logger.error("Erreur de connexion à MT5: serveur invalide")
            elif error_code == 10016:
                logger.error("Erreur de connexion à MT5: trop de connexions")
            else:
                logger.error(f"Erreur de connexion à MT5: code {error_code}")
            
            mt5.shutdown()
            return False
        
        # Afficher les informations du compte
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connexion réussie!")
            logger.info(f"Compte: {account_info.login}, Nom: {account_info.name}")
            logger.info(f"Broker: {account_info.company}")
            logger.info(f"Solde: {account_info.balance}, Equity: {account_info.equity}")
            logger.info(f"Marge: {account_info.margin}, Marge libre: {account_info.margin_free}")
        
        # Vérifier les symboles disponibles
        symbols = mt5.symbols_get()
        if symbols:
            logger.info(f"Nombre de symboles disponibles: {len(symbols)}")
            crypto_symbols = [s.name for s in symbols if 'USD' in s.name and any(crypto in s.name for crypto in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH'])]
            logger.info(f"Symboles crypto disponibles: {crypto_symbols}")
        
        # Fermer MT5
        mt5.shutdown()
        return True
    
    except Exception as e:
        logger.error(f"Exception lors du test de connexion: {str(e)}")
        try:
            mt5.shutdown()
        except:
            pass
        return False

def load_config():
    """Charge la configuration."""
    logger.info("Chargement de la configuration...")
    
    try:
        # Essayer d'importer la configuration unifiée
        try:
            from config.unified_config import config
            login = config.get('exchange.login')
            password = config.get('exchange.password')
            server = config.get('exchange.server')
            logger.info("Configuration chargée depuis unified_config")
        except ImportError:
            # Essayer de charger depuis un fichier JSON
            import json
            try:
                with open('config/unified_config.json', 'r') as f:
                    cfg = json.load(f)
                    login = cfg['exchange']['login']
                    password = cfg['exchange']['password']
                    server = cfg['exchange']['server']
                    logger.info("Configuration chargée depuis unified_config.json")
            except:
                # Valeurs par défaut
                login = "101490774"
                password = "MatLB356&"
                server = "Ava-Demo 1-MT5"
                logger.info("Configuration par défaut utilisée")
        
        logger.info(f"Login: {login}, Serveur: {server}")
        return login, password, server
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return None, None, None

def main():
    """Fonction principale."""
    logger.info("-" * 60)
    logger.info("DIAGNOSTIC METATRADER 5")
    logger.info("-" * 60)
    
    # Vérifier l'installation du module
    if not check_module_installation():
        logger.info("Installation du module MetaTrader5...")
        os.system("pip install --no-cache-dir --upgrade MetaTrader5")
        
        # Vérifier à nouveau
        if not check_module_installation():
            logger.error("Impossible d'installer le module MetaTrader5. Diagnostic arrêté.")
            return
    
    # Vérifier l'installation du terminal
    if not check_mt5_terminal():
        logger.warning("Le terminal MetaTrader 5 n'est pas installé dans les emplacements standards.")
        logger.info("Le diagnostic va continuer, mais la connexion peut échouer.")
    
    # Charger la configuration
    login, password, server = load_config()
    if not login or not password or not server:
        logger.error("Impossible de charger la configuration. Diagnostic arrêté.")
        return
    
    # Tester la connexion
    if test_mt5_connection(login, password, server):
        logger.info("-" * 60)
        logger.info("✅ DIAGNOSTIC RÉUSSI: Connexion établie avec succès!")
        logger.info("-" * 60)
        
        # Lancer l'application
        logger.info("Vous pouvez maintenant lancer l'application avec: streamlit run app.py")
    else:
        logger.info("-" * 60)
        logger.error("❌ DIAGNOSTIC ÉCHOUÉ: Impossible d'établir la connexion à MT5")
        logger.info("-" * 60)
        
        # Suggestions
        logger.info("Suggestions:")
        logger.info("1. Vérifiez que vos identifiants sont corrects dans config/unified_config.json")
        logger.info("2. Vérifiez que le serveur spécifié existe et est accessible")
        logger.info("3. Assurez-vous que MetaTrader 5 est correctement installé")
        logger.info("4. Essayez de vous connecter manuellement via l'application MT5")
        logger.info("5. Vérifiez votre connexion Internet")

if __name__ == "__main__":
    main() 