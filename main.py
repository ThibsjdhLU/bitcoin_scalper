#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bot de Scalping Crypto "Scalper Adaptatif BTC/USDT"
Point d'entrée principal de l'application

Ce script initialise et orchestre tous les composants du bot de scalping:
- Connexion API (MT5/AvaTrade)
- Indicateurs techniques
- Stratégie de trading
- Gestion des risques
- Interface utilisateur
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from queue import Queue
from typing import Optional
import MetaTrader5 as mt5
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QGuiApplication
from dotenv import load_dotenv

from utils.logger import logger
from config.scalper_config import DEFAULT_CONFIG
from app import ScalperBotApp
from styles import STYLE_SHEET

# Chargement des variables d'environnement
load_dotenv()

def initialize_mt5() -> bool:
    """Initialise la connexion avec MetaTrader 5."""
    try:
        logger.info("Tentative d'initialisation de MT5...")
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"Échec de l'initialisation de MT5. Code d'erreur: {error[0]}, Message: {error[1]}")
            return False
            
        # Vérification des informations de connexion
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Impossible de récupérer les informations du compte MT5")
            return False
            
        logger.info(f"MT5 initialisé avec succès - Compte: {account_info.login}, Serveur: {account_info.server}")
        logger.debug(f"Balance: {account_info.balance}, Equity: {account_info.equity}, Margin: {account_info.margin}")
        return True
        
    except Exception as e:
        logger.exception("Erreur critique lors de l'initialisation de MT5")
        return False

def shutdown_mt5():
    """Ferme la connexion avec MetaTrader 5."""
    try:
        mt5.shutdown()
        logger.info("MT5 arrêté avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt de MT5: {str(e)}")

def check_environment() -> bool:
    """Vérifie l'environnement d'exécution."""
    try:
        logger.info("Vérification de l'environnement...")
        
        # Vérification du dossier logs
        if not os.path.exists("logs"):
            os.makedirs("logs")
            logger.info("Dossier logs créé")
            
        # Vérification des variables d'environnement
        required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
            return False
            
        logger.info("Vérification de l'environnement terminée avec succès")
        return True
        
    except Exception as e:
        logger.exception("Erreur lors de la vérification de l'environnement")
        return False

def main():
    """Fonction principale."""
    try:
        logger.info("=== Démarrage de l'application Bitcoin Scalper ===")
        logger.info(f"Version Python: {sys.version}")
        logger.info(f"Répertoire de travail: {os.getcwd()}")
        
        # Vérification de l'environnement
        if not check_environment():
            logger.critical("Échec de la vérification de l'environnement")
            return
            
        # Initialisation de MT5
        if not initialize_mt5():
            logger.critical("Échec de l'initialisation de MT5")
            QMessageBox.critical(None, "Erreur", "Échec de l'initialisation de MT5. Veuillez vérifier les logs pour plus de détails.")
            return
            
        # Configuration de Qt pour le scaling haute résolution
        if sys.platform == 'win32':
            logger.debug("Configuration de la mise à l'échelle pour Windows")
            os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        
        # Création de l'application Qt
        logger.info("Initialisation de l'interface graphique...")
        app = QApplication(sys.argv)
        
        # Configuration du style
        app.setStyleSheet(STYLE_SHEET)
        logger.debug("Style appliqué à l'application")
        
        # Création et affichage de la fenêtre principale
        logger.info("Création de la fenêtre principale...")
        window = ScalperBotApp()
        
        if not window:
            logger.critical("Échec de la création de la fenêtre principale")
            QMessageBox.critical(None, "Erreur", "Échec de la création de la fenêtre principale")
            return
            
        logger.info("Affichage de la fenêtre principale")
        window.show()
        
        # Exécution de l'application
        logger.info("Démarrage de la boucle d'événements Qt")
        exit_code = app.exec()
        
        # Nettoyage avant de quitter
        logger.info("Nettoyage des ressources...")
        if hasattr(window, 'stop_bot'):
            window.stop_bot()
        shutdown_mt5()
        
        logger.info(f"Application terminée avec le code de sortie: {exit_code}")
        sys.exit(exit_code)
            
    except Exception as e:
        logger.exception("Erreur critique dans la fonction principale")
        QMessageBox.critical(None, "Erreur Critique", str(e))
        sys.exit(1)
    finally:
        shutdown_mt5()
        logger.info("=== Fin de l'application Bitcoin Scalper ===")

if __name__ == "__main__":
    main()