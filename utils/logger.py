#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de journalisation pour le bot de scalping
Gère la configuration et la rotation des logs
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

class DetailedLogger:
    """
    Gestionnaire de logging détaillé avec rotation des fichiers
    """
    def __init__(self, name: str = "bitcoin_scalper"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure les handlers et le format de logging"""
        # Création du dossier logs s'il n'existe pas
        os.makedirs("logs", exist_ok=True)
        
        # Format détaillé pour les logs
        detailed_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler pour le fichier de log principal avec rotation
        main_handler = logging.handlers.RotatingFileHandler(
            filename="logs/bitcoin_scalper.log",
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.DEBUG)
        
        # Handler pour les erreurs uniquement
        error_handler = logging.handlers.RotatingFileHandler(
            filename="logs/errors.log",
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Handler pour la console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Ajout des handlers au logger
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
    def debug(self, message: str, *args, **kwargs):
        """Log un message de niveau DEBUG"""
        self.logger.debug(message, *args, **kwargs)
        
    def info(self, message: str, *args, **kwargs):
        """Log un message de niveau INFO"""
        self.logger.info(message, *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs):
        """Log un message de niveau WARNING"""
        self.logger.warning(message, *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs):
        """Log un message de niveau ERROR"""
        self.logger.error(message, exc_info=True, *args, **kwargs)
        
    def critical(self, message: str, *args, **kwargs):
        """Log un message de niveau CRITICAL"""
        self.logger.critical(message, exc_info=True, *args, **kwargs)
        
    def exception(self, message: str, *args, **kwargs):
        """Log une exception avec stack trace"""
        self.logger.exception(message, *args, **kwargs)

# Instance globale du logger
logger = DetailedLogger()

def log_trade(logger, trade_info):
    """
    Fonction utilitaire pour loguer les informations de trade
    
    Args:
        logger (logging.Logger): Logger
        trade_info (dict): Informations sur le trade
    """
    logger.info(
        "TRADE %s: %s %.6f à %.2f | SL: %.2f | TP: %.2f",
        trade_info.get("order_id", ""),
        trade_info.get("type", "").upper(),
        trade_info.get("volume", 0),
        trade_info.get("price", 0),
        trade_info.get("sl", 0),
        trade_info.get("tp", 0)
    )