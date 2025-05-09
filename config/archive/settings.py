#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gestion de la configuration
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


class Config:
    """Classe de gestion de la configuration"""

    def __init__(self, config_path: str = "config/.env"):
        """
        Initialise la configuration

        Args:
            config_path: Chemin du fichier de configuration
        """
        self.logger = logging.getLogger("Config")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier .env

        Returns:
            Dict: Configuration chargée
        """
        try:
            # Chargement des variables d'environnement
            load_dotenv(self.config_path)

            # Configuration par défaut
            config = {
                "mt5": {
                    "login": int(os.getenv("AVATRADE_LOGIN", 0)),
                    "password": os.getenv("AVATRADE_PASSWORD", ""),
                    "server": os.getenv("AVATRADE_SERVER", "Ava-Demo 1-MT5"),
                    "symbol": os.getenv("SYMBOL", "BTCUSD"),
                    "timeframe": os.getenv("TIMEFRAME", "1m"),
                },
                "trading": {
                    "volume_min": float(os.getenv("VOLUME_MIN", 0.01)),
                    "volume_max": float(os.getenv("VOLUME_MAX", 1.0)),
                    "risk_percent": float(os.getenv("RISK_PERCENT", 1.0)),
                },
                "logging": {
                    "level": os.getenv("LOG_LEVEL", "INFO"),
                    "file": os.getenv("LOG_FILE", "logs/bitcoin_scalper.log"),
                },
            }

            return config

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration

        Args:
            key: Clé de configuration
            default: Valeur par défaut

        Returns:
            Any: Valeur de configuration
        """
        try:
            keys = key.split(".")
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def save(self) -> bool:
        """
        Sauvegarde la configuration

        Returns:
            bool: True si sauvegarde réussie
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
