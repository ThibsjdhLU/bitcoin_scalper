"""
Configuration centralisée du bot de trading
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class Config:
    """Gestionnaire de configuration centralisé"""

    def __init__(self, config_dir: str = "config"):
        """
        Initialise la configuration

        Args:
            config_dir: Répertoire contenant les fichiers de configuration
        """
        self.config_dir = Path(config_dir)
        self._load_env()
        self._load_configs()

    def _load_env(self) -> None:
        """Charge les variables d'environnement"""
        env_path = self.config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    def _load_configs(self) -> None:
        """Charge tous les fichiers de configuration"""
        self.configs = {}

        # Chargement des fichiers JSON
        for config_file in self.config_dir.glob("*.json"):
            with open(config_file, "r") as f:
                self.configs[config_file.stem] = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration

        Args:
            key: Clé de configuration (format: 'section.key')
            default: Valeur par défaut si la clé n'existe pas

        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        # Vérification des variables d'environnement
        env_key = key.upper().replace(".", "_")
        if env_value := os.getenv(env_key):
            return env_value

        # Vérification des fichiers de configuration
        section, *subkeys = key.split(".")
        if section in self.configs:
            value = self.configs[section]
            for subkey in subkeys:
                if isinstance(value, dict) and subkey in value:
                    value = value[subkey]
                else:
                    return default
            return value

        return default

    def get_all(self) -> Dict[str, Any]:
        """
        Récupère toutes les configurations

        Returns:
            Dictionnaire contenant toutes les configurations
        """
        return self.configs
