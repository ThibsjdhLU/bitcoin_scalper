"""
Module de logging centralisé pour le bot de trading.
Utilise loguru pour une gestion avancée des logs.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Codes ANSI pour les couleurs
GREEN = "\033[92m"  # Vert clair
YELLOW = "\033[93m"  # Jaune clair
RESET = "\033[0m"  # Réinitialisation

from loguru import logger


def format_boolean(value: bool) -> str:
    """
    Formate une valeur booléenne avec des couleurs.

    Args:
        value: Valeur booléenne à formater

    Returns:
        str: Valeur booléenne formatée avec des couleurs
    """
    if value is True:
        return f"{GREEN}true{RESET}"
    elif value is False:
        return f"{YELLOW}false{RESET}"
    return str(value)


def setup_logger(config_path: str = "config/config.json") -> None:
    """
    Configure le système de logging avec les paramètres du fichier de configuration.

    Args:
        config_path: Chemin vers le fichier de configuration
    """
    # Charger la configuration
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    log_config = config.get(
        "logging",
        {
            "level": "INFO",
            "file": "logs/trading.log",
            "max_size": "100 MB",
            "backup_count": 10,
        },
    )

    # Supprimer les handlers par défaut
    logger.remove()

    # Ajouter le handler console avec un format concis
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <5} | {message}",
        level="INFO",  # Niveau INFO par défaut pour la console
        filter=lambda record: record["level"].name
        in ["INFO", "WARNING", "ERROR", "CRITICAL"],  # Filtrer les DEBUG
        colorize=True,
        enqueue=True,
    )

    # Ajouter le handler fichier avec plus de détails
    log_file = Path(log_config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        rotation=log_config["max_size"],
        retention=log_config["backup_count"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_config["level"],  # Utiliser le niveau configuré pour le fichier
        backtrace=True,  # Activer les backtraces pour les erreurs dans le fichier
        diagnose=True,  # Ajouter des informations de diagnostic pour les erreurs
    )


def get_logger():
    """
    Retourne l'instance du logger configuré.

    Returns:
        Logger: Instance du logger configuré
    """
    return logger
