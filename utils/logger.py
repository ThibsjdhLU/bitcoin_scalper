"""
Module de logging centralisé pour le bot de trading.
Utilise loguru pour une gestion avancée des logs.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

def setup_logger(config_path: str = "config/config.json") -> None:
    """
    Configure le système de logging avec les paramètres du fichier de configuration.
    
    Args:
        config_path: Chemin vers le fichier de configuration
    """
    # Charger la configuration
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = json.load(f)
    
    log_config = config['logging']
    
    # Supprimer les handlers par défaut
    logger.remove()
    
    # Ajouter le handler console
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_config['level']
    )
    
    # Ajouter le handler fichier
    log_file = Path(log_config['file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        rotation=log_config['max_size'],
        retention=log_config['backup_count'],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_config['level']
    )

def get_logger():
    """
    Retourne l'instance du logger configuré.
    
    Returns:
        Logger: Instance du logger configuré
    """
    return logger 