"""
Tests unitaires pour le module de logging.
"""
import json
import os
from pathlib import Path

import pytest
from loguru import logger

from src.bitcoin_scalper.utils.logger import get_logger, setup_logger


@pytest.fixture
def temp_config(tmp_path):
    """Crée un fichier de configuration temporaire pour les tests."""
    config = {
        "logging": {
            "level": "DEBUG",
            "file": str(tmp_path / "test.log"),
            "max_size": 1024,
            "backup_count": 2,
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


def test_setup_logger(temp_config):
    """Teste la configuration du logger."""
    # Vérifier que le fichier de log est créé
    setup_logger(temp_config)
    log_file = Path(json.load(open(temp_config))["logging"]["file"])
    assert log_file.parent.exists()

    # Vérifier que le logger est configuré
    logger_instance = get_logger()
    assert logger_instance is not None

    # Tester les différents niveaux de log
    logger_instance.debug("Test debug")
    logger_instance.info("Test info")
    logger_instance.warning("Test warning")
    logger_instance.error("Test error")

    # Vérifier que le fichier de log contient les messages
    assert log_file.exists()
    with open(log_file, "r") as f:
        content = f.read()
        assert "Test debug" in content
        assert "Test info" in content
        assert "Test warning" in content
        assert "Test error" in content


def test_get_logger():
    """Teste la récupération de l'instance du logger."""
    logger_instance = get_logger()
    assert logger_instance is not None
    assert isinstance(logger_instance, type(logger))
