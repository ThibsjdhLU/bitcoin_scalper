"""
Tests unitaires pour le module MT5Connector.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.bitcoin_scalper.core.mt5_connector import MT5Connector


@pytest.fixture
def temp_config(tmp_path):
    """Crée un fichier de configuration temporaire pour les tests."""
    config = {
        "broker": {
            "mt5": {
                "server": "test-server",
                "login": "12345",
                "password": "test-password",
                "symbols": ["BTCUSD", "ETHUSD"],
            }
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def mock_mt5():
    """Mock de MetaTrader5 pour les tests."""
    with patch("core.mt5_connector.mt5") as mock:
        # Mock des fonctions de base
        mock.initialize.return_value = True
        mock.login.return_value = True
        mock.shutdown.return_value = True
        mock.last_error.return_value = "Test error"

        # Mock des symboles
        symbol_info = MagicMock()
        symbol_info.name = "BTCUSD"
        symbol_info.visible = True
        symbol_info.bid = 50000.0
        symbol_info.ask = 50001.0
        symbol_info.spread = 1
        symbol_info.volume_min = 0.01
        symbol_info.volume_max = 100.0
        symbol_info.digits = 2

        mock.symbol_info.return_value = symbol_info
        mock.symbol_select.return_value = True

        yield mock


def test_init(temp_config):
    """Teste l'initialisation du connecteur."""
    connector = MT5Connector(temp_config)
    assert connector.server == "test-server"
    assert connector.login == 12345
    assert connector.password == "test-password"
    assert connector.symbols == ["BTCUSD", "ETHUSD"]
    assert not connector.connected


def test_connect_success(mock_mt5, temp_config):
    """Teste une connexion réussie."""
    connector = MT5Connector(temp_config)
    assert connector.connect()
    assert connector.connected

    mock_mt5.initialize.assert_called_once()
    mock_mt5.login.assert_called_once_with(
        login=12345, password="test-password", server="test-server"
    )


def test_connect_failure(mock_mt5, temp_config):
    """Teste un échec de connexion."""
    mock_mt5.initialize.return_value = False
    connector = MT5Connector(temp_config)
    assert not connector.connect()
    assert not connector.connected


def test_verify_symbols(mock_mt5, temp_config):
    """Teste la vérification des symboles."""
    connector = MT5Connector(temp_config)
    assert connector._verify_symbols()

    # Test avec un symbole invalide
    mock_mt5.symbol_info.side_effect = (
        lambda symbol: None if symbol == "BTCUSD" else mock_mt5.symbol_info.return_value
    )
    assert not connector._verify_symbols()


def test_reconnect(mock_mt5, temp_config):
    """Teste la reconnection."""
    connector = MT5Connector(temp_config)

    # Premier appel : échec de l'initialisation
    mock_mt5.initialize.side_effect = [False, True]
    assert not connector.connect()

    # Deuxième appel : succès
    mock_mt5.initialize.side_effect = None
    mock_mt5.initialize.return_value = True
    assert connector.reconnect(max_attempts=2, delay=0)
    assert connector.connected


def test_disconnect(mock_mt5, temp_config):
    """Teste la déconnexion."""
    connector = MT5Connector(temp_config)
    connector.connected = True
    connector.disconnect()
    assert not connector.connected
    mock_mt5.shutdown.assert_called_once()


def test_get_symbol_info(mock_mt5, temp_config):
    """Teste la récupération des informations d'un symbole."""
    connector = MT5Connector(temp_config)
    connector.connected = True

    info = connector.get_symbol_info("BTCUSD")
    assert info is not None
    assert info["name"] == "BTCUSD"
    assert info["bid"] == 50000.0
    assert info["ask"] == 50001.0


def test_context_manager(mock_mt5, temp_config):
    """Teste l'utilisation du context manager."""
    with MT5Connector(temp_config) as connector:
        assert connector.connected
    assert not connector.connected
    mock_mt5.shutdown.assert_called_once()
