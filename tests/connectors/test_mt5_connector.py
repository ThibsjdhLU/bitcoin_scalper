import pytest
from unittest.mock import patch, MagicMock
from bot.connectors.mt5_connector import MT5Connector, MT5ConnectorError
from app.core.config import SecureConfig

class DummyConfig:
    def get(self, key, default=None):
        return {"mt5_login": 1, "mt5_password": "pw", "mt5_server": "srv"}.get(key, default)

@patch("bot.connectors.mt5_connector.mt5")
def test_connect_success(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    config = DummyConfig()
    connector = MT5Connector(config)
    assert connector.connect() is True
    assert connector.connected

@patch("bot.connectors.mt5_connector.mt5")
def test_connect_fail_then_success(mock_mt5):
    mock_mt5.initialize.side_effect = [False, True]
    mock_mt5.login.return_value = True
    config = DummyConfig()
    connector = MT5Connector(config, max_retries=2, retry_delay=0)
    assert connector.connect() is True

@patch("bot.connectors.mt5_connector.mt5")
def test_connect_fail_all(mock_mt5):
    mock_mt5.initialize.return_value = False
    config = DummyConfig()
    connector = MT5Connector(config, max_retries=2, retry_delay=0)
    with pytest.raises(MT5ConnectorError):
        connector.connect()

@patch("bot.connectors.mt5_connector.mt5")
def test_disconnect(mock_mt5):
    config = DummyConfig()
    connector = MT5Connector(config)
    connector.connected = True
    connector.disconnect()
    assert not connector.connected

@patch("bot.connectors.mt5_connector.mt5")
def test_is_connected_true(mock_mt5):
    config = DummyConfig()
    connector = MT5Connector(config)
    connector.connected = True
    mock_mt5.terminal_info.return_value = object()
    assert connector.is_connected() is True

@patch("bot.connectors.mt5_connector.mt5")
def test_is_connected_false(mock_mt5):
    config = DummyConfig()
    connector = MT5Connector(config)
    connector.connected = False
    mock_mt5.terminal_info.return_value = object()
    assert connector.is_connected() is False

@patch("bot.connectors.mt5_connector.mt5")
def test_ensure_connection_reconnects(mock_mt5):
    config = DummyConfig()
    connector = MT5Connector(config)
    connector.connected = False
    mock_mt5.terminal_info.return_value = None
    with patch.object(connector, "connect") as mock_connect:
        connector.ensure_connection()
        mock_connect.assert_called()

def test_connect_incomplete_credentials():
    class IncompleteConfig:
        def get(self, key, default=None):
            # Simule un login manquant
            return None if key == "mt5_login" else "ok"
    connector = MT5Connector(IncompleteConfig())
    with patch("bot.connectors.mt5_connector.mt5.initialize", return_value=True):
        with pytest.raises(MT5ConnectorError, match="incomplets"):
            connector.connect() 