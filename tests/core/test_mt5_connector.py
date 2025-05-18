import pytest
from unittest.mock import patch, MagicMock
from app.core.mt5_connector import MT5Connector, MT5ConnectionError
from app.core.config import SecureConfig

@pytest.fixture
def config_mock():
    class DummyConfig:
        def get_encrypted(self, key):
            if key == 'mt5_login':
                return '123456'
            if key == 'mt5_password':
                return 'password'
            raise KeyError(key)
        def get(self, key, default=None):
            if key == 'mt5_server':
                return 'AvaTrade-Demo'
            if key == 'mt5_path':
                return '/fake/path'
            return default
    return DummyConfig()

@patch('app.core.mt5_connector.mt5')
def test_connect_success(mt5_mock, config_mock):
    mt5_mock.initialize.return_value = True
    mt5_mock.login.return_value = True
    mt5_mock.last_error.return_value = None
    connector = MT5Connector(config_mock)
    assert connector.connect() is True
    assert connector.is_connected()

@patch('app.core.mt5_connector.mt5')
def test_connect_failure(mt5_mock, config_mock):
    mt5_mock.initialize.return_value = False
    mt5_mock.last_error.return_value = 'init error'
    connector = MT5Connector(config_mock, max_retries=2, retry_delay=0)
    assert connector.connect() is False
    assert not connector.is_connected()

@patch('app.core.mt5_connector.mt5')
def test_disconnect(mt5_mock, config_mock):
    mt5_mock.initialize.return_value = True
    mt5_mock.login.return_value = True
    mt5_mock.last_error.return_value = None
    mt5_mock.terminal_info.return_value = True
    connector = MT5Connector(config_mock)
    connector.connect()
    connector.disconnect()
    assert not connector.is_connected() 