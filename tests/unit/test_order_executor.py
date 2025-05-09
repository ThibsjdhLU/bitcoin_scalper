"""
Tests unitaires pour le module OrderExecutor.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.bitcoin_scalper.core.mt5_connector import MT5Connector
from src.bitcoin_scalper.core.order_executor import OrderExecutor, OrderSide, OrderType


@pytest.fixture
def mock_mt5():
    """Mock de MetaTrader5 pour les tests."""
    with patch("core.order_executor.mt5") as mock:
        # Constantes MT5
        mock.TRADE_ACTION_DEAL = 1
        mock.TRADE_ACTION_PENDING = 5
        mock.TRADE_ACTION_MODIFY = 6
        mock.TRADE_ACTION_REMOVE = 8
        mock.ORDER_TYPE_BUY = 0
        mock.ORDER_TYPE_SELL = 1
        mock.ORDER_TYPE_BUY_LIMIT = 2
        mock.ORDER_TYPE_SELL_LIMIT = 3
        mock.ORDER_TYPE_BUY_STOP = 4
        mock.ORDER_TYPE_SELL_STOP = 5
        mock.ORDER_TIME_GTC = 0
        mock.ORDER_FILLING_IOC = 1
        mock.TRADE_RETCODE_DONE = 10009

        # Mock des résultats d'ordre
        result = MagicMock()
        result.retcode = mock.TRADE_RETCODE_DONE
        result.order = 12345
        result.comment = "Test order"
        mock.order_send.return_value = result

        # Mock des informations d'ordre
        order = MagicMock()
        order.symbol = "BTCUSD"
        order.volume_initial = 0.1
        order.type = 0
        order.position_id = 12345
        order.price_open = 50000.0
        order.sl = 49000.0
        order.tp = 51000.0
        mock.order_get.return_value = order

        yield mock


@pytest.fixture
def mock_connector():
    """Mock du connecteur MT5."""
    connector = MagicMock(spec=MT5Connector)
    connector.connected = True
    connector.get_symbol_info.return_value = {
        "name": "BTCUSD",
        "volume_min": 0.01,
        "volume_max": 1.0,
        "digits": 2,
        "bid": 49999.0,
        "ask": 50001.0,
        "spread": 2,
    }

    # Mock du résultat de place_order
    order_result = MagicMock()
    order_result.ticket = 12345
    order_result.volume = 0.1
    order_result.price = 50000.0
    order_result.sl = 49000.0
    order_result.tp = 51000.0

    connector.place_order.return_value = {
        "order": order_result,
        "request": {
            "action": 1,
            "symbol": "BTCUSD",
            "volume": 0.1,
            "type": 0,
            "price": 50000.0,
        },
    }

    return connector


@pytest.fixture
def order_executor(mock_connector, tmp_path):
    """Instance d'OrderExecutor pour les tests."""
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

    return OrderExecutor(mock_connector, str(config_path))


def test_init(order_executor, mock_connector):
    """Teste l'initialisation de l'exécuteur d'ordres."""
    assert order_executor.connector == mock_connector
    assert hasattr(order_executor, "config")


def test_validate_order_params(order_executor):
    """Teste la validation des paramètres d'ordre."""
    # Test avec des paramètres valides
    assert order_executor._validate_order_params(
        symbol="BTCUSD", volume=0.1, order_type=OrderType.MARKET_BUY, side=OrderSide.BUY
    )

    # Test avec un volume invalide
    assert not order_executor._validate_order_params(
        symbol="BTCUSD",
        volume=2.0,  # > volume_max
        order_type=OrderType.MARKET_BUY,
        side=OrderSide.BUY,
    )

    # Test avec un symbole invalide
    order_executor.connector.get_symbol_info.return_value = None
    assert not order_executor._validate_order_params(
        symbol="INVALID",
        volume=0.1,
        order_type=OrderType.MARKET_BUY,
        side=OrderSide.BUY,
    )


def test_execute_market_order(order_executor, mock_mt5):
    """Teste l'exécution d'un ordre au marché."""
    success, order_id = order_executor.execute_market_order(
        symbol="BTCUSD", volume=0.1, side=OrderSide.BUY, sl=49000.0, tp=51000.0
    )

    assert success
    assert order_id == 12345
    mock_mt5.order_send.assert_called_once()


def test_execute_limit_order(order_executor, mock_mt5):
    """Teste l'exécution d'un ordre limite."""
    success, order_id = order_executor.execute_limit_order(
        symbol="BTCUSD",
        volume=0.1,
        side=OrderSide.BUY,
        price=49000.0,
        sl=48000.0,
        tp=50000.0,
    )

    assert success
    assert order_id == 12345
    mock_mt5.order_send.assert_called_once()


def test_execute_stop_order(order_executor, mock_mt5):
    """Teste l'exécution d'un ordre stop."""
    success, order_id = order_executor.execute_stop_order(
        symbol="BTCUSD",
        volume=0.1,
        side=OrderSide.BUY,
        price=51000.0,
        sl=50000.0,
        tp=52000.0,
    )

    assert success
    assert order_id == 12345
    mock_mt5.order_send.assert_called_once()


def test_place_order(order_executor, mock_mt5):
    """Teste le placement d'un ordre avec la nouvelle méthode."""
    # Test d'un ordre d'achat au marché
    status = order_executor.place_order(
        symbol="BTCUSD",
        order_type="MARKET",
        volume=0.1,
        price=50002.0,  # > ask -> BUY
        stop_loss=49000.0,
        take_profit=51000.0,
    )

    assert status is not None
    assert status.type == "MARKET_BUY"
    assert status.order_id == 12345

    # Test d'un ordre de vente au marché
    status = order_executor.place_order(
        symbol="BTCUSD",
        order_type="MARKET",
        volume=0.1,
        price=49998.0,  # < bid -> SELL
        stop_loss=48000.0,
        take_profit=49500.0,
    )

    assert status is not None
    assert status.type == "MARKET_SELL"
    assert status.order_id == 12345


def test_order_type_conversion():
    """Teste la conversion des types d'ordres."""
    # Test des conversions valides
    assert OrderType.from_string("MARKET", "BUY") == OrderType.MARKET_BUY
    assert OrderType.from_string("MARKET", "SELL") == OrderType.MARKET_SELL
    assert OrderType.from_string("LIMIT", "BUY") == OrderType.LIMIT_BUY
    assert OrderType.from_string("LIMIT", "SELL") == OrderType.LIMIT_SELL
    assert OrderType.from_string("STOP", "BUY") == OrderType.STOP_BUY
    assert OrderType.from_string("STOP", "SELL") == OrderType.STOP_SELL

    # Test des conversions invalides
    with pytest.raises(ValueError):
        OrderType.from_string("INVALID", "BUY")
    with pytest.raises(ValueError):
        OrderType.from_string("MARKET", "INVALID")


def test_modify_order(order_executor, mock_mt5):
    """Teste la modification d'un ordre."""
    success = order_executor.modify_order(order_id=12345, sl=48000.0, tp=52000.0)

    assert success
    mock_mt5.order_get.assert_called_once_with(12345)
    mock_mt5.order_send.assert_called_once()


def test_cancel_order(order_executor, mock_mt5):
    """Teste l'annulation d'un ordre."""
    success = order_executor.cancel_order(order_id=12345)

    assert success
    mock_mt5.order_send.assert_called_once()


def test_order_failure(order_executor, mock_mt5):
    """Teste la gestion des échecs d'ordre."""
    # Simuler un échec d'ordre
    mock_mt5.order_send.return_value.retcode = 10018  # TRADE_RETCODE_INVALID_PRICE

    success, order_id = order_executor.execute_market_order(
        symbol="BTCUSD", volume=0.1, side=OrderSide.BUY
    )

    assert not success
    assert order_id is None
