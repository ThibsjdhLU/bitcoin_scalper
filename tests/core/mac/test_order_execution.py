import pytest
from unittest.mock import MagicMock, patch
from bitcoin_scalper.core.order_execution import send_order
from bot.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError

class DummyClient:
    def send_order(self, *args, **kwargs):
        return {"order_id": 123, "status": "filled"}

def test_send_order_success():
    client = DummyClient()
    result = send_order("BTCUSD", 0.01, "buy", client=client)
    assert result["success"] is True
    assert "data" in result
    assert result["data"]["order_id"] == 123

def test_send_order_mt5_error():
    class FailingClient:
        def send_order(self, *args, **kwargs):
            raise MT5RestClientError("Erreur d'authentification")
    client = FailingClient()
    result = send_order("BTCUSD", 0.01, "buy", client=client)
    assert result["success"] is False
    assert "authentification" in result["error"].lower()

def test_send_order_network_error():
    class NetworkFailClient:
        def send_order(self, *args, **kwargs):
            raise MT5RestClientError("Erreur réseau persistante: timeout")
    client = NetworkFailClient()
    result = send_order("BTCUSD", 0.01, "buy", client=client)
    assert result["success"] is False
    assert "réseau" in result["error"].lower() or "timeout" in result["error"].lower()

def test_send_order_unexpected_exception():
    class CrashClient:
        def send_order(self, *args, **kwargs):
            raise RuntimeError("Crash interne")
    client = CrashClient()
    result = send_order("BTCUSD", 0.01, "buy", client=client)
    assert result["success"] is False
    assert "inattendue" in result["error"].lower()
    assert "crash interne" in result["error"].lower()

def test_send_order_no_client():
    with pytest.raises(ValueError):
        send_order("BTCUSD", 0.01, "buy", client=None)

def test_send_order_payload_format():
    captured = {}
    class DummyClient:
        def send_order(self, symbol, volume, action, **kwargs):
            captured['symbol'] = symbol
            captured['volume'] = volume
            captured['action'] = action
            captured.update(kwargs)
            return {'order_id': 123, 'status': 'OK'}
    client = DummyClient()
    from bitcoin_scalper.core.order_execution import send_order
    result = send_order('BTCUSD', 0.01, 'buy', client=client, price=12345.6, order_type='market', sl=100, tp=200)
    assert result['success'] is True
    assert captured['symbol'] == 'BTCUSD'
    assert captured['action'] == 'buy'
    assert captured['volume'] == 0.01
    assert captured['price'] == 12345.6
    assert captured['order_type'] == 'market'
    assert captured['sl'] == 100
    assert captured['tp'] == 200 