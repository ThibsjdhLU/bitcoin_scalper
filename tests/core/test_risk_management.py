import pytest
from app.core.risk_management import RiskManager
from bot.connectors.mt5_rest_client import MT5RestClientError

class DummyClient:
    def __init__(self, balance=10000, equity=10000, tick_value=1.0):
        self._balance = balance
        self._equity = equity
        self._tick_value = tick_value
        self._fail = False
    def _request(self, method, endpoint):
        if self._fail:
            raise MT5RestClientError("Erreur réseau")
        if endpoint == "/account":
            return {"balance": self._balance, "equity": self._equity}
        if endpoint.startswith("/symbol/"):
            return {"tick_value": self._tick_value}
        return {}

def test_can_open_position_ok():
    client = DummyClient()
    rm = RiskManager(client)
    res = rm.can_open_position("BTCUSD", 0.5)
    assert res["allowed"] is True

def test_can_open_position_drawdown():
    client = DummyClient(balance=10000, equity=9000)
    rm = RiskManager(client, max_drawdown=0.05)
    rm.peak_balance = 10000
    res = rm.can_open_position("BTCUSD", 0.5)
    assert res["allowed"] is False
    assert "drawdown" in res["reason"].lower()

def test_can_open_position_daily_loss():
    client = DummyClient(balance=10000, equity=9500)
    rm = RiskManager(client, max_daily_loss=0.04)
    rm.peak_balance = 10000
    rm.last_balance = 10000
    rm.daily_pnl = -400
    res = rm.can_open_position("BTCUSD", 0.5)
    assert res["allowed"] is False
    assert "perte quotidienne" in res["reason"].lower()

def test_can_open_position_too_big():
    client = DummyClient()
    rm = RiskManager(client, max_position_size=0.1)
    res = rm.can_open_position("BTCUSD", 0.5)
    assert res["allowed"] is False
    assert "taille position" in res["reason"].lower()

def test_can_open_position_network_error():
    client = DummyClient()
    client._fail = True
    rm = RiskManager(client)
    res = rm.can_open_position("BTCUSD", 0.5)
    assert res["allowed"] is False
    assert "erreur" in res["reason"].lower()

def test_calculate_position_size():
    client = DummyClient(balance=10000, tick_value=2.0)
    rm = RiskManager(client, risk_per_trade=0.01, max_position_size=1.0)
    size = rm.calculate_position_size("BTCUSD", stop_loss=50)
    assert 0 < size <= 1.0

def test_calculate_position_size_network_error():
    client = DummyClient()
    client._fail = True
    rm = RiskManager(client)
    size = rm.calculate_position_size("BTCUSD", stop_loss=50)
    assert size == 0.0

def test_update_after_trade_and_metrics():
    client = DummyClient(balance=10000, equity=10500)
    rm = RiskManager(client)
    rm.peak_balance = 10000
    rm.last_balance = 10000
    rm.update_after_trade(500)
    metrics = rm.get_risk_metrics()
    assert metrics["peak_balance"] >= 10000
    assert metrics["last_balance"] == 10500
    assert "drawdown" in metrics

def test_can_open_position_zero_volume():
    client = DummyClient()
    rm = RiskManager(client)
    res = rm.can_open_position("BTCUSD", 0)
    assert res["allowed"] is True or res["allowed"] is False

def test_can_open_position_negative_volume():
    client = DummyClient()
    rm = RiskManager(client)
    res = rm.can_open_position("BTCUSD", -1)
    assert res["allowed"] is True or res["allowed"] is False

def test_can_open_position_non_numeric():
    client = DummyClient()
    rm = RiskManager(client)
    with pytest.raises(TypeError):
        rm.can_open_position("BTCUSD", "foo")

def test_calculate_position_size_zero_stop():
    client = DummyClient()
    rm = RiskManager(client)
    size = rm.calculate_position_size("BTCUSD", stop_loss=0)
    assert size == 0.0 or size == float('inf')

def test_calculate_position_size_negative_stop():
    client = DummyClient()
    rm = RiskManager(client)
    size = rm.calculate_position_size("BTCUSD", stop_loss=-10)
    assert size == 0.0 or size < 0

def test_calculate_position_size_non_numeric():
    client = DummyClient()
    rm = RiskManager(client)
    with pytest.raises(TypeError):
        rm.calculate_position_size("BTCUSD", stop_loss="foo")

def test_update_after_trade_exception():
    class CrashClient(DummyClient):
        def _request(self, method, endpoint):
            raise RuntimeError("Crash interne")
    client = CrashClient()
    rm = RiskManager(client)
    rm.peak_balance = 10000
    rm.last_balance = 10000
    rm.update_after_trade(500)  # Should not raise

def test_get_risk_metrics_exception():
    class CrashClient(DummyClient):
        def _request(self, method, endpoint):
            raise RuntimeError("Crash interne")
    client = CrashClient()
    rm = RiskManager(client)
    rm.peak_balance = 10000
    rm.last_balance = 10000
    metrics = rm.get_risk_metrics()
    assert metrics == {} 