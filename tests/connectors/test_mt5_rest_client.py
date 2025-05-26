import pytest
from unittest.mock import patch, MagicMock
from bot.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_get_ticks_success(mock_session):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"symbol": "BTCUSD", "bid": 1, "ask": 2}]
    mock_session.return_value.request.return_value = mock_resp
    client = MT5RestClient("http://localhost:8000", api_key="key")
    ticks = client.get_ticks("BTCUSD", limit=2)
    assert isinstance(ticks, list)
    assert ticks[0]["symbol"] == "BTCUSD"

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_get_ohlcv_success(mock_session):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"symbol": "BTCUSD", "open": 1, "close": 2}]
    mock_session.return_value.request.return_value = mock_resp
    client = MT5RestClient("http://localhost:8000")
    ohlcv = client.get_ohlcv("BTCUSD", timeframe="M1", limit=1)
    assert isinstance(ohlcv, list)
    assert ohlcv[0]["symbol"] == "BTCUSD"

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_send_order_success(mock_session):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"order_id": 123, "status": "filled"}
    mock_session.return_value.request.return_value = mock_resp
    client = MT5RestClient("http://localhost:8000")
    res = client.send_order("BTCUSD", action="buy", volume=0.01)
    assert res["status"] == "filled"

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_get_status_success(mock_session):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_session.return_value.request.return_value = mock_resp
    client = MT5RestClient("http://localhost:8000")
    status = client.get_status()
    assert status["status"] == "ok"

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_auth_error(mock_session):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_session.return_value.request.return_value = mock_resp
    client = MT5RestClient("http://localhost:8000", api_key="bad")
    with pytest.raises(MT5RestClientError, match="Authentification"):
        client.get_ticks("BTCUSD")

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_network_error_retry(mock_session):
    # Simule une erreur réseau puis un succès
    mock_session.return_value.request.side_effect = [Exception("fail"), MagicMock(status_code=200, json=lambda: [{"symbol": "BTCUSD"}])]
    client = MT5RestClient("http://localhost:8000", max_retries=2)
    res = client.get_ticks("BTCUSD")
    assert res[0]["symbol"] == "BTCUSD"

@patch("bot.connectors.mt5_rest_client.requests.Session")
def test_network_error_persistent(mock_session):
    mock_session.return_value.request.side_effect = Exception("fail")
    client = MT5RestClient("http://localhost:8000", max_retries=2)
    with pytest.raises(MT5RestClientError, match="réseau persistante"):
        client.get_ticks("BTCUSD") 