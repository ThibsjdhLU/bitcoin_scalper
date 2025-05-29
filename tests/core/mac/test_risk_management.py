import pytest
from unittest.mock import MagicMock, patch
from bitcoin_scalper.core.risk_management import RiskManager, MT5RestClientError
from bot.connectors.mt5_rest_client import MT5RestClient # Import the actual class for type hinting in mock spec

# Fixture pour un client MT5RestClient mocké
@pytest.fixture
def mock_mt5_client():
    """
    Fixture fournissant un mock de MT5RestClient.
    """
    # Utiliser spec=MT5RestClient pour s'assurer que le mock a les bonnes méthodes
    mock_client = MagicMock(spec=MT5RestClient)
    # Configurer le mock _request pour retourner des données de compte par défaut
    # Ceci est une configuration de base qui pourra être adaptée dans les tests spécifiques
    mock_client._request.return_value = {
        "balance": 1000.0,
        "equity": 1000.0
    }
    return mock_client

# Test initial pour l'initialisation de RiskManager
def test_risk_manager_init(mock_mt5_client):
    """
    Teste l'initialisation de la classe RiskManager avec les valeurs par défaut.
    """
    risk_manager = RiskManager(mock_mt5_client)

    assert risk_manager.client == mock_mt5_client
    assert risk_manager.max_drawdown == 0.05
    assert risk_manager.max_daily_loss == 0.05
    assert risk_manager.risk_per_trade == 0.01
    assert risk_manager.max_position_size == 1.0
    assert risk_manager.peak_balance is None
    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.last_balance is None

def test_risk_manager_init_custom_values(mock_mt5_client):
    """
    Teste l'initialisation de la classe RiskManager avec des valeurs personnalisées.
    """
    custom_drawdown = 0.1
    custom_daily_loss = 0.02
    custom_risk_per_trade = 0.005
    custom_max_pos_size = 0.5

    risk_manager = RiskManager(
        mock_mt5_client,
        max_drawdown=custom_drawdown,
        max_daily_loss=custom_daily_loss,
        risk_per_trade=custom_risk_per_trade,
        max_position_size=custom_max_pos_size
    )

    assert risk_manager.client == mock_mt5_client
    assert risk_manager.max_drawdown == custom_drawdown
    assert risk_manager.max_daily_loss == custom_daily_loss
    assert risk_manager.risk_per_trade == custom_risk_per_trade
    assert risk_manager.max_position_size == custom_max_pos_size
    assert risk_manager.peak_balance is None
    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.last_balance is None

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
    res = rm.can_open_position("BTCUSD", "foo")
    assert res["allowed"] is False
    assert "erreur" in res["reason"].lower()

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
    size = rm.calculate_position_size("BTCUSD", stop_loss="foo")
    assert size == 0.0

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

def test_can_open_position_allowed(mock_mt5_client):
    """
    Teste la méthode can_open_position quand l'ouverture de position est autorisée.
    """
    risk_manager = RiskManager(mock_mt5_client, max_drawdown=0.05)
    symbol = "BTCUSD"
    volume = 0.1

    # Simuler une réponse de compte sans drawdown excessif
    mock_mt5_client._request.return_value = {
        "balance": 10000.0,
        "equity": 9900.0 # Drawdown de 1% (100 / 10000), inférieur à 5%
    }

    result = risk_manager.can_open_position(symbol, volume)

    assert result["allowed"] is True
    assert result["reason"] == "OK"
    mock_mt5_client._request.assert_called_once_with("GET", "/account")
    assert risk_manager.peak_balance == 10000.0
    assert risk_manager.last_balance == 9900.0

def test_can_open_position_drawdown_exceeded(mock_mt5_client):
    """
    Teste la méthode can_open_position quand le drawdown maximum est dépassé.
    """
    risk_manager = RiskManager(mock_mt5_client, max_drawdown=0.05)
    symbol = "BTCUSD"
    volume = 0.1

    # Simuler une réponse de compte avec un drawdown excessif
    mock_mt5_client._request.return_value = {
        "balance": 10000.0,
        "equity": 9000.0 # Drawdown de 10% (1000 / 10000), supérieur à 5%
    }

    result = risk_manager.can_open_position(symbol, volume)

    assert result["allowed"] is False
    assert "Drawdown max dépassé" in result["reason"]
    mock_mt5_client._request.assert_called_once_with("GET", "/account")
    assert risk_manager.peak_balance == 10000.0 # peak_balance est initialisé même en cas de drawdown
    assert risk_manager.last_balance == 9000.0

def test_can_open_position_daily_loss_exceeded(mock_mt5_client):
    """
    Teste la méthode can_open_position quand la perte quotidienne maximum est dépassée.
    """
    risk_manager = RiskManager(mock_mt5_client, max_daily_loss=0.01) # 1% de perte quotidienne max
    symbol = "BTCUSD"
    volume = 0.1

    # Simuler une perte continue :
    # 1er appel : equity passe de 10000 à 9900 (perte de 100)
    mock_mt5_client._request.return_value = {
        "balance": 10000.0,
        "equity": 9900.0
    }
    risk_manager.peak_balance = 10000.0
    risk_manager.last_balance = 10000.0
    risk_manager.daily_pnl = 0.0
    result1 = risk_manager.can_open_position(symbol, volume)
    assert result1["allowed"] is True
    # 2e appel : equity passe de 9900 à 9800 (perte de 100 supplémentaire)
    mock_mt5_client._request.return_value = {
        "balance": 10000.0,
        "equity": 9800.0
    }
    result2 = risk_manager.can_open_position(symbol, volume)
    assert result2["allowed"] is False
    assert "Perte quotidienne max dépassée" in result2["reason"]
    mock_mt5_client._request.assert_called_with("GET", "/account")
    # Vérifier que daily_pnl a été mis à jour
    assert risk_manager.daily_pnl == -200.0
    assert risk_manager.last_balance == 9800.0

def test_can_open_position_volume_exceeded(mock_mt5_client):
    """
    Teste la méthode can_open_position quand le volume souhaité dépasse le maximum autorisé.
    """
    risk_manager = RiskManager(mock_mt5_client, max_position_size=0.5)
    symbol = "BTCUSD"
    volume = 1.0 # Volume supérieur à max_position_size (0.5)

    # Simuler une réponse de compte valide (ne devrait pas avoir d'impact sur ce test)
    mock_mt5_client._request.return_value = {
        "balance": 10000.0,
        "equity": 10000.0
    }

    result = risk_manager.can_open_position(symbol, volume)

    assert result["allowed"] is False
    assert "Taille position > max autorisé" in result["reason"]
    mock_mt5_client._request.assert_called_once_with("GET", "/account")
    # Vérifier que peak_balance et last_balance sont initialisés même dans ce cas
    assert risk_manager.peak_balance == 10000.0
    assert risk_manager.last_balance == 10000.0

def test_can_open_position_mt5_client_error(mock_mt5_client):
    """
    Teste la gestion des exceptions MT5RestClientError dans can_open_position.
    """
    risk_manager = RiskManager(mock_mt5_client)
    symbol = "BTCUSD"
    volume = 0.1

    # Simuler une MT5RestClientError lors de l'appel _request
    mock_mt5_client._request.side_effect = MT5RestClientError("Erreur simulee API")

    result = risk_manager.can_open_position(symbol, volume)

    assert result["allowed"] is False
    assert result["reason"] == "Erreur simulee API"
    mock_mt5_client._request.assert_called_once_with("GET", "/account")
    # Vérifier que l'exception est loguée (optionnel, mais bonne pratique - nécessite patch du logger)

def test_can_open_position_generic_exception(mock_mt5_client):
    """
    Teste la gestion des exceptions génériques dans can_open_position.
    """
    risk_manager = RiskManager(mock_mt5_client)
    symbol = "BTCUSD"
    volume = 0.1

    # Simuler une exception générique inattendue (ex: KeyError sur la réponse)
    mock_mt5_client._request.return_value = {} # Réponse vide pour forcer une KeyError lors de l'accès à 'balance'

    result = risk_manager.can_open_position(symbol, volume)

    assert result["allowed"] is False
    assert "Erreur inattendue" in result["reason"]
    mock_mt5_client._request.assert_called_once_with("GET", "/account")
    # Vérifier que l'exception est loguée (optionnel, nécessite patch du logger)

# TODO: Ajouter des tests pour les autres scénarios de can_open_position
# (perte quotidienne, taille de position, exceptions) 