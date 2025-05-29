import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("mt5_rest_client")

class MT5RestClientError(Exception):
    """Exception personnalisée pour le client REST MT5."""
    pass

class MT5RestClient:
    """
    Client REST multiplateforme pour interagir avec un serveur MT5 distant (API REST).
    Permet de récupérer les ticks, OHLCV, exécuter des ordres, etc.
    """
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code == 401:
                    raise MT5RestClientError("Authentification échouée (clé API invalide)")
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(f"Erreur réseau (tentative {attempt}): {e}")
                if attempt == self.max_retries:
                    raise MT5RestClientError(f"Erreur réseau persistante: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue (tentative {attempt}): {e}")
                if attempt == self.max_retries:
                    raise MT5RestClientError(f"Erreur réseau persistante: {e}")

    def get_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les derniers ticks pour un symbole donné."""
        params = {"limit": limit}
        return self._request("GET", f"/ticks/{symbol}", params=params)

    def get_ohlcv(self, symbol: str, timeframe: str = "M1", limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les dernières bougies OHLCV pour un symbole et timeframe donnés."""
        params = {"timeframe": timeframe, "limit": limit}
        return self._request("GET", f"/ohlcv/{symbol}", params=params)

    def send_order(self, symbol: str, action: str, volume: float, price: Optional[float] = None, order_type: str = "market", **kwargs) -> Dict[str, Any]:
        """Envoie un ordre (buy/sell) au serveur MT5."""
        data = {
            "symbol": symbol,
            "action": action,
            "volume": volume,
            "order_type": order_type,
        }
        if price is not None:
            data["price"] = price
        data.update(kwargs)
        return self._request("POST", "/order", json=data)

    def get_status(self) -> Dict[str, Any]:
        """Récupère le statut du serveur MT5 distant."""
        return self._request("GET", "/status")

"""
Exemple d'utilisation :
client = MT5RestClient("https://mt5-server.example.com/api", api_key="votre_cle_api")
ticks = client.get_ticks("BTCUSD", limit=200)
res = client.send_order("BTCUSD", action="buy", volume=0.01)
""" 