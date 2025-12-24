import requests
import logging
from typing import List, Dict, Any, Optional

from bitcoin_scalper.core.data_requirements import DEFAULT_FETCH_LIMIT

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
                logger.warning(f"Erreur réseau (tentative {attempt}) vers {url}: {e}")
                # Fallback localhost si connexion échouée vers une IP distante
                if attempt == self.max_retries and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
                    logger.info(f"Tentative de fallback sur localhost:8000 suite aux échecs vers {self.base_url}")
                    try:
                        fallback_url = f"http://localhost:8000{endpoint}"
                        resp = self.session.request(method, fallback_url, timeout=self.timeout, **kwargs)
                        resp.raise_for_status()
                        logger.info("Connexion réussie via fallback localhost.")
                        # On met à jour la base_url pour les prochains appels
                        self.base_url = "http://localhost:8000"
                        return resp.json()
                    except Exception as fallback_e:
                        logger.error(f"Echec du fallback localhost: {fallback_e}")
                        raise MT5RestClientError(f"Erreur réseau persistante (y compris fallback localhost): {e}")

                if attempt == self.max_retries:
                    raise MT5RestClientError(f"Erreur réseau persistante: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue (tentative {attempt}): {e}")
                if attempt == self.max_retries:
                    raise MT5RestClientError(f"Erreur réseau persistante: {e}")

    def get_ticks(self, symbol: str, limit: int = DEFAULT_FETCH_LIMIT) -> List[Dict[str, Any]]:
        """
        Récupère les derniers ticks pour un symbole donné.
        
        Args:
            symbol: Symbol to fetch ticks for
            limit: Number of ticks (default: 1500 for proper feature engineering)
        """
        params = {"limit": limit}
        return self._request("GET", f"/ticks/{symbol}", params=params)

    def get_ohlcv(self, symbol: str, timeframe: str = "M1", limit: int = DEFAULT_FETCH_LIMIT) -> List[Dict[str, Any]]:
        """
        Récupère les dernières bougies OHLCV pour un symbole et timeframe donnés.
        
        Args:
            symbol: Symbol to fetch OHLCV for
            timeframe: Timeframe (default: M1)
            limit: Number of candles (default: 1500 for proper feature engineering)
        """
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
        logger.info(f"[MT5RestClient] Payload POST /order : {data}")
        return self._request("POST", "/order", json=data)

    def get_status(self) -> Dict[str, Any]:
        """Récupère le statut du serveur MT5 distant."""
        return self._request("GET", "/status")

    def get_positions(self) -> List[Dict[str, Any]]:
        """Récupère la liste des positions ouvertes sur le compte MT5."""
        return self._request("GET", "/positions")

"""
Exemple d'utilisation :
client = MT5RestClient("https://mt5-server.example.com/api", api_key="votre_cle_api")
ticks = client.get_ticks("BTCUSD", limit=200)
res = client.send_order("BTCUSD", action="buy", volume=0.01)
""" 