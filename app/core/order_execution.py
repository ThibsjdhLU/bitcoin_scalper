"""
Module d'exécution d'ordres via MT5RestClient (REST, multiplateforme).
Compatible macOS, aucune dépendance native MetaTrader5 requise.
"""
from typing import Optional, Dict, Any
from bot.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError
import logging

logger = logging.getLogger(__name__)

def send_order(
    symbol: str,
    volume: float,
    order_type: str,
    price: Optional[float] = None,
    client: Optional[MT5RestClient] = None,
) -> Dict[str, Any]:
    """
    Envoie un ordre d'achat ou de vente via MT5RestClient.

    Args:
        symbol (str): Symbole (ex: 'BTCUSD').
        volume (float): Volume à trader.
        order_type (str): 'buy' ou 'sell'.
        price (float, optional): Prix limite (None pour market).
        client (MT5RestClient, optional): Instance du client REST. Si None, une exception est levée.

    Returns:
        dict: Réponse JSON du serveur MT5, ou dict d'erreur normalisé.

    Raises:
        ValueError: Si le client n'est pas fourni.
    """
    if client is None:
        raise ValueError("Un client MT5RestClient doit être fourni.")
    try:
        response = client.send_order(
            symbol=symbol,
            action=order_type,
            volume=volume,
            price=price,
            order_type="market" if price is None else "limit"
        )
        return {"success": True, "data": response}
    except MT5RestClientError as e:
        logger.error(f"Erreur MT5RestClient: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Erreur inattendue lors de l'envoi d'ordre")
        return {"success": False, "error": f"Erreur inattendue: {e}"} 