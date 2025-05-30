import logging
from typing import Any, Dict
from bot.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError

logger = logging.getLogger("order_execution")

def send_order(symbol: str, volume: float, action: str, client: MT5RestClient = None, **kwargs) -> Dict[str, Any]:
    """
    Envoie un ordre au broker via MT5RestClient avec gestion d'erreur sécurisée.
    Args:
        symbol (str): Symbole à trader.
        volume (float): Volume de l'ordre.
        action (str): 'buy' ou 'sell'.
        client (MT5RestClient): Client REST pour l'envoi d'ordre.
    Returns:
        dict: Résultat de l'envoi (succès, données, erreur)
    """
    if client is None:
        raise ValueError("Un client MT5RestClient doit être fourni pour l'envoi d'ordre.")
    try:
        res = client.send_order(symbol, volume=volume, action=action, **kwargs)
        logger.info(f"[order_execution] Réponse brute API /order : {res}")
        return {"success": True, "data": res}
    except MT5RestClientError as e:
        logger.error(f"Erreur MT5RestClient lors de l'envoi d'ordre: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Erreur inattendue lors de l'envoi d'ordre")
        return {"success": False, "error": f"Erreur inattendue: {e}"} 