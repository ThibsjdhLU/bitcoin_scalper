import logging
from typing import Optional
import threading

logger = logging.getLogger("bitcoin_scalper.monitoring")
logger.setLevel(logging.INFO)

def start_prometheus_server(port: int = 8000):
    """
    Démarre un serveur Prometheus pour exposer les métriques du bot.
    :param port: port d'écoute
    """
    try:
        from prometheus_client import start_http_server, Gauge
    except ImportError:
        logger.error("prometheus_client n'est pas installé")
        return None
    # Exemple de métrique : capital courant
    capital_gauge = Gauge('bot_capital', 'Capital courant du bot')
    def update_metrics():
        import time
        while True:
            # TODO : remplacer par la vraie valeur du capital
            capital_gauge.set(10000)
            time.sleep(10)
    threading.Thread(target=update_metrics, daemon=True).start()
    start_http_server(port)
    logger.info(f"Serveur Prometheus démarré sur le port {port}")
    return capital_gauge

def send_alert(message: str, channel: str = "console", webhook_url: Optional[str] = None):
    """
    Envoie une alerte (console, email, Slack, etc.).
    :param message: texte de l'alerte
    :param channel: "console", "slack", "email"
    :param webhook_url: URL du webhook Slack (si channel=slack)
    """
    if channel == "console":
        logger.warning(f"ALERTE : {message}")
    elif channel == "slack" and webhook_url:
        import requests
        resp = requests.post(webhook_url, json={"text": message})
        if resp.status_code != 200:
            logger.error(f"Erreur Slack : {resp.text}")
    elif channel == "email":
        # TODO : implémenter l'envoi d'email sécurisé
        logger.warning(f"[EMAIL] {message}")
    else:
        logger.error(f"Canal d'alerte inconnu : {channel}")

def healthcheck() -> bool:
    """
    Vérifie la santé du bot (ex : latence, capital, erreurs critiques).
    :return: True si OK, False sinon
    """
    # TODO : ajouter des checks réels (latence, capital, erreurs, etc.)
    logger.info("Healthcheck : OK")
    return True 