import logging
from typing import Optional, List, Dict, Any
import threading
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger("bitcoin_scalper.monitoring")
logger.setLevel(logging.INFO)

class DriftMonitor:
    """
    Moniteur de dérive de données (Data Drift) utilisant le test de Kolmogorov-Smirnov (KS-Test).
    Compare la distribution des features en production avec celle de l'entraînement.
    """
    def __init__(self, reference_data: pd.DataFrame, key_features: Optional[List[str]] = None, p_value_threshold: float = 0.05):
        """
        :param reference_data: DataFrame d'entraînement (référence).
        :param key_features: Liste des features les plus importantes à surveiller (ex: top 3).
        :param p_value_threshold: Seuil p-value pour déclencher une alerte (défaut 0.05).
        """
        self.reference_data = reference_data
        # Si key_features non fourni, on prend tout (attention performance) ou top N si possible
        self.key_features = key_features or list(reference_data.select_dtypes(include=[np.number]).columns)
        self.p_value_threshold = p_value_threshold
        self.drift_status = {feat: False for feat in self.key_features}

    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Effectue le test KS sur les nouvelles données.
        :param new_data: DataFrame récent (ex: 4h de données).
        :return: Rapport de drift.
        """
        report = {}
        drift_detected = False

        for feature in self.key_features:
            if feature not in new_data.columns:
                logger.warning(f"Feature {feature} absente des nouvelles données.")
                continue

            # KS Test
            # Null hypothesis: distributions are the same.
            # If p_value < threshold, we reject null hypothesis -> Drift Detected.
            stat, p_value = ks_2samp(self.reference_data[feature].dropna(), new_data[feature].dropna())

            is_drifting = p_value < self.p_value_threshold
            self.drift_status[feature] = is_drifting

            report[feature] = {
                "ks_stat": stat,
                "p_value": p_value,
                "drift": is_drifting
            }

            if is_drifting:
                drift_detected = True
                logger.warning(f"🚨 DRIFT DETECTED on {feature} (p={p_value:.4f} < {self.p_value_threshold})")

        return {"drift_detected": drift_detected, "details": report}

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