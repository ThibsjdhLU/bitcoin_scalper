import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("orderbook_monitor")

class OrderBookMonitor:
    """
    Surveillance de l'orderbook pour calcul d'imbalance, détection de pression et extraction des meilleurs prix.
    """
    def __init__(self, levels: int = 5):
        self.levels = levels

    def compute_imbalance(self, orderbook: Dict[str, Any]) -> float:
        """
        Calcule l'imbalance entre bids et asks sur les N premiers niveaux.
        """
        try:
            bids = orderbook.get('bids', [])[:self.levels]
            asks = orderbook.get('asks', [])[:self.levels]
            for b in bids:
                if not isinstance(b.get('volume', 0), (int, float)):
                    raise TypeError(f"Volume bid non numérique: {b.get('volume')}")
            for a in asks:
                if not isinstance(a.get('volume', 0), (int, float)):
                    raise TypeError(f"Volume ask non numérique: {a.get('volume')}")
            bid_vol = sum(float(b['volume']) for b in bids)
            ask_vol = sum(float(a['volume']) for a in asks)
            if bid_vol + ask_vol == 0:
                return 0.0
            return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        except Exception as e:
            logger.error(f"Erreur compute_imbalance: {e}")
            raise

    def detect_pressure(self, orderbook: Dict[str, Any], threshold: float = 0.1) -> int:
        """
        Détecte la pression d'achat/vente selon l'imbalance.
        Retourne 1 (achat), -1 (vente), 0 (neutre).
        """
        imb = self.compute_imbalance(orderbook)
        if imb > threshold:
            return 1
        elif imb < -threshold:
            return -1
        return 0

    def get_best_bid_ask(self, orderbook: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Retourne le meilleur bid et ask de l'orderbook.
        """
        try:
            best_bid = orderbook.get('bids', [{}])[0].get('price') if orderbook.get('bids') else None
            best_ask = orderbook.get('asks', [{}])[0].get('price') if orderbook.get('asks') else None
            return {'bid': best_bid, 'ask': best_ask}
        except Exception as e:
            logger.error(f"Erreur get_best_bid_ask: {e}")
            return {'bid': None, 'ask': None} 