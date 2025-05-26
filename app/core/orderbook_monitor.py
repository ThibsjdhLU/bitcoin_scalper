import numpy as np
import pandas as pd
from typing import List, Dict, Any

class OrderBookMonitor:
    """
    Surveillance du carnet d'ordres (market depth), détection de déséquilibres, signaux de pression.
    """
    def __init__(self, levels: int = 5):
        self.levels = levels

    def compute_imbalance(self, orderbook: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calcule le déséquilibre du carnet (bid vs ask) sur les N premiers niveaux.
        orderbook : {'bids': [{'price':..., 'volume':...}, ...], 'asks': [...]}
        Retourne un score [-1, 1] (pression vendeuse à acheteuse).
        """
        bids = orderbook.get('bids', [])[:self.levels]
        asks = orderbook.get('asks', [])[:self.levels]
        bid_vol = sum([b['volume'] for b in bids])
        ask_vol = sum([a['volume'] for a in asks])
        total = bid_vol + ask_vol + 1e-9
        imbalance = (bid_vol - ask_vol) / total
        return imbalance

    def detect_pressure(self, orderbook: Dict[str, List[Dict[str, Any]]], threshold: float = 0.2) -> int:
        """
        Retourne 1 si pression acheteuse, -1 si vendeuse, 0 sinon.
        """
        imbalance = self.compute_imbalance(orderbook)
        if imbalance > threshold:
            return 1
        elif imbalance < -threshold:
            return -1
        else:
            return 0

    def get_best_bid_ask(self, orderbook: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Retourne le meilleur bid/ask du carnet.
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        best_bid = bids[0]['price'] if bids else None
        best_ask = asks[0]['price'] if asks else None
        return {'bid': best_bid, 'ask': best_ask} 