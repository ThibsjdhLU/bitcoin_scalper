import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("order_algos")

def execute_iceberg(total_qty: float, max_child: float, price: float, send_order_fn, **kwargs) -> List[Dict[str, Any]]:
    """
    Exécute un ordre Iceberg en fragmentant la quantité totale en ordres enfants de taille max_child.
    send_order_fn : fonction d'envoi d'ordre (signature : qty, price, **kwargs)
    Retourne la liste des résultats d'exécution.
    """
    results = []
    qty_left = total_qty
    while qty_left > 0:
        child_qty = min(max_child, qty_left)
        res = send_order_fn(qty=child_qty, price=price, **kwargs)
        results.append(res)
        qty_left -= child_qty
        logger.info(f"Iceberg: envoyé {child_qty} à {price}, reste {qty_left}")
    return results

def execute_twap(total_qty: float, n_slices: int, price: float, send_order_fn, interval_sec: float = 1.0, **kwargs) -> List[Dict[str, Any]]:
    """
    Exécute un TWAP (Time-Weighted Average Price) : envoie n_slices ordres de taille égale à intervalle régulier.
    (interval_sec ignoré ici, à gérer en live)
    """
    results = []
    slice_qty = total_qty / n_slices
    for i in range(n_slices):
        res = send_order_fn(qty=slice_qty, price=price, **kwargs)
        results.append(res)
        logger.info(f"TWAP: envoyé {slice_qty} à {price} (slice {i+1}/{n_slices})")
    return results

def execute_vwap(total_qty: float, price_series: List[float], send_order_fn, **kwargs) -> List[Dict[str, Any]]:
    """
    Exécute un VWAP (Volume-Weighted Average Price) : répartit la quantité selon le volume de chaque intervalle de prix.
    price_series : liste des prix/volumes historiques (ex: ticks ou OHLCV)
    """
    results = []
    n = len(price_series)
    weights = np.ones(n) / n
    slice_qtys = total_qty * weights
    for i, (qty, price) in enumerate(zip(slice_qtys, price_series)):
        res = send_order_fn(qty=qty, price=price, **kwargs)
        results.append(res)
        logger.info(f"VWAP: envoyé {qty} à {price} (slice {i+1}/{n})")
    return results 