import logging
from typing import Any, Dict, Optional, List, Union
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError
import time
import random
import os
import numpy as np
from logging.handlers import RotatingFileHandler
from bitcoin_scalper.core.adaptive_scheduler import AdaptiveStrategyScheduler

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

class OrderSimulator:
    """
    Simulateur d'ordres pour backtest ou mode démo. Gère latence, slippage, journalisation.
    """
    def __init__(self, slippage: float = 0.0, latency: float = 0.0):
        self.slippage = slippage
        self.latency = latency
        self.orders = []

    def send_order(self, symbol: str, action: str, volume: float, price: float, **kwargs) -> dict:
        """
        Simule l'envoi d'un ordre (buy/sell) avec slippage et latence.
        """
        time.sleep(self.latency)
        slip = self.slippage * price
        exec_price = price + slip if action == 'buy' else price - slip
        order = {
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'exec_price': exec_price,
            'timestamp': time.time(),
            'status': 'filled'
        }
        self.orders.append(order)
        logger.info(f"[OrderSimulator] {action.upper()} {volume} {symbol} @ {exec_price:.2f} (slip={slip:.2f})")
        return {'success': True, 'data': order}

def automate_trading(signal_series, price_series, simulator: OrderSimulator, symbol: str, volume: float):
    """
    Automatise la prise de décision : exécute un ordre à chaque changement de signal.
    :param signal_series: Série de signaux (-1, 0, 1)
    :param price_series: Série de prix
    :param simulator: Instance OrderSimulator
    :param symbol: Symbole à trader
    :param volume: Volume de l'ordre
    :return: Liste des ordres exécutés
    """
    last_signal = 0
    orders = []
    for t, sig in enumerate(signal_series):
        if sig != last_signal and sig != 0:
            action = 'buy' if sig == 1 else 'sell'
            price = price_series.iloc[t]
            res = simulator.send_order(symbol, action, volume, price)
            orders.append(res)
            last_signal = sig
    return orders

def test_order_simulator():
    """
    Test unitaire : vérifie la simulation d'un ordre avec slippage et latence.
    """
    sim = OrderSimulator(slippage=0.01, latency=0.01)
    res = sim.send_order('BTCUSD', 'buy', 0.1, 30000)
    assert res['success']
    assert abs(res['data']['exec_price'] - 30000.3) < 1e-3, f"Slippage incorrect : {res['data']['exec_price']}"
    print("Test OrderSimulator OK.")

def test_automate_trading():
    """
    Test unitaire : vérifie l'automatisation de la prise de décision sur une série de signaux.
    """
    import pandas as pd
    sim = OrderSimulator(slippage=0.0, latency=0.0)
    signals = pd.Series([0, 1, 1, -1, 0, 1])
    prices = pd.Series([100, 101, 102, 103, 104, 105])
    orders = automate_trading(signals, prices, sim, 'BTCUSD', 0.1)
    assert len(orders) == 3, f"Nombre d'ordres incorrect : {len(orders)}"
    print("Test automate_trading OK.")

def secure_audit_log(message: str, level: str = "INFO", audit_file: str = "audit.log"):
    """
    Journalise une action critique dans un fichier d'audit sécurisé avec rotation et contrôle d'accès.
    :param message: Message à journaliser
    :param level: Niveau de log (INFO, WARNING, ERROR)
    :param audit_file: Fichier d'audit
    """
    # Création du handler de rotation si besoin
    logger_audit = logging.getLogger("audit")
    if not logger_audit.hasHandlers():
        handler = RotatingFileHandler(audit_file, maxBytes=1000000, backupCount=5)
        handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger_audit.addHandler(handler)
        logger_audit.setLevel(logging.INFO)
        # Contrôle d'accès : permissions strictes
        try:
            os.chmod(audit_file, 0o600)
        except Exception:
            pass
    if level == "INFO":
        logger_audit.info(message)
    elif level == "WARNING":
        logger_audit.warning(message)
    elif level == "ERROR":
        logger_audit.error(message)
    else:
        logger_audit.info(message)

def test_secure_audit_log():
    """
    Test unitaire : vérifie la journalisation sécurisée dans le fichier d'audit.
    """
    import os
    audit_file = "test_audit.log"
    if os.path.exists(audit_file):
        os.remove(audit_file)
    secure_audit_log("Test audit info", level="INFO", audit_file=audit_file)
    secure_audit_log("Test audit warning", level="WARNING", audit_file=audit_file)
    secure_audit_log("Test audit error", level="ERROR", audit_file=audit_file)
    with open(audit_file, "r") as f:
        content = f.read()
        assert "Test audit info" in content
        assert "Test audit warning" in content
        assert "Test audit error" in content
    os.remove(audit_file)
    print("Test secure_audit_log OK.")

def execute_adaptive_trade(
    scheduler: AdaptiveStrategyScheduler,
    symbol: str,
    signal: int,
    proba: float,
    client: MT5RestClient,
    win_rate: float = None,
    reward_risk: float = None,
    pnl_history: list = None,
    stop_loss: float = None,
    probs: Optional[Union[np.ndarray, List[float]]] = None,
    **kwargs
) -> dict:
    """
    Exécute un trade adaptatif via le scheduler (filtrage, sizing dynamique, risk management).
    Args:
        scheduler (AdaptiveStrategyScheduler): Scheduler adaptatif
        symbol (str): Symbole à trader
        signal (int): Signal (-1, 0, 1)
        proba (float): Confiance du modèle
        client (MT5RestClient): Client REST
        win_rate (float): Taux de réussite estimé (pour Kelly)
        reward_risk (float): Ratio gain/risque (pour Kelly)
        pnl_history (list): Historique PnL (pour VaR)
        stop_loss (float): Distance stop loss (pour sizing)
        probs (list/array): Distribution complète des probabilités (pour Entropie)
    Returns:
        dict: Résultat de l'exécution (succès, raison, data)
    """
    decision = scheduler.schedule_trade(
        symbol=symbol,
        signal=signal,
        proba=proba,
        probs=probs,
        win_rate=win_rate,
        reward_risk=reward_risk,
        pnl_history=pnl_history,
        stop_loss=stop_loss
    )
    if decision is None:
        return {"success": False, "reason": "Aucun trade exécuté (filtre/scheduler)", "data": None}
    action = decision["action"]
    volume = decision["volume"]

    # On passe le stop_loss et take_profit potentiels s'ils sont dans kwargs
    # La logique Dynamic SL a peut-être déjà calculé les prix exacts
    # Mais ici on s'occupe de l'exécution
    res = send_order(symbol, volume, action, client, **kwargs)
    res["reason"] = decision["reason"]
    return res
