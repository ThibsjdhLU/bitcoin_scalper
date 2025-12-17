"""
Scheduler adaptatif pour exécution de stratégie ML avec gestion du risque et sizing dynamique.
Conforme PEP8, sécurité, docstring, et journalisation.
"""
from typing import Optional, Dict, Any, List, Union
import numpy as np
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.strategies_hybrid import KellySizer, VaRSizer
import logging

logger = logging.getLogger("AdaptiveStrategyScheduler")

class AdaptiveStrategyScheduler:
    """
    Orchestration adaptative :
    - Filtrage du signal (zone d'incertitude + Entropie de Shannon)
    - Sizing dynamique (Risk %, Kelly, VaR, confiance ML)
    - Validation risk manager (drawdown, perte max, etc)
    """
    def __init__(
        self,
        risk_manager: RiskManager,
        filter: Optional[TradeDecisionFilter] = None,
        sizing_method: str = "risk_adjusted", # Default changed to leverage dynamic SL
        min_size: float = 0.01,
        max_size: float = 1.0
    ):
        self.risk_manager = risk_manager
        self.filter = filter or TradeDecisionFilter()
        self.sizing_method = sizing_method
        self.min_size = min_size
        self.max_size = max_size
        self.kelly = KellySizer()
        self.var = VaRSizer()

    def schedule_trade(
        self,
        symbol: str,
        signal: int,
        proba: float,
        probs: Optional[Union[np.ndarray, List[float]]] = None,
        win_rate: Optional[float] = None,
        reward_risk: Optional[float] = None,
        pnl_history: Optional[list] = None,
        stop_loss: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Décide d'exécuter un trade ou non, et calcule le sizing optimal.
        Args:
            symbol (str): Symbole à trader
            signal (int): Signal (-1, 0, 1)
            proba (float): Confiance du modèle (classe dominante)
            probs (list/array): Distribution complète des probabilités (pour Entropie)
            win_rate (float): Taux de réussite estimé (pour Kelly)
            reward_risk (float): Ratio gain/risque (pour Kelly)
            pnl_history (list): Historique PnL (pour VaR)
            stop_loss (float): Distance stop loss (pour sizing basé sur le risque)
        Returns:
            dict ou None : {'action': 'buy'/'sell', 'volume': float, 'reason': str}
        """
        # 1. Pas de signal = pas de trade
        if signal == 0:
            logger.info("Aucun signal, aucun trade exécuté.")
            return None

        # 2. Filtrage adaptatif (Incertitude + Entropie)
        # On passe 'probs' au filtre pour activer le check d'entropie
        accepted, reason = self.filter.filter(proba, probs)
        if not accepted:
            logger.info(f"Trade refusé par filtre : {reason}")
            return None

        # 3. Sizing dynamique
        size = self.min_size

        if self.sizing_method == "risk_adjusted" and stop_loss is not None:
            # Sizing basé sur le risque par trade (ex: 1% du capital) et la distance du SL
            # C'est la méthode "Pro" recommandée avec des stops dynamiques
            calc_size = self.risk_manager.calculate_position_size(symbol, stop_loss)
            size = calc_size
        elif self.sizing_method == "kelly" and win_rate is not None and reward_risk is not None:
            size = self.kelly.size(win_rate, reward_risk)
        elif self.sizing_method == "var" and pnl_history is not None:
            size = self.var.size(pnl_history)
        elif self.sizing_method == "confidence":
            # Sizing proportionnel à la confiance
            size = max(self.min_size, min(self.max_size, proba))

        # Bornage du volume
        size = max(self.min_size, min(self.max_size, size))

        # 4. Validation risk manager (Limites globales)
        risk_check = self.risk_manager.can_open_position(symbol, size)
        if not risk_check["allowed"]:
            logger.info(f"Trade bloqué par risk manager : {risk_check['reason']}")
            return None

        action = "buy" if signal == 1 else "sell"
        logger.info(f"Trade accepté : {action} {size} {symbol} (raison : {reason})")
        return {"action": action, "volume": size, "reason": reason}
