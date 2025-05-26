"""
Gestionnaire de risques REST multiplateforme pour trading BTC/USD.
Utilise exclusivement MT5RestClient (aucune dépendance native MT5, compatible macOS).
"""
from typing import Optional, Dict, Any
from bot.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Gestionnaire de risques : drawdown, VaR, stops dynamiques, limites de perte, taille de position.
    Toutes les interactions broker passent par MT5RestClient (REST).
    """
    def __init__(
        self,
        client: MT5RestClient,
        max_drawdown: float = 0.05,
        max_daily_loss: float = 0.05,
        risk_per_trade: float = 0.01,
        max_position_size: float = 1.0,
    ):
        """
        Args:
            client (MT5RestClient): Client REST pour requêtes broker.
            max_drawdown (float): Drawdown max autorisé (ex: 0.05 pour 5%).
            max_daily_loss (float): Perte quotidienne max autorisée (ratio capital).
            risk_per_trade (float): Risque max par trade (ratio capital).
            max_position_size (float): Taille max d'une position (en lots).
        """
        self.client = client
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.peak_balance = None
        self.daily_pnl = 0.0
        self.last_balance = None

    def can_open_position(self, symbol: str, volume: float) -> Dict[str, Any]:
        """
        Vérifie si une nouvelle position peut être ouverte selon les règles de risque.
        Args:
            symbol (str): Symbole à trader.
            volume (float): Volume souhaité.
        Returns:
            dict: {'allowed': bool, 'reason': str}
        """
        try:
            account = self.client._request("GET", "/account")
            balance = account.get("balance", 0.0)
            equity = account.get("equity", balance)
            if self.peak_balance is None:
                self.peak_balance = balance
            if self.last_balance is None:
                self.last_balance = balance
            # Drawdown
            drawdown = (self.peak_balance - equity) / self.peak_balance if self.peak_balance else 0.0
            if drawdown > self.max_drawdown:
                return {"allowed": False, "reason": f"Drawdown max dépassé ({drawdown:.2%})"}
            # Perte quotidienne
            pnl = equity - self.last_balance
            self.daily_pnl += pnl
            if abs(self.daily_pnl) > self.max_daily_loss * self.peak_balance:
                return {"allowed": False, "reason": f"Perte quotidienne max dépassée ({self.daily_pnl:.2f})"}
            # Taille position
            if volume > self.max_position_size:
                return {"allowed": False, "reason": f"Taille position > max autorisé ({self.max_position_size})"}
            return {"allowed": True, "reason": "OK"}
        except MT5RestClientError as e:
            logger.error(f"Erreur MT5RestClient: {e}")
            return {"allowed": False, "reason": str(e)}
        except Exception as e:
            logger.exception("Erreur inattendue risk check")
            return {"allowed": False, "reason": f"Erreur inattendue: {e}"}

    def calculate_position_size(self, symbol: str, stop_loss: float) -> float:
        """
        Calcule la taille de position optimale selon le risque par trade et le stop loss.
        Args:
            symbol (str): Symbole à trader.
            stop_loss (float): Distance stop loss (en points/pips).
        Returns:
            float: Taille de position (lots)
        """
        try:
            account = self.client._request("GET", "/account")
            balance = account.get("balance", 0.0)
            risk_amount = balance * self.risk_per_trade
            # Récupère la valeur du point (tick_value) via REST
            symbol_info = self.client._request("GET", f"/symbol/{symbol}")
            tick_value = symbol_info.get("tick_value", 1.0)
            position_size = risk_amount / (stop_loss * tick_value)
            position_size = min(position_size, self.max_position_size)
            return max(0.0, position_size)
        except Exception as e:
            logger.error(f"Erreur calcul taille position: {e}")
            return 0.0

    def update_after_trade(self, profit: float):
        """
        Met à jour les métriques de risque après un trade (PnL, peak balance).
        Args:
            profit (float): Profit du trade (peut être négatif).
        """
        try:
            account = self.client._request("GET", "/account")
            equity = account.get("equity", 0.0)
            self.last_balance = equity
            if self.peak_balance is None or equity > self.peak_balance:
                self.peak_balance = equity
            self.daily_pnl += profit
        except Exception as e:
            logger.error(f"Erreur update_after_trade: {e}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de risque courantes (drawdown, PnL, etc).
        Returns:
            dict: Métriques de risque
        """
        try:
            account = self.client._request("GET", "/account")
            equity = account.get("equity", 0.0)
            drawdown = (self.peak_balance - equity) / self.peak_balance if self.peak_balance else 0.0
            return {
                "drawdown": drawdown,
                "daily_pnl": self.daily_pnl,
                "peak_balance": self.peak_balance,
                "last_balance": self.last_balance,
            }
        except Exception as e:
            logger.error(f"Erreur get_risk_metrics: {e}")
            return {} 