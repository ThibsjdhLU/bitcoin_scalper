"""
Gestionnaire de risques REST multiplateforme pour trading BTC/USD.
Utilise exclusivement MT5RestClient (aucune dépendance native MT5, compatible macOS).
Intègre le calcul dynamique de SL/TP et des simulations Monte Carlo.
"""
from typing import Optional, Dict, Any, List
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient, MT5RestClientError
import logging
import numpy as np

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
            balance = account["balance"]
            equity = account["equity"]
            if self.peak_balance is None:
                self.peak_balance = balance
            if self.last_balance is None:
                self.last_balance = balance
            # Met à jour peak_balance si equity > peak_balance
            if equity > self.peak_balance:
                self.peak_balance = equity
            # Vérifie que volume est numérique
            if not isinstance(volume, (int, float)):
                self.last_balance = equity
                return {"allowed": False, "reason": "Erreur : volume non numérique"}
            # Drawdown
            drawdown = (self.peak_balance - equity) / self.peak_balance if self.peak_balance else 0.0
            # Calcul du PnL quotidien AVANT maj last_balance
            pnl = equity - self.last_balance
            daily_pnl_temp = self.daily_pnl + pnl
            seuil_daily = self.max_daily_loss * self.peak_balance
            logger.debug(f"[RiskManager] daily_pnl_temp={daily_pnl_temp}, seuil={seuil_daily}")
            self.daily_pnl = daily_pnl_temp  # Toujours cumuler le PnL
            if drawdown > self.max_drawdown:
                self.last_balance = equity
                return {"allowed": False, "reason": f"Drawdown max dépassé ({drawdown:.2%})"}
            if abs(daily_pnl_temp) > seuil_daily:
                self.last_balance = equity
                return {"allowed": False, "reason": f"Perte quotidienne max dépassée ({daily_pnl_temp:.2f})"}
            if volume > self.max_position_size:
                self.last_balance = equity
                return {"allowed": False, "reason": f"Taille position > max autorisé ({self.max_position_size})"}
            # Si tout est OK, on met à jour last_balance
            self.last_balance = equity
            return {"allowed": True, "reason": "OK"}
        except KeyError as e:
            logger.exception("Clé manquante dans la réponse du compte")
            return {"allowed": False, "reason": f"Erreur inattendue: clé manquante {e}"}
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
            if stop_loss <= 0 or tick_value <= 0:
                 return 0.0
            position_size = risk_amount / (stop_loss * tick_value)
            position_size = min(position_size, self.max_position_size)
            return max(0.0, position_size)
        except Exception as e:
            logger.error(f"Erreur calcul taille position: {e}")
            return 0.0

    def calculate_dynamic_stops(self, entry_price: float, atr: float, side: str, confidence: float, high_conf_threshold: float = 0.8) -> Dict[str, float]:
        """
        Calcule SL et TP dynamiques basés sur l'ATR et la confiance du modèle (Approche Hybride).

        Logique:
        - Si confiance haute (> high_conf_threshold) : On laisse respirer (SL plus large, k=2).
        - Si confiance moyenne : On resserre le SL (k=1.5 ou 1).

        Args:
            entry_price: Prix d'entrée
            atr: Valeur de l'ATR (volatilité)
            side: 'buy' ou 'sell'
            confidence: Probabilité du modèle (0.0 à 1.0)
            high_conf_threshold: Seuil de haute confiance

        Returns:
            Dict avec 'sl' et 'tp'
        """
        # Ajustement du facteur k selon la confiance
        if confidence > high_conf_threshold:
            k_sl = 2.0
            k_tp = 3.0 # R:R meilleur si haute confiance
        else:
            k_sl = 1.5
            k_tp = 2.0

        if side == 'buy':
            sl = entry_price - (atr * k_sl)
            tp = entry_price + (atr * k_tp)
        else: # sell
            sl = entry_price + (atr * k_sl)
            tp = entry_price - (atr * k_tp)

        return {"sl": sl, "tp": tp, "k_sl": k_sl}

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

class MonteCarloSimulator:
    """
    Simulateur Monte Carlo pour estimer la robustesse de la stratégie.
    À utiliser hors ligne ou périodiquement, PAS en temps réel critique.
    """
    def __init__(self, pnl_history: List[float], initial_capital: float = 10000.0):
        self.pnl_history = np.array(pnl_history)
        self.initial_capital = initial_capital

    def run_simulation(self, n_simulations: int = 1000, n_trades: int = 100) -> Dict[str, float]:
        """
        Exécute n_simulations de n_trades en échantillonnant l'historique PnL.

        Returns:
            Dict contenant :
            - risk_of_ruin: % de simulations finissant ruinées (capital <= 0)
            - max_drawdown_95: Drawdown max au 95ème percentile (pire cas réaliste)
            - median_final_capital: Capital final médian
        """
        if len(self.pnl_history) < 10:
            logger.warning("Historique PnL insuffisant pour Monte Carlo.")
            return {"risk_of_ruin": 0.0, "max_drawdown_95": 0.0, "median_final_capital": self.initial_capital}

        final_capitals = []
        max_drawdowns = []
        ruined_count = 0

        for _ in range(n_simulations):
            # Echantillonnage avec remise
            trades = np.random.choice(self.pnl_history, size=n_trades, replace=True)
            equity_curve = np.cumsum(trades) + self.initial_capital

            # Check Ruin
            if np.any(equity_curve <= 0):
                ruined_count += 1
                final_capitals.append(0)
                max_drawdowns.append(1.0) # 100% DD
                continue

            final_capitals.append(equity_curve[-1])

            # Calculate Max Drawdown for this sim
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            max_drawdowns.append(np.max(dd))

        return {
            "risk_of_ruin": ruined_count / n_simulations,
            "max_drawdown_95": np.percentile(max_drawdowns, 95),
            "median_final_capital": np.median(final_capitals)
        }
