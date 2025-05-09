"""
Gestionnaire de risques pour le bot de trading.
"""
from typing import Any, Dict, Optional, Tuple

from loguru import logger


class RiskManager:
    """Gestionnaire de risques pour le bot de trading."""

    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de risques.

        Args:
            config: Configuration du gestionnaire de risques
        """
        self.max_position_size = config["risk"]["max_position_size"]
        self.max_daily_trades = config["risk"]["max_daily_trades"]
        self.max_daily_loss = config["risk"]["max_daily_loss"]
        self.max_drawdown = config["risk"]["max_drawdown"]
        self.risk_per_trade = config["risk"]["risk_per_trade"]

        # État du gestionnaire
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 10000.0  # Balance initiale
        self.current_balance = 10000.0  # Balance initiale
        self.trades = []
        self.strategy_trades = {}  # Compteur de trades par stratégie

        logger.info(f"Risk Manager initialisé avec les paramètres suivants:")
        logger.info(f"- Taille max position: {self.max_position_size}")
        logger.info(f"- Trades quotidiens max: {self.max_daily_trades}")
        logger.info(f"- Perte quotidienne max: {self.max_daily_loss}")
        logger.info(f"- Drawdown max: {self.max_drawdown}")
        logger.info(f"- Risque par trade: {self.risk_per_trade}")

    def validate_position(
        self,
        symbol: str,
        volume: float,
        side: str,
        stop_loss: float,
        take_profit: float,
        mt5_connector,
    ) -> bool:
        """Valide une position avant son ouverture."""
        if volume <= 0:
            logger.warning(f"Volume invalide: {volume} <= 0")
            raise ValueError("Le volume doit être positif")

        if volume > self.max_position_size:
            logger.warning(
                f"Volume {volume} supérieur au maximum autorisé {self.max_position_size}"
            )
            raise ValueError(
                f"Volume {volume} supérieur au maximum autorisé {self.max_position_size}"
            )

        if side not in ["BUY", "SELL"]:
            logger.warning(f"Side invalide: {side}")
            raise ValueError(f"Side invalide: {side}. Doit être 'BUY' ou 'SELL'")

        if not self.can_open_position(symbol, volume, mt5_connector):
            logger.warning(
                f"Position rejetée pour {symbol} - Volume: {volume}, Side: {side}"
            )
            raise ValueError("Limites de risque dépassées")

        logger.info(f"Position validée pour {symbol} - Volume: {volume}, Side: {side}")
        return True

    def can_open_position(
        self, symbol: str, volume: float, strategy: str = None
    ) -> Tuple[bool, float]:
        """
        Vérifie si une nouvelle position peut être ouverte.

        Args:
            symbol: Symbole à trader
            volume: Volume souhaité
            strategy: Nom de la stratégie (optionnel)

        Returns:
            Tuple[bool, float]: (Position autorisée, Volume ajusté)
        """
        try:
            # Vérifier le nombre de trades quotidiens
            if self.daily_trades >= self.max_daily_trades:
                logger.warning(
                    f"Nombre maximum de trades quotidiens atteint ({self.max_daily_trades})"
                )
                return False, 0.0

            # Vérifier la perte quotidienne
            if abs(self.daily_pnl) >= self.max_daily_loss:
                logger.warning(
                    f"Perte quotidienne maximale atteinte ({self.max_daily_loss})"
                )
                return False, 0.0

            # Vérifier le drawdown
            current_drawdown = (
                self.peak_balance - self.current_balance
            ) / self.peak_balance
            if current_drawdown >= self.max_drawdown:
                logger.warning(f"Drawdown maximal atteint ({self.max_drawdown})")
                return False, 0.0

            # Ajuster le volume si nécessaire
            adjusted_volume = min(volume, self.max_position_size)
            if adjusted_volume != volume:
                logger.info(f"Volume ajusté de {volume} à {adjusted_volume}")

            # Afficher l'état actuel
            logger.info(
                f"État actuel - Trades: {self.daily_trades}/{self.max_daily_trades}, PnL: {self.daily_pnl:.2f}, Drawdown: {current_drawdown:.2%}"
            )

            return True, adjusted_volume

        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la position: {str(e)}")
            return False, 0.0

    def calculate_position_size(
        self, account_balance: float, stop_loss_pips: float, symbol: str, mt5_connector
    ) -> float:
        """
        Calcule la taille de position optimale.

        Args:
            account_balance: Solde du compte
            stop_loss_pips: Stop loss en pips
            symbol: Symbole de trading
            mt5_connector: Connecteur MT5

        Returns:
            float: Taille de position en lots
        """
        # Récupérer le tick value
        symbol_info = mt5_connector.symbol_info_tick(symbol)
        tick_value = (symbol_info.ask - symbol_info.bid) * 10

        # Calculer le risque monétaire
        risk_amount = account_balance * self.risk_per_trade

        # Calculer la taille de position
        position_size = risk_amount / (stop_loss_pips * tick_value)

        # Limiter au maximum autorisé
        position_size = min(position_size, self.max_position_size)

        logger.info(f"Calcul taille position pour {symbol}:")
        logger.info(f"- Balance: {account_balance:.2f}")
        logger.info(f"- Stop loss (pips): {stop_loss_pips}")
        logger.info(f"- Risque monétaire: {risk_amount:.2f}")
        logger.info(f"- Taille position calculée: {position_size:.4f}")

        return position_size

    def update_trade(self, profit: float, strategy: str = None):
        """Met à jour les métriques après un trade."""
        self.current_balance += profit
        self.daily_pnl += profit
        self.daily_trades += 1

        # Mettre à jour le peak balance si profit positif
        if profit > 0:
            self.peak_balance = max(self.peak_balance, self.current_balance)

        # Mettre à jour le compteur de trades par stratégie
        if strategy:
            if strategy not in self.strategy_trades:
                self.strategy_trades[strategy] = 0
            self.strategy_trades[strategy] += 1

        self.trades.append(
            {
                "profit": profit,
                "strategy": strategy,
                "timestamp": None,  # À implémenter si nécessaire
            }
        )

        logger.info(f"Trade mis à jour - Profit: {profit:.2f}, Strategy: {strategy}")
        logger.info(
            f"État actuel - Balance: {self.current_balance:.2f}, PnL quotidien: {self.daily_pnl:.2f}, Trades: {self.daily_trades}"
        )

    def get_risk_metrics(self) -> dict:
        """
        Retourne les métriques de risque.

        Returns:
            dict: Métriques de risque
        """
        if not self.trades:
            logger.info("Aucun trade effectué")
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "total_trades": 0,
                "total_profit": 0.0,
            }

        wins = [t["profit"] for t in self.trades if t["profit"] > 0]
        losses = [t["profit"] for t in self.trades if t["profit"] < 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

        metrics = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "total_trades": len(self.trades),
            "total_profit": sum(t["profit"] for t in self.trades),
        }

        logger.info("Métriques de risque actuelles:")
        logger.info(f"- Win rate: {win_rate:.2%}")
        logger.info(f"- Profit factor: {profit_factor:.2f}")
        logger.info(f"- Gain moyen: {avg_win:.2f}")
        logger.info(f"- Perte moyenne: {avg_loss:.2f}")
        logger.info(f"- Total trades: {len(self.trades)}")
        logger.info(f"- Profit total: {metrics['total_profit']:.2f}")

        return metrics
