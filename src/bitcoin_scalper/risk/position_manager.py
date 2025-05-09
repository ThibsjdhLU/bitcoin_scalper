import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Gère les positions ouvertes et leurs SL/TP dynamiques.
    """

    def __init__(self, connector, risk_manager):
        self.connector = connector
        self.risk_manager = risk_manager
        self.positions: Dict[int, Dict] = {}  # ticket -> position info
        self.update_interval = 60  # secondes

    def open_position(
        self,
        symbol: str,
        volume: float,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        params: Dict,
    ) -> Optional[int]:
        """
        Ouvre une nouvelle position avec SL/TP.

        Args:
            symbol: Symbole à trader
            volume: Volume de la position
            side: Sens de la position ('BUY' ou 'SELL')
            entry_price: Prix d'entrée
            stop_loss: Stop loss initial
            take_profit: Take profit initial
            strategy: Nom de la stratégie
            params: Paramètres de la stratégie

        Returns:
            Optional[int]: Ticket de la position ou None si échec

        Raises:
            ValueError: Si les paramètres sont invalides ou si les limites de risque sont dépassées
        """
        # Vérifier le risque
        self.risk_manager.validate_position(
            symbol, volume, side, stop_loss, take_profit, self.connector
        )

        try:
            # Placer l'ordre
            result = self.connector.place_order(
                symbol=symbol,
                order_type="MARKET",
                volume=volume,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"{strategy} {side}",
            )

            if not result or "order" not in result:
                logger.error(f"Échec d'ouverture de position: {result}")
                return None

            # Récupérer le ticket de l'ordre
            ticket = result["order"].ticket

            # Enregistrer la position
            self.positions[ticket] = {
                "symbol": symbol,
                "volume": volume,
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy": strategy,
                "params": params,
                "open_time": datetime.now(),
                "last_update": datetime.now(),
            }

            logger.info(f"Position ouverte: {ticket} ({symbol} {side} {volume})")
            return ticket

        except Exception as e:
            logger.error(
                f"Erreur lors de l'ouverture de position: {str(e)}", exc_info=True
            )
            return None

    def update_sl_tp(self, ticket: int) -> bool:
        """
        Met à jour les SL/TP d'une position selon sa stratégie.

        Args:
            ticket: Ticket de la position

        Returns:
            bool: True si mise à jour réussie
        """
        if ticket not in self.positions:
            logger.error(f"Position {ticket} non trouvée")
            return False

        position = self.positions[ticket]

        # Vérifier si une mise à jour est nécessaire
        if (datetime.now() - position["last_update"]).seconds < self.update_interval:
            return True

        try:
            # Récupérer les données récentes
            data = self.connector.get_rates(
                position["symbol"],
                timeframe=position["params"].get("timeframe", "M5"),
                start_pos=0,
                count=100,
            )

            if data is None:
                return False

            df = pd.DataFrame(data)

            # Calculer les nouveaux niveaux selon la stratégie
            if position["strategy"] == "BollingerBandsReversal":
                new_sl, new_tp = self._calculate_bb_levels(df, position)
            elif position["strategy"] == "MACD":
                new_sl, new_tp = self._calculate_macd_levels(df, position)
            else:
                return True  # Pas de mise à jour pour cette stratégie

            # Vérifier si les niveaux ont changé significativement
            if (
                abs(new_sl - position["stop_loss"]) > 0.0001
                or abs(new_tp - position["take_profit"]) > 0.0001
            ):
                # Modifier l'ordre
                if self.connector.modify_order(
                    ticket=ticket, stop_loss=new_sl, take_profit=new_tp
                ):
                    position["stop_loss"] = new_sl
                    position["take_profit"] = new_tp
                    position["last_update"] = datetime.now()
                    logger.info(
                        f"SL/TP mis à jour pour {ticket}: SL={new_sl}, TP={new_tp}"
                    )
                    return True

            return True

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour SL/TP: {str(e)}")
            return False

    def _calculate_bb_levels(
        self, data: pd.DataFrame, position: Dict
    ) -> Tuple[float, float]:
        """Calcule les niveaux SL/TP pour Bollinger Bands."""
        from ..strategies.bollinger_bands_reversal import \
            calculate_bollinger_bands

        upper, middle, lower = calculate_bollinger_bands(
            data["close"],
            period=position["params"]["bb_period"],
            num_std=position["params"]["bb_std"],
        )

        if position["side"] == "BUY":
            new_sl = min(
                data["low"].iloc[-position["params"]["bb_period"] :].min(),
                middle.iloc[-1],
            )
            new_tp = upper.iloc[-1]
        else:
            new_sl = max(
                data["high"].iloc[-position["params"]["bb_period"] :].max(),
                middle.iloc[-1],
            )
            new_tp = lower.iloc[-1]

        return new_sl, new_tp

    def _calculate_macd_levels(
        self, data: pd.DataFrame, position: Dict
    ) -> Tuple[float, float]:
        """Calcule les niveaux SL/TP pour MACD."""
        from ..strategies.macd_strategy import calculate_atr

        current_price = data["close"].iloc[-1]
        atr = calculate_atr(
            data["high"], data["low"], data["close"], position["params"]["atr_period"]
        ).iloc[-1]

        if position["side"] == "BUY":
            new_sl = current_price - (
                position["params"]["stop_loss_atr_multiplier"] * atr
            )
            new_tp = current_price + (
                position["params"]["take_profit_atr_multiplier"] * atr
            )
        else:
            new_sl = current_price + (
                position["params"]["stop_loss_atr_multiplier"] * atr
            )
            new_tp = current_price - (
                position["params"]["take_profit_atr_multiplier"] * atr
            )

        return new_sl, new_tp

    def close_position(self, ticket: int) -> bool:
        """
        Ferme une position.

        Args:
            ticket: Ticket de la position

        Returns:
            bool: True si fermeture réussie
        """
        try:
            # Vérifier si la position existe dans notre dictionnaire
            if ticket not in self.positions:
                logger.warning(f"Position {ticket} non trouvée dans le gestionnaire")
                return True  # Considérer comme un succès si la position n'existe plus

            # Vérifier si la position existe toujours dans MT5
            if not self.connector.position_exists(ticket):
                logger.warning(f"Position {ticket} n'existe plus dans MT5")
                del self.positions[ticket]
                return True

            # Fermer la position
            success = self.connector.close_position(ticket)

            if success:
                logger.info(f"Position fermée: {ticket}")
                del self.positions[ticket]
                return True

            logger.error(f"Échec de fermeture de la position {ticket}")
            return False

        except Exception as e:
            logger.error(
                f"Erreur lors de la fermeture de position: {str(e)}", exc_info=True
            )
            return False

    def update_all_positions(self) -> None:
        """Met à jour toutes les positions ouvertes."""
        for ticket in list(self.positions.keys()):
            self.update_sl_tp(ticket)
