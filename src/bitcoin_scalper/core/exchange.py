#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interface avec AvaTrade MT5
"""

import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

from .mt5_connection import MT5Connection


class Exchange:
    """Interface avec AvaTrade MT5"""

    def __init__(self, login: int, password: str, server: str):
        """
        Initialise l'interface avec AvaTrade MT5

        Args:
            login: Identifiant du compte
            password: Mot de passe du compte
            server: Nom du serveur AvaTrade
        """
        self.logger = logging.getLogger("TradingBot.Exchange")

        # Chemin de l'application MT5
        mt5_path = "C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe"

        # Vérification si MT5 est installé
        if not os.path.exists(mt5_path):
            self.logger.error(f"MetaTrader 5 n'est pas installé dans {mt5_path}")
            self.logger.info(
                "Vérifiez que le chemin est correct: C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe"
            )
            return

        # Utilisation de la connexion singleton
        self.mt5_connection = MT5Connection()
        if not self.mt5_connection.initialize(login, password, server):
            self.logger.error("Échec de l'initialisation de la connexion MT5")
            return

        self.logger.info("Connexion MT5 établie avec succès")

    def __del__(self):
        """Ferme la connexion MT5"""
        mt5.shutdown()

    def get_market_data(
        self, symbol: str, timeframe: str, num_bars: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les données du marché

        Args:
            symbol: Symbole à trader
            timeframe: Intervalle de temps
            num_bars: Nombre de bougies à récupérer

        Returns:
            Dict contenant les données du marché ou None en cas d'erreur
        """
        try:
            # Vérification de la connexion
            if not mt5.terminal_info().connected:
                self.logger.error(
                    "MetaTrader 5 n'est pas connecté. Veuillez vérifier la connexion."
                )
                return None

            # Conversion du timeframe en format MT5
            tf_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }

            tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)
            if rates is None:
                self.logger.error(
                    f"Erreur lors de la récupération des données: {mt5.last_error()}"
                )
                return None

            # Conversion en format standard
            closes = [rate[4] for rate in rates]  # Prix de clôture
            highs = [rate[2] for rate in rates]  # Plus haut
            lows = [rate[3] for rate in rates]  # Plus bas
            volumes = [rate[5] for rate in rates]  # Volume

            return {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": volumes,
                "timestamp": [rate[0] for rate in rates],
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel

        Args:
            symbol: Symbole à trader

        Returns:
            Prix actuel ou None en cas d'erreur
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(
                    f"Erreur lors de la récupération du prix: {mt5.last_error()}"
                )
                return None

            return tick.ask  # Prix d'achat actuel

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du prix: {e}")
            return None

    def get_balance(self) -> Optional[float]:
        """
        Récupère le solde disponible

        Returns:
            Solde disponible ou None en cas d'erreur
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(
                    f"Erreur lors de la récupération du solde: {mt5.last_error()}"
                )
                return None

            return account_info.balance

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du solde: {e}")
            return None

    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Récupère les positions ouvertes

        Args:
            symbol: Symbole spécifique à filtrer (optionnel)

        Returns:
            Liste des positions ouvertes
        """
        try:
            positions = []
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                self.logger.error(
                    f"Erreur lors de la récupération des positions: {mt5.last_error()}"
                )
                return []

            return [
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                }
                for pos in positions
            ]

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des positions: {e}")
            return []

    def create_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float = None,
        sl: float = None,
        tp: float = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordre sur MT5

        Args:
            symbol: Symbole à trader
            order_type: Type d'ordre ('BUY' ou 'SELL')
            volume: Volume à trader
            price: Prix d'entrée (optionnel, marché par défaut)
            sl: Stop Loss (optionnel)
            tp: Take Profit (optionnel)

        Returns:
            Détails de l'ordre ou None en cas d'erreur
        """
        try:
            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY
                if order_type == "BUY"
                else mt5.ORDER_TYPE_SELL,
                "price": price if price else mt5.symbol_info_tick(symbol).ask,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "python-bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Envoi de l'ordre
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Erreur lors du placement de l'ordre: {result.comment}"
                )
                return None

            return {
                "ticket": result.order,
                "volume": volume,
                "price": request["price"],
                "sl": sl,
                "tp": tp,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            return None

    def close_position(self, ticket: int) -> bool:
        """
        Ferme une position

        Args:
            ticket: Numéro de ticket de la position

        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} non trouvée")
                return False

            position = position[0]

            # Préparation de la requête de fermeture
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL
                if position.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "python-bot-close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Envoi de la requête
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Erreur lors de la fermeture de la position: {result.comment}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la fermeture de la position: {e}")
            return False
