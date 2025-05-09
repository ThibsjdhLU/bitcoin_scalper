#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bot de trading automatisé
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..analysis.indicators import TechnicalIndicators
from ..connectors.metatrader import Exchange
from .config import Config


class TradingBot:
    """Bot de trading automatisé"""

    def __init__(self, config_path: str = "config/.env"):
        """
        Initialise le bot de trading

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Configuration
        self.config = Config(config_path)
        self._setup_logging()

        # Composants
        self.exchange = Exchange(
            login=int(self.config.get("mt5.login", 0)),
            password=self.config.get("mt5.password", ""),
            server=self.config.get("mt5.server", "Ava-Demo 1-MT5"),
        )

        self.indicators = TechnicalIndicators(
            rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )

        # État
        self.running = False
        self.last_trade_time = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_prices: List[float] = []
        self.ui_queue = None

    def set_ui_queue(self, queue):
        """Configure la queue de communication avec l'interface"""
        self.ui_queue = queue

    def _send_to_ui(self, message_type: str, data: Any = None):
        """Envoie un message à l'interface utilisateur"""
        if self.ui_queue:
            self.ui_queue.put({"type": message_type, "data": data})

    def _setup_logging(self) -> None:
        """Configure le système de logging"""
        log_level = getattr(logging, self.config.get("logging.level", "INFO"))
        log_file = self.config.get("logging.file", "logs/bitcoin_scalper.log")

        # Création du dossier logs s'il n'existe pas
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger("TradingBot")

    def start(self) -> None:
        """Démarre le bot de trading"""
        self.logger.info("Démarrage du bot de trading")
        self.running = True

        while self.running:
            try:
                self._process_trading_cycle()
                time.sleep(
                    1
                )  # Réduit à 1 seconde pour des mises à jour plus fréquentes

            except KeyboardInterrupt:
                self.logger.info("Arrêt demandé par l'utilisateur")
                self.stop()

            except Exception as e:
                self.logger.error(f"Erreur lors du cycle de trading: {e}")
                self._send_to_ui("log", f"Erreur: {str(e)}")
                time.sleep(5)  # Attente plus courte en cas d'erreur

    def stop(self) -> None:
        """Arrête le bot de trading"""
        self.logger.info("Arrêt du bot de trading")
        self.running = False

    def _process_trading_cycle(self) -> None:
        """Exécute un cycle de trading"""
        try:
            # Récupération des données de marché
            symbol = self.config.get("mt5.symbol", "BTCUSD")
            timeframe = self.config.get("mt5.timeframe", "1m")

            market_data = self.exchange.get_market_data(symbol, timeframe)
            if market_data is not None and not market_data.empty:
                # Mise à jour des derniers prix
                self.last_prices = market_data["close"].tolist()

                # Envoi des données à l'interface
                self._send_to_ui(
                    "market_data",
                    {
                        "prices": self.last_prices,
                        "time": market_data.index[-1].strftime("%H:%M:%S"),
                    },
                )

                # Calcul des indicateurs
                signals = self.indicators.analyze_signals(market_data["close"])

                # Envoi des indicateurs à l'interface
                self._send_to_ui(
                    "indicators",
                    {
                        "rsi": self.indicators.rsi[-1],
                        "macd": self.indicators.macd[-1],
                        "signal": self.indicators.signal[-1],
                        "histogram": self.indicators.histogram[-1],
                    },
                )

                # Vérification des limites quotidiennes
                if self._check_daily_limits():
                    # Analyse des signaux
                    if signals:
                        self._process_signals(signals, symbol)

                # Mise à jour des informations du compte
                balance = self.exchange.get_balance()
                if balance is not None:
                    self._send_to_ui(
                        "account_info",
                        {
                            "balance": balance,
                            "equity": balance,  # Pour simplifier
                            "profit": self.daily_pnl,
                        },
                    )

        except Exception as e:
            self.logger.error(f"Erreur dans le cycle de trading: {e}")
            self._send_to_ui("log", f"Erreur dans le cycle: {str(e)}")

    def _check_daily_limits(self) -> bool:
        """
        Vérifie les limites quotidiennes

        Returns:
            bool: True si les limites sont respectées
        """
        # Réinitialisation quotidienne
        current_time = time.time()
        if current_time - self.last_trade_time > 86400:  # 24 heures
            self.daily_trades = 0
            self.daily_pnl = 0.0

        # Vérification des limites
        max_trades = 5  # Limite fixe de 5 trades par jour
        max_loss = 5.0  # Limite fixe de 5% de perte par jour

        if self.daily_trades >= max_trades:
            self.logger.info("Limite quotidienne de trades atteinte")
            return False

        if self.daily_pnl <= -max_loss:
            self.logger.info("Limite de perte quotidienne atteinte")
            return False

        return True

    def _process_signals(self, signals: Dict[str, Any], symbol: str) -> None:
        """
        Traite les signaux de trading

        Args:
            signals: Signaux calculés
            symbol: Symbole à trader
        """
        # Signal d'achat
        if signals.get("buy"):
            # Calcul de la taille de la position
            balance = self.exchange.get_balance()
            if balance is None:
                return

            # Calcul du volume en fonction du risque
            risk_percent = self.config.get("trading.risk_percent", 1.0)
            volume_min = self.config.get("trading.volume_min", 0.01)
            volume_max = self.config.get("trading.volume_max", 1.0)

            position_size = self.indicators.get_position_size(balance, risk_percent)

            # Vérification des limites de volume
            position_size = max(min(position_size, volume_max), volume_min)

            # Placement de l'ordre
            order = self.exchange.create_order(
                symbol=symbol, order_type="BUY", volume=position_size
            )

            if order:
                self.logger.info(f"Ordre d'achat placé: {order}")
                self.last_trade_time = time.time()
                self.daily_trades += 1

        # Signal de vente
        elif signals.get("sell"):
            # Récupération des positions ouvertes
            positions = self.exchange.get_open_positions(symbol)
            for position in positions:
                if position["type"] == "BUY":
                    # Fermeture de la position
                    if self.exchange.close_position(position["ticket"]):
                        self.logger.info(f"Position fermée: {position}")
                        self.last_trade_time = time.time()
                        self.daily_trades += 1
                        self.daily_pnl += position["profit"]

    def run_bot(self):
        """
        Fonction principale du bot de trading.
        """
        try:
            print("\n=== Démarrage du bot de trading ===")

            while True:
                try:
                    # Récupération des données de marché
                    market_data = self.exchange.get_market_data(
                        self.config.get("mt5.symbol", "BTCUSD"),
                        self.config.get("mt5.timeframe", "1m"),
                    )
                    print("\nDonnées de marché récupérées")

                    # Analyse du marché
                    signal = self.indicators.analyze_signals(market_data["close"])
                    print(f"\nSignal généré: {signal}")

                    if signal.get("buy") or signal.get("sell"):
                        # Validation du signal
                        if self._check_daily_limits():
                            # Calcul de la taille de position
                            position_size = self.indicators.get_position_size(
                                self.exchange.get_balance(),
                                self.config.get("trading.risk_percent", 1.0),
                            )
                            print(
                                f"\nTaille de position calculée: {position_size:.8f} BTC"
                            )

                            # Exécution de l'ordre
                            if signal.get("buy"):
                                print("\nExécution de l'ordre d'achat...")
                                self.exchange.create_order(
                                    symbol=self.config.get("mt5.symbol", "BTCUSD"),
                                    order_type="BUY",
                                    volume=position_size,
                                )
                            else:
                                print("\nExécution de l'ordre de vente...")
                                self.exchange.close_position(
                                    self.exchange.get_open_positions(
                                        self.config.get("mt5.symbol", "BTCUSD")
                                    )[0]["ticket"]
                                )

                            print("Ordre exécuté avec succès")
                        else:
                            print("\nSignal rejeté par le gestionnaire de risques")

                    # Mise à jour du solde
                    self.exchange.fetch_balance()
                    print(f"\nSolde mis à jour: {self.exchange.get_balance():.2f} USDT")

                    # Attente avant la prochaine itération
                    time.sleep(self.config.get("trading_interval", 60))

                except Exception as e:
                    print(f"\nErreur lors de l'itération: {str(e)}")
                    time.sleep(60)  # Attente plus longue en cas d'erreur

        except KeyboardInterrupt:
            print("\nArrêt du bot...")
        except Exception as e:
            print(f"\nErreur critique: {str(e)}")

        self.stop()
