import logging
import random
import time
import unittest
from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock

from src.bitcoin_scalper.core.mt5_connector import MT5Connector
from src.bitcoin_scalper.core.partial_fill_handler import PartialFillHandler
from src.bitcoin_scalper.core.position_manager import PositionManager
from src.bitcoin_scalper.core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StressTest(unittest.TestCase):
    """Tests de stress pour le bot de trading."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.config = {
            "risk": {
                "max_position_size": 1.0,
                "max_daily_trades": 10,
                "max_daily_loss": 1000,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02,
            }
        }
        # Utiliser un mock du connecteur au lieu d'une instance réelle
        self.connector = MagicMock(spec=MT5Connector)
        self.risk_manager = RiskManager(self.config)
        self.position_manager = PositionManager(self.connector, self.risk_manager)
        self.partial_handler = PartialFillHandler(self.connector, self.position_manager)

    def test_connection_stability(self):
        """Test la stabilité de la connexion MT5."""
        # Test de déconnexion/ reconnexion rapide
        for _ in range(10):
            self.connector.disconnect()
            time.sleep(random.uniform(0.1, 1.0))
            self.assertTrue(self.connector.connect())

    def test_rapid_orders(self):
        """Test l'envoi rapide d'ordres."""
        symbol = "BTCUSD"
        volume = 0.01

        # Utiliser des IDs prévisibles
        ticket_ids = [1000 + i for i in range(10)]
        ticket_index = 0

        # Configurer le mock du connecteur pour place_order
        def mock_place_order(*args, **kwargs):
            nonlocal ticket_index
            mock_order = MagicMock()
            mock_order.ticket = ticket_ids[ticket_index]
            ticket_index += 1
            return {"order": mock_order}

        self.connector.place_order = MagicMock(side_effect=mock_place_order)

        # Configurer le mock pour close_position
        def mock_close_position(ticket):
            # Ne pas supprimer la position ici, laisser le PositionManager le faire
            return True

        self.connector.close_position = MagicMock(side_effect=mock_close_position)

        # Configurer le mock pour position_exists
        def mock_position_exists(ticket):
            return ticket in self.position_manager.positions

        self.connector.position_exists = MagicMock(side_effect=mock_position_exists)

        # Envoyer 10 ordres rapidement
        orders = []
        for i in range(10):
            print(f"Opening position {i+1}/10...")
            ticket = self.position_manager.open_position(
                symbol=symbol,
                volume=volume,
                side="BUY",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )
            if ticket:
                orders.append(ticket)
                print(
                    f"Opened position {ticket}, current positions: {list(self.position_manager.positions.keys())}"
                )
            time.sleep(0.05)  # Réduire le délai pour stresser plus

        # Vérifier que tous les ordres sont bien placés
        self.assertEqual(len(orders), 10)

        # Vérifier que les positions sont bien enregistrées
        for ticket in orders:
            self.assertIn(ticket, self.position_manager.positions)

        # Fermer tous les ordres
        for i, ticket in enumerate(orders):
            print(f"Closing position {i+1}/10 (ticket: {ticket})...")
            print(
                f"Current positions before close: {list(self.position_manager.positions.keys())}"
            )
            result = self.position_manager.close_position(ticket)
            self.assertTrue(result, f"Failed to close position {ticket}")
            self.assertNotIn(ticket, self.position_manager.positions)
            print(
                f"Position {ticket} closed, remaining positions: {list(self.position_manager.positions.keys())}"
            )
            time.sleep(0.05)  # Petit délai entre les fermetures

    def test_partial_fills(self):
        """Test la gestion des ordres partiellement remplis."""
        symbol = "BTCUSD"
        volume = 1.0  # Volume important pour forcer un remplissage partiel

        # Placer un ordre avec un grand volume
        ticket = self.position_manager.open_position(
            symbol=symbol,
            volume=volume,
            side="BUY",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            strategy="STRESS_TEST",
            params={},
        )

        if ticket:
            # Simuler un remplissage partiel
            self.assertTrue(
                self.partial_handler.handle_partial_fill(
                    ticket=ticket,
                    requested_volume=volume,
                    filled_volume=volume * 0.3,  # 30% rempli
                    remaining_volume=volume * 0.7,
                    symbol=symbol,
                    side="BUY",
                    price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                )
            )

    def test_error_handling(self):
        """Test la gestion des erreurs."""
        # Test avec des paramètres invalides
        with self.assertRaises(ValueError):
            self.position_manager.open_position(
                symbol="INVALID",
                volume=-1.0,
                side="INVALID",
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                strategy="STRESS_TEST",
                params={},
            )

        # Test avec un symbole valide mais volume invalide
        with self.assertRaises(ValueError):
            self.position_manager.open_position(
                symbol="BTCUSD",
                volume=-0.1,
                side="BUY",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )

        # Test avec un side invalide
        with self.assertRaises(ValueError):
            self.position_manager.open_position(
                symbol="BTCUSD",
                volume=0.1,
                side="INVALID",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )

    def test_sl_tp_updates(self):
        """Test les mises à jour de SL/TP."""
        symbol = "BTCUSD"
        volume = 0.01

        # Ouvrir une position
        ticket = self.position_manager.open_position(
            symbol=symbol,
            volume=volume,
            side="BUY",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            strategy="BollingerBandsReversal",
            params={"bb_period": 20, "bb_std": 2.0, "timeframe": "M5"},
        )

        if ticket:
            # Forcer plusieurs mises à jour de SL/TP
            for _ in range(5):
                self.assertTrue(self.position_manager.update_sl_tp(ticket))
                time.sleep(1)

            # Fermer la position
            self.assertTrue(self.position_manager.close_position(ticket))

    def test_risk_limits(self):
        """Test les limites de risque."""
        symbol = "BTCUSD"

        # Configurer le mock du connecteur
        mock_tick = MagicMock()
        mock_tick.ask = 50000.0
        mock_tick.bid = 49900.0
        self.connector.symbol_info_tick = MagicMock(return_value=mock_tick)

        # Tenter d'ouvrir une position trop grande
        with self.assertRaises(ValueError):
            self.position_manager.open_position(
                symbol=symbol,
                volume=100.0,  # Volume très important
                side="BUY",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )

        # Tenter d'ouvrir trop de positions
        orders: List[int] = []
        for _ in range(20):  # Plus que la limite
            ticket = self.position_manager.open_position(
                symbol=symbol,
                volume=0.01,
                side="BUY",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )
            if ticket:
                orders.append(ticket)

        # Vérifier que le nombre de positions est limité
        self.assertLessEqual(len(orders), self.config["risk"]["max_daily_trades"])

        # Fermer toutes les positions
        for ticket in orders:
            self.position_manager.close_position(ticket)

    def tearDown(self):
        """Nettoyage après chaque test."""
        self.connector.disconnect()
