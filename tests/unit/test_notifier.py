"""
Tests unitaires pour le module de notification.
"""
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from src.bitcoin_scalper.utils.notifier import AlertType, EmailNotifier


class TestEmailNotifier(unittest.TestCase):
    """Tests pour le notifier email."""

    def setUp(self):
        """Initialise les données de test."""
        self.notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="bot@test.com",
            sender_password="password123",
            recipient_email="trader@test.com",
        )

    def test_validate_config(self):
        """Teste la validation de la configuration."""
        # Test avec des paramètres invalides
        with self.assertRaises(ValueError):
            EmailNotifier(
                smtp_server="",  # serveur vide
                smtp_port=587,
                sender_email="bot@test.com",
                sender_password="password123",
                recipient_email="trader@test.com",
            )

        with self.assertRaises(ValueError):
            EmailNotifier(
                smtp_server="smtp.gmail.com",
                smtp_port=-1,  # port invalide
                sender_email="bot@test.com",
                sender_password="password123",
                recipient_email="trader@test.com",
            )

        with self.assertRaises(ValueError):
            EmailNotifier(
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                sender_email="invalid_email",  # email invalide
                sender_password="password123",
                recipient_email="trader@test.com",
            )

    def test_format_signal_alert(self):
        """Teste le formatage des alertes de signal."""
        message = self.notifier._format_signal_alert(
            symbol="BTC/USD",
            signal_type="BUY",
            price=50000.0,
            strategy_name="MACD Strategy",
            metadata={"macd": 100.5, "signal": 90.2, "trend": "bullish"},
        )

        # Vérifier que le message contient les informations importantes
        self.assertIn("BTC/USD", message)
        self.assertIn("BUY", message)
        self.assertIn("50000.00", message)
        self.assertIn("MACD Strategy", message)
        self.assertIn("100.5000", message)
        self.assertIn("90.2000", message)
        self.assertIn("bullish", message)

    def test_format_order_alert(self):
        """Teste le formatage des alertes d'ordre."""
        message = self.notifier._format_order_alert(
            symbol="BTC/USD",
            order_type="MARKET",
            price=50000.0,
            volume=0.1,
            status="FILLED",
            metadata={"stop_loss": 49000.0, "take_profit": 51000.0},
        )

        # Vérifier que le message contient les informations importantes
        self.assertIn("BTC/USD", message)
        self.assertIn("MARKET", message)
        self.assertIn("50000.00", message)
        self.assertIn("0.1000", message)
        self.assertIn("FILLED", message)
        self.assertIn("49000.0000", message)
        self.assertIn("51000.0000", message)

    def test_format_risk_alert(self):
        """Teste le formatage des alertes de risque."""
        message = self.notifier._format_risk_alert(
            alert_type="DRAWDOWN",
            current_value=15.5,
            threshold=10.0,
            metadata={"balance": 10000.0, "equity": 8500.0},
        )

        # Vérifier que le message contient les informations importantes
        self.assertIn("DRAWDOWN", message)
        self.assertIn("15.50", message)
        self.assertIn("10.00", message)
        self.assertIn("10000.0000", message)
        self.assertIn("8500.0000", message)

    @patch("smtplib.SMTP")
    def test_send_alert(self, mock_smtp):
        """Teste l'envoi d'alertes."""
        # Configurer le mock
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Tester l'envoi d'une alerte
        success = self.notifier.send_alert(
            alert_type=AlertType.SIGNAL,
            subject="Test Alert",
            message="This is a test alert",
        )

        # Vérifier que l'alerte a été envoyée
        self.assertTrue(success)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with(
            self.notifier.sender_email, self.notifier.sender_password
        )
        mock_server.send_message.assert_called_once()

        # Tester avec une erreur SMTP
        mock_server.send_message.side_effect = Exception("SMTP Error")
        success = self.notifier.send_alert(
            alert_type=AlertType.ERROR, subject="Test Error", message="This should fail"
        )

        # Vérifier que l'erreur est gérée
        self.assertFalse(success)

    @patch("smtplib.SMTP")
    def test_send_signal_alert(self, mock_smtp):
        """Teste l'envoi d'alertes de signal."""
        # Configurer le mock
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Tester l'envoi d'une alerte de signal
        success = self.notifier.send_signal_alert(
            symbol="BTC/USD",
            signal_type="BUY",
            price=50000.0,
            strategy_name="MACD Strategy",
            metadata={"macd": 100.5, "signal": 90.2},
        )

        # Vérifier que l'alerte a été envoyée
        self.assertTrue(success)
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_order_alert(self, mock_smtp):
        """Teste l'envoi d'alertes d'ordre."""
        # Configurer le mock
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Tester l'envoi d'une alerte d'ordre
        success = self.notifier.send_order_alert(
            symbol="BTC/USD",
            order_type="MARKET",
            price=50000.0,
            volume=0.1,
            status="FILLED",
            metadata={"stop_loss": 49000.0, "take_profit": 51000.0},
        )

        # Vérifier que l'alerte a été envoyée
        self.assertTrue(success)
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_risk_alert(self, mock_smtp):
        """Teste l'envoi d'alertes de risque."""
        # Configurer le mock
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Tester l'envoi d'une alerte de risque
        success = self.notifier.send_risk_alert(
            alert_type="DRAWDOWN",
            current_value=15.5,
            threshold=10.0,
            metadata={"balance": 10000.0, "equity": 8500.0},
        )

        # Vérifier que l'alerte a été envoyée
        self.assertTrue(success)
        mock_server.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
