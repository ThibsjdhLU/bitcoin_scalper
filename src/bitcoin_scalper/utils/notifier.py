"""
Module de gestion des notifications.
Permet d'envoyer des alertes par email pour les événements importants du bot.
"""
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Union

from loguru import logger


class AlertType(Enum):
    """Types d'alertes possibles."""

    SIGNAL = "SIGNAL"  # Signal de trading détecté
    ORDER = "ORDER"  # Ordre exécuté
    ERROR = "ERROR"  # Erreur système
    INFO = "INFO"  # Information générale
    RISK = "RISK"  # Alerte de risque (drawdown, etc.)


class EmailNotifier:
    """
    Gestionnaire de notifications par email.

    Attributes:
        smtp_server: Serveur SMTP à utiliser
        smtp_port: Port du serveur SMTP
        sender_email: Adresse email d'envoi
        sender_password: Mot de passe de l'email d'envoi
        recipient_email: Adresse email de réception
    """

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
    ):
        """
        Initialise le notifier.

        Args:
            smtp_server: Serveur SMTP à utiliser
            smtp_port: Port du serveur SMTP
            sender_email: Adresse email d'envoi
            sender_password: Mot de passe de l'email d'envoi
            recipient_email: Adresse email de réception
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email

        # Vérifier la configuration
        self._validate_config()

        logger.info("Notifier email initialisé")

    def _validate_config(self) -> None:
        """
        Vérifie que la configuration est valide.

        Raises:
            ValueError: Si la configuration est invalide
        """
        if not all(
            [
                self.smtp_server,
                self.smtp_port,
                self.sender_email,
                self.sender_password,
                self.recipient_email,
            ]
        ):
            raise ValueError("Tous les paramètres de configuration sont requis")

        if not isinstance(self.smtp_port, int) or self.smtp_port <= 0:
            raise ValueError("Le port SMTP doit être un entier positif")

        if "@" not in self.sender_email or "@" not in self.recipient_email:
            raise ValueError("Les adresses email doivent être valides")

    def _format_signal_alert(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        strategy_name: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Formate une alerte de signal de trading.

        Args:
            symbol: Symbole concerné
            signal_type: Type de signal (BUY/SELL)
            price: Prix du signal
            strategy_name: Nom de la stratégie
            metadata: Métadonnées additionnelles

        Returns:
            str: Message formaté
        """
        message = f"""
        🔔 Nouveau signal de trading détecté !
        
        Symbol: {symbol}
        Type: {signal_type}
        Prix: {price:.2f}
        Stratégie: {strategy_name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if metadata:
            message += "\nMétadonnées:\n"
            for key, value in metadata.items():
                if isinstance(value, float):
                    message += f"{key}: {value:.4f}\n"
                else:
                    message += f"{key}: {value}\n"

        return message

    def _format_order_alert(
        self,
        symbol: str,
        order_type: str,
        price: float,
        volume: float,
        status: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Formate une alerte d'ordre exécuté.

        Args:
            symbol: Symbole concerné
            order_type: Type d'ordre
            price: Prix d'exécution
            volume: Volume de l'ordre
            status: Statut de l'ordre
            metadata: Métadonnées additionnelles

        Returns:
            str: Message formaté
        """
        message = f"""
        📊 Ordre de trading exécuté !
        
        Symbol: {symbol}
        Type: {order_type}
        Prix: {price:.2f}
        Volume: {volume:.4f}
        Statut: {status}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if metadata:
            message += "\nMétadonnées:\n"
            for key, value in metadata.items():
                if isinstance(value, float):
                    message += f"{key}: {value:.4f}\n"
                else:
                    message += f"{key}: {value}\n"

        return message

    def _format_risk_alert(
        self,
        alert_type: str,
        current_value: float,
        threshold: float,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Formate une alerte de risque.

        Args:
            alert_type: Type d'alerte de risque
            current_value: Valeur actuelle
            threshold: Seuil d'alerte
            metadata: Métadonnées additionnelles

        Returns:
            str: Message formaté
        """
        message = f"""
        ⚠️ Alerte de risque !
        
        Type: {alert_type}
        Valeur actuelle: {current_value:.2f}
        Seuil: {threshold:.2f}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if metadata:
            message += "\nMétadonnées:\n"
            for key, value in metadata.items():
                if isinstance(value, float):
                    message += f"{key}: {value:.4f}\n"
                else:
                    message += f"{key}: {value}\n"

        return message

    def send_alert(self, alert_type: AlertType, subject: str, message: str) -> bool:
        """
        Envoie une alerte par email.

        Args:
            alert_type: Type d'alerte
            subject: Sujet de l'email
            message: Corps du message

        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        try:
            # Créer le message
            email = MIMEMultipart()
            email["From"] = self.sender_email
            email["To"] = self.recipient_email
            email["Subject"] = f"[{alert_type.value}] {subject}"

            # Ajouter le corps du message
            email.attach(MIMEText(message, "plain"))

            # Se connecter au serveur SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)

                # Envoyer l'email
                server.send_message(email)

            logger.info(f"Alerte {alert_type.value} envoyée avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")
            return False

    def send_signal_alert(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        strategy_name: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Envoie une alerte de signal de trading.

        Args:
            symbol: Symbole concerné
            signal_type: Type de signal (BUY/SELL)
            price: Prix du signal
            strategy_name: Nom de la stratégie
            metadata: Métadonnées additionnelles

        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        subject = f"Signal {signal_type} sur {symbol}"
        message = self._format_signal_alert(
            symbol, signal_type, price, strategy_name, metadata
        )
        return self.send_alert(AlertType.SIGNAL, subject, message)

    def send_order_alert(
        self,
        symbol: str,
        order_type: str,
        price: float,
        volume: float,
        status: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Envoie une alerte d'ordre exécuté.

        Args:
            symbol: Symbole concerné
            order_type: Type d'ordre
            price: Prix d'exécution
            volume: Volume de l'ordre
            status: Statut de l'ordre
            metadata: Métadonnées additionnelles

        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        subject = f"Ordre {order_type} sur {symbol}"
        message = self._format_order_alert(
            symbol, order_type, price, volume, status, metadata
        )
        return self.send_alert(AlertType.ORDER, subject, message)

    def send_risk_alert(
        self,
        alert_type: str,
        current_value: float,
        threshold: float,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Envoie une alerte de risque.

        Args:
            alert_type: Type d'alerte de risque
            current_value: Valeur actuelle
            threshold: Seuil d'alerte
            metadata: Métadonnées additionnelles

        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        subject = f"Alerte de risque - {alert_type}"
        message = self._format_risk_alert(
            alert_type, current_value, threshold, metadata
        )
        return self.send_alert(AlertType.RISK, subject, message)


class Notifier:
    """
    Façade pour la gestion des notifications.
    Permet d'envoyer des notifications via différents canaux (email, etc.).
    """

    def __init__(self, config: Dict):
        """
        Initialise le notifier.

        Args:
            config: Configuration du notifier
        """
        self.email_notifier = None

        # Initialiser le notifier email si configuré
        if config.get("notifications", {}).get("email", {}).get("enabled", False):
            email_config = config["notifications"]["email"]
            self.email_notifier = EmailNotifier(
                smtp_server=email_config["smtp_server"],
                smtp_port=email_config["smtp_port"],
                sender_email=email_config["sender_email"],
                sender_password=email_config["sender_password"],
                recipient_email=email_config["recipient_email"],
            )

        logger.info("Notifier initialisé")

    def send_notification(
        self, message: str, alert_type: AlertType = AlertType.INFO
    ) -> bool:
        """
        Envoie une notification.

        Args:
            message: Message à envoyer
            alert_type: Type d'alerte

        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        success = True

        # Envoyer par email si configuré
        if self.email_notifier:
            success = self.email_notifier.send_alert(
                alert_type=alert_type,
                subject=f"Bot Trading - {alert_type.value}",
                message=message,
            )

        return success
