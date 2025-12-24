"""
AccountInfoPanel : panneau d'informations de compte pour l'application de trading Bitcoin Scalper.
Affiche le solde, le PnL, le statut du bot et le dernier prix.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt
from pathlib import Path
import os

# Déterminer le chemin racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources" / "icons"

class AccountInfoPanel(QWidget):
    """
    Widget d'informations de compte pour l'application de trading Bitcoin Scalper.
    Affiche le solde, le PnL, le statut du bot (avec icône) et le dernier prix.
    """
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        print("DEBUG AccountInfoPanel __init__", id(self))
        self.setObjectName("AccountInfoPanel")
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Statut bot
        self.status_layout = QHBoxLayout()
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(24, 24)
        self.status_icon.setObjectName("status_icon")
        self.status_label = QLabel("Statut : Déconnecté")
        self.status_label.setObjectName("status_label")
        self.status_label.setProperty("role", "status")
        self.status_layout.addWidget(self.status_icon)
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addStretch()
        self.layout.addLayout(self.status_layout)

        # Solde
        self.balance_label = QLabel("Solde : N/A")
        self.balance_label.setObjectName("balance_label")
        self.balance_label.setProperty("role", "balance")
        self.layout.addWidget(self.balance_label)

        # Profit/Perte
        self.pnl_label = QLabel("Profit/Perte : N/A")
        self.pnl_label.setObjectName("pnl_label")
        self.pnl_label.setProperty("role", "pnl")
        self.layout.addWidget(self.pnl_label)

        # Dernier prix
        self.price_label = QLabel("Dernier prix : N/A")
        self.price_label.setObjectName("price_label")
        self.price_label.setProperty("role", "price")
        self.layout.addWidget(self.price_label)

        # Ligne de séparation
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(line)

        self.setLayout(self.layout)
        self.set_status("disconnected")

    def set_status(self, status: str) -> None:
        """Met à jour l'icône et le texte du statut du bot."""
        icon_path = RESOURCES_DIR / f"status_{status}.svg"
        if icon_path.exists():
            self.status_icon.setPixmap(QPixmap(str(icon_path)).scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.status_icon.clear()
        if status == "running":
            self.status_label.setText("Statut : En cours")
            self.status_label.setProperty("state", "running")
        elif status == "stopped":
            self.status_label.setText("Statut : Arrêté")
            self.status_label.setProperty("state", "stopped")
        elif status == "disconnected":
            self.status_label.setText("Statut : Déconnecté")
            self.status_label.setProperty("state", "disconnected")
        else:
            self.status_label.setText("")
            self.status_label.setProperty("state", "unknown")

    def set_balance(self, balance: float) -> None:
        """Met à jour l'affichage du solde du compte."""
        print("DEBUG set_balance:", balance)
        if balance is None:
            self.balance_label.setText("Solde : -")
        else:
            self.balance_label.setText(f"Solde : {balance:,.2f} $")

    def set_pnl(self, pnl: float) -> None:
        """Met à jour l'affichage du profit/perte du compte."""
        print("DEBUG set_pnl:", pnl)
        if pnl is None:
            self.pnl_label.setText("Profit/Perte : -")
        else:
            color = "#4caf50" if pnl >= 0 else "#f44336"
            self.pnl_label.setText(f"Profit/Perte : <span style='color:{color};'>{pnl:,.2f} $</span>")
            self.pnl_label.setProperty("positive", str(pnl >= 0).lower())

    def set_last_price(self, price: float) -> None:
        """Met à jour l'affichage du dernier prix connu."""
        if price is None:
            self.price_label.setText("Dernier prix : -")
        else:
            self.price_label.setText(f"Dernier prix : {price:,.2f} $")

    def set_disconnected(self) -> None:
        """Affiche le statut 'déconnecté'."""
        self.set_status("disconnected") 