from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt
import os

class AccountInfoPanel(QWidget):
    """
    Widget d'informations de compte pour l'application de trading.
    Affiche :
        - Solde
        - Profit/Perte
        - Statut du bot (avec icône)
        - Dernier prix
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AccountInfoPanel")
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Statut bot
        self.status_layout = QHBoxLayout()
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(24, 24)
        self.status_label = QLabel("Statut : Déconnecté")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.status_layout.addWidget(self.status_icon)
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addStretch()
        self.layout.addLayout(self.status_layout)

        # Solde
        self.balance_label = QLabel("Solde : N/A")
        self.balance_label.setStyleSheet("font-size: 15px;")
        self.layout.addWidget(self.balance_label)

        # Profit/Perte
        self.pnl_label = QLabel("Profit/Perte : N/A")
        self.pnl_label.setStyleSheet("font-size: 15px;")
        self.layout.addWidget(self.pnl_label)

        # Dernier prix
        self.price_label = QLabel("Dernier prix : N/A")
        self.price_label.setStyleSheet("font-size: 15px;")
        self.layout.addWidget(self.price_label)

        # Ligne de séparation
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(line)

        self.setLayout(self.layout)
        self.set_status("disconnected")

    def set_status(self, status: str):
        """Met à jour l'icône et le texte du statut du bot."""
        icon_path = os.path.join("resources", f"status_{status}.svg")
        if os.path.exists(icon_path):
            self.status_icon.setPixmap(QPixmap(icon_path).scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.status_icon.clear()
        if status == "running":
            self.status_label.setText("Statut : En cours")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
        elif status == "stopped":
            self.status_label.setText("Statut : Arrêté")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
        else:
            self.status_label.setText("Statut : Déconnecté")
            self.status_label.setStyleSheet("color: #b0b0b0; font-weight: bold; font-size: 16px;")

    def set_balance(self, balance: float):
        self.balance_label.setText(f"Solde : {balance:,.2f} $")

    def set_pnl(self, pnl: float):
        color = "#4caf50" if pnl >= 0 else "#f44336"
        self.pnl_label.setText(f"Profit/Perte : <span style='color:{color};'>{pnl:,.2f} $</span>")

    def set_last_price(self, price: float):
        self.price_label.setText(f"Dernier prix : {price:,.2f} $")

    def set_disconnected(self):
        self.set_status("disconnected") 