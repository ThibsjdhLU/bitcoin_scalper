"""
SignalPanel : panneau d'affichage du signal de trading (buy/sell/hold) pour le bot Bitcoin Scalper.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class SignalPanel(QWidget):
    """
    Widget d'affichage du signal de trading (buy/sell/hold) pour le bot Bitcoin Scalper.
    """
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.label = QLabel("Signal: HOLD")
        self.label.setObjectName("signal_label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setProperty("role", "signal")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def set_signal(self, signal: str) -> None:
        """Met à jour l'affichage du signal de trading (buy/sell/hold)."""
        text = f"Signal: {signal.upper()}"
        style = "font-size: 24px; "
        if signal == "buy":
            self.label.setProperty("state", "buy")
        elif signal == "sell":
            self.label.setProperty("state", "sell")
        else:
            self.label.setProperty("state", "hold")
        self.label.setText(text) 