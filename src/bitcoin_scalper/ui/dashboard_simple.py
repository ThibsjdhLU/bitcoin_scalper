"""
DashboardSimple : panneau synthétique pour l'affichage du statut, PnL, capital et alertes du bot de trading Bitcoin Scalper.
Permet un accès rapide aux informations clés et à l'arrêt d'urgence du bot.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class DashboardSimple(QWidget):
    """
    Widget de dashboard synthétique pour le bot de trading Bitcoin Scalper.
    Affiche le statut, le PnL, le capital, le nombre de positions et un graphique d'évolution du capital.
    Fournit un bouton d'arrêt d'urgence et un bandeau d'alerte critique.
    """
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setObjectName("DashboardSimple")
        self.layout = QVBoxLayout()
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 16, 16, 16)

        # Bandeau alerte critique
        self.alert_banner = QLabel()
        self.alert_banner.setObjectName("alert_banner")
        self.alert_banner.setProperty("role", "alert")
        self.alert_banner.setVisible(False)
        self.layout.addWidget(self.alert_banner)

        # Statut + STOP
        top_layout = QHBoxLayout()
        self.status_label = QLabel("Statut : OFF")
        self.status_label.setObjectName("status_label")
        self.status_label.setProperty("role", "status")
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setProperty("role", "stop")
        self.stop_btn.clicked.connect(self._on_stop)
        top_layout.addWidget(self.stop_btn)
        self.layout.addLayout(top_layout)

        # PnL et capital
        pnl_layout = QHBoxLayout()
        self.pnl_label = QLabel("PnL : 0.00 € (0.00%)")
        self.pnl_label.setObjectName("pnl_label")
        self.pnl_label.setProperty("role", "pnl")
        pnl_layout.addWidget(self.pnl_label)
        self.capital_label = QLabel("Capital : 0.00 €")
        self.capital_label.setObjectName("capital_label")
        self.capital_label.setProperty("role", "capital")
        pnl_layout.addWidget(self.capital_label)
        self.positions_label = QLabel("Positions : 0")
        self.positions_label.setObjectName("positions_label")
        self.positions_label.setProperty("role", "positions")
        pnl_layout.addWidget(self.positions_label)
        self.layout.addLayout(pnl_layout)

        # Graphique capital
        self.graph = pg.PlotWidget(title="Évolution du capital")
        self.graph.setBackground('dimgrey')
        self.layout.addWidget(self.graph, stretch=1)

        self.setLayout(self.layout)

    def set_status(self, running: bool) -> None:
        """Met à jour l'affichage du statut (ON/OFF) du bot."""
        if running:
            self.status_label.setText("Statut : ON")
            self.status_label.setProperty("state", "on")
        else:
            self.status_label.setText("Statut : OFF")
            self.status_label.setProperty("state", "off")

    def set_pnl(self, pnl_eur: float, pnl_pct: float) -> None:
        """Met à jour l'affichage du PnL en euros et en pourcentage."""
        if pnl_eur is None or pnl_pct is None:
            self.pnl_label.setText("PnL : -")
        else:
            color = "#4caf50" if pnl_eur >= 0 else "#f44336"
            self.pnl_label.setText(f"PnL : <span style='color:{color};'>{pnl_eur:,.2f} € ({pnl_pct:.2f}%)</span>")
            self.pnl_label.setProperty("positive", str(pnl_eur >= 0).lower())

    def set_capital(self, capital: float) -> None:
        """Met à jour l'affichage du capital actuel."""
        if capital is None:
            self.capital_label.setText("Capital : -")
        else:
            self.capital_label.setText(f"Capital : {capital:,.2f} €")

    def set_positions(self, n: int) -> None:
        """Met à jour l'affichage du nombre de positions ouvertes."""
        if n is None:
            self.positions_label.setText("Positions : -")
        else:
            self.positions_label.setText(f"Positions : {n}")

    def update_graph(self, capital_history: list[float]) -> None:
        """Met à jour le graphique d'évolution du capital."""
        self.graph.clear()
        if capital_history and any(capital_history):
            self.graph.plot(list(range(len(capital_history))), capital_history, pen=pg.mkPen('#4caf50', width=2))
        else:
            self.graph.addItem(pg.TextItem("Aucune donnée de capital", color='w', anchor=(0.5,0.5)))

    def show_alert(self, message: str) -> None:
        """Affiche un message d'alerte critique dans le bandeau supérieur."""
        self.alert_banner.setText(message)
        self.alert_banner.setVisible(True)

    def hide_alert(self) -> None:
        """Masque le bandeau d'alerte."""
        self.alert_banner.setVisible(False)

    def _on_stop(self) -> None:
        """Émet le signal d'arrêt d'urgence du bot."""
        self.stop_requested.emit() 