from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class DashboardSimple(QWidget):
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DashboardSimple")
        self.layout = QVBoxLayout()
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 16, 16, 16)

        # Bandeau alerte critique
        self.alert_banner = QLabel()
        self.alert_banner.setVisible(False)
        self.alert_banner.setStyleSheet("background:#f44336;color:white;font-weight:bold;padding:8px;border-radius:6px;")
        self.layout.addWidget(self.alert_banner)

        # Statut + STOP
        top_layout = QHBoxLayout()
        self.status_label = QLabel("Statut : OFF")
        self.status_label.setStyleSheet("font-size:18px;font-weight:bold;color:#f44336;")
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("background:#f44336;color:white;font-weight:bold;padding:8px 24px;border-radius:6px;font-size:16px;")
        self.stop_btn.clicked.connect(self._on_stop)
        top_layout.addWidget(self.stop_btn)
        self.layout.addLayout(top_layout)

        # PnL et capital
        pnl_layout = QHBoxLayout()
        self.pnl_label = QLabel("PnL : 0.00 € (0.00%)")
        self.pnl_label.setStyleSheet("font-size:16px;")
        pnl_layout.addWidget(self.pnl_label)
        self.capital_label = QLabel("Capital : 0.00 €")
        self.capital_label.setStyleSheet("font-size:16px;")
        pnl_layout.addWidget(self.capital_label)
        self.positions_label = QLabel("Positions : 0")
        self.positions_label.setStyleSheet("font-size:16px;")
        pnl_layout.addWidget(self.positions_label)
        self.layout.addLayout(pnl_layout)

        # Graphique capital
        self.graph = pg.PlotWidget(title="Évolution du capital")
        self.graph.setBackground('dimgrey')
        self.layout.addWidget(self.graph, stretch=1)

        self.setLayout(self.layout)

    def set_status(self, running: bool):
        if running:
            self.status_label.setText("Statut : ON")
            self.status_label.setStyleSheet("font-size:18px;font-weight:bold;color:#4caf50;")
        else:
            self.status_label.setText("Statut : OFF")
            self.status_label.setStyleSheet("font-size:18px;font-weight:bold;color:#f44336;")

    def set_pnl(self, pnl_eur: float, pnl_pct: float):
        color = "#4caf50" if pnl_eur >= 0 else "#f44336"
        self.pnl_label.setText(f"PnL : <span style='color:{color};'>{pnl_eur:,.2f} € ({pnl_pct:.2f}%)</span>")

    def set_capital(self, capital: float):
        self.capital_label.setText(f"Capital : {capital:,.2f} €")

    def set_positions(self, n: int):
        self.positions_label.setText(f"Positions : {n}")

    def update_graph(self, capital_history):
        self.graph.clear()
        if capital_history:
            self.graph.plot(list(range(len(capital_history))), capital_history, pen=pg.mkPen('#4caf50', width=2))
        else:
            self.graph.addItem(pg.TextItem("Aucune donnée de capital", color='w', anchor=(0.5,0.5)))

    def show_alert(self, message: str):
        self.alert_banner.setText(message)
        self.alert_banner.setVisible(True)

    def hide_alert(self):
        self.alert_banner.setVisible(False)

    def _on_stop(self):
        self.stop_requested.emit() 