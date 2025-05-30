from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt

class RiskPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.labels = {
            "drawdown": QLabel("Drawdown: N/A"),
            "daily_pnl": QLabel("PnL Journalier: N/A"),
            "peak_balance": QLabel("Pic Capital: N/A"),
            "last_balance": QLabel("Capital Actuel: N/A"),
        }

        row = 0
        for text, label in self.labels.items():
            label.setStyleSheet("font-size: 14px;")
            self.grid_layout.addWidget(label, row, 0)
            row += 1

        self.layout.addLayout(self.grid_layout)
        self.layout.addStretch()
        self.setLayout(self.layout)

    def set_metrics(self, metrics):
        try:
            if metrics:
                self.labels["drawdown"].setText(f"Drawdown: {metrics.get('drawdown', 0.0):.2%}")
                self.labels["daily_pnl"].setText(f"PnL Journalier: {metrics.get('daily_pnl', 0.0):.2f}")
                self.labels["peak_balance"].setText(f"Pic Capital: {metrics.get('peak_balance', 0.0):.2f}")
                self.labels["last_balance"].setText(f"Capital Actuel: {metrics.get('last_balance', 0.0):.2f}")
        except Exception as e:
            print(f"Erreur dans RiskPanel.set_metrics: {e}")
 