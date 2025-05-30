from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class SignalPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.label = QLabel("Signal: HOLD")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 24px; color: gray;")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def set_signal(self, signal):
        text = f"Signal: {signal.upper()}"
        style = "font-size: 24px; "
        if signal == "buy":
            style += "color: green;"
        elif signal == "sell":
            style += "color: red;"
        else:
            style += "color: gray;"
        self.label.setText(text)
        self.label.setStyleSheet(style) 