STYLE_SHEET = """
/* Thème sombre moderne avec meilleure lisibilité */
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    color: #cdd6f4;
    background-color: #1e1e2e;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
}

/* Groupes */
QGroupBox {
    border: 2px solid #313244;
    border-radius: 12px;
    margin-top: 1.5em;
    padding: 1.5em;
    background-color: #181825;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 8px;
    color: #89b4fa;
    font-weight: bold;
    font-size: 12pt;
}

/* Boutons */
QPushButton {
    background-color: #89b4fa;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    color: #1e1e2e;
    font-weight: bold;
    min-width: 120px;
    font-size: 11pt;
}

QPushButton:hover {
    background-color: #74c7ec;
}

QPushButton:pressed {
    background-color: #89b4fa;
}

QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#startButton {
    background-color: #a6e3a1;
    color: #1e1e2e;
}

QPushButton#startButton:hover {
    background-color: #94e2d5;
}

QPushButton#stopButton {
    background-color: #f38ba8;
    color: #1e1e2e;
}

QPushButton#stopButton:hover {
    background-color: #f5c2e7;
}

QPushButton#pauseButton {
    background-color: #f9e2af;
    color: #1e1e2e;
}

QPushButton#pauseButton:hover {
    background-color: #f5c2e7;
}

/* Tableaux */
QTableWidget {
    border: 2px solid #313244;
    border-radius: 8px;
    gridline-color: #313244;
    background-color: #181825;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #313244;
}

QTableWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QHeaderView::section {
    background-color: #313244;
    padding: 10px;
    border: none;
    font-weight: bold;
    color: #cdd6f4;
}

/* Champs de texte et zones d'édition */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    padding: 10px;
    color: #cdd6f4;
    min-width: 140px;
    font-size: 11pt;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: url(resources/down-arrow.png);
    width: 12px;
    height: 12px;
}

/* Onglets */
QTabWidget::pane {
    border: 2px solid #313244;
    border-radius: 8px;
    background-color: #181825;
}

QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 12px 24px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
    font-size: 11pt;
}

QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

/* Labels */
QLabel {
    color: #cdd6f4;
    font-size: 11pt;
}

QLabel#statusLabel {
    font-weight: bold;
    font-size: 12pt;
    color: #89b4fa;
}

QLabel#priceLabel, QLabel#pnlLabel {
    font-weight: bold;
    font-size: 14pt;
    color: #a6e3a1;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #313244;
    width: 12px;
    border-radius: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #89b4fa;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #74c7ec;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    border: none;
    background-color: #313244;
    height: 12px;
    border-radius: 6px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #89b4fa;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #74c7ec;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* Tooltips */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    font-size: 10pt;
}
""" 