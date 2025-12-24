"""
Dark Theme Styles for Bitcoin Scalper Dashboard.

Provides a modern "Mission Control" dark cyberpunk aesthetic
optimized for macOS with professional trading interface colors.
"""

# Color Palette
COLORS = {
    # Background colors
    'bg_primary': '#1e1e1e',      # Main background
    'bg_secondary': '#2d2d2d',    # Secondary panels
    'bg_tertiary': '#3d3d3d',     # Hover states
    'bg_chart': '#0a0a0a',        # Chart background
    
    # Text colors
    'text_primary': '#e0e0e0',    # Main text
    'text_secondary': '#a0a0a0',  # Secondary text
    'text_disabled': '#606060',   # Disabled text
    
    # Accent colors
    'accent_green': '#00ff00',    # Buy signals, profit
    'accent_red': '#ff0044',      # Sell signals, loss
    'accent_yellow': '#ffaa00',   # Warnings
    'accent_blue': '#00aaff',     # Info
    'accent_purple': '#aa00ff',   # Meta-labeling
    
    # Status colors
    'profit': '#00ff00',
    'loss': '#ff0044',
    'neutral': '#808080',
    'hold': '#888888',
    'filtered': '#555555',
    
    # Border colors
    'border': '#404040',
    'border_active': '#606060',
}


def get_main_stylesheet() -> str:
    """
    Get the main QSS stylesheet for the application.
    
    Returns:
        Complete QSS stylesheet as string
    """
    return f"""
    /* ===== MAIN WINDOW ===== */
    QMainWindow {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}
    
    /* ===== WIDGETS ===== */
    QWidget {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
        font-family: "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
    }}
    
    /* ===== PANELS ===== */
    QFrame {{
        background-color: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px;
    }}
    
    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['border_active']};
        border-color: {COLORS['accent_blue']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['bg_secondary']};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_disabled']};
        border-color: {COLORS['border']};
    }}
    
    /* Start Button */
    QPushButton#startButton {{
        background-color: #006600;
        border-color: {COLORS['accent_green']};
        color: white;
    }}
    
    QPushButton#startButton:hover {{
        background-color: #008800;
    }}
    
    /* Stop Button */
    QPushButton#stopButton {{
        background-color: #660000;
        border-color: {COLORS['accent_red']};
        color: white;
    }}
    
    QPushButton#stopButton:hover {{
        background-color: #880000;
    }}
    
    /* ===== LABELS ===== */
    QLabel {{
        background: transparent;
        color: {COLORS['text_primary']};
        border: none;
    }}
    
    QLabel#titleLabel {{
        font-size: 18px;
        font-weight: bold;
        color: {COLORS['accent_blue']};
    }}
    
    QLabel#metricLabel {{
        font-size: 14px;
        color: {COLORS['text_secondary']};
    }}
    
    QLabel#valueLabel {{
        font-size: 24px;
        font-weight: bold;
        color: {COLORS['text_primary']};
    }}
    
    /* ===== TEXT EDIT ===== */
    QTextEdit {{
        background-color: {COLORS['bg_chart']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px;
        font-family: "SF Mono", "Monaco", "Courier New", monospace;
        font-size: 11px;
    }}
    
    /* ===== SLIDERS ===== */
    QSlider::groove:horizontal {{
        background: {COLORS['bg_tertiary']};
        height: 8px;
        border-radius: 4px;
    }}
    
    QSlider::handle:horizontal {{
        background: {COLORS['accent_purple']};
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {COLORS['accent_blue']};
    }}
    
    /* ===== SPINBOX ===== */
    QDoubleSpinBox, QSpinBox {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QDoubleSpinBox:focus, QSpinBox:focus {{
        border-color: {COLORS['accent_blue']};
    }}
    
    /* ===== PROGRESS BAR ===== */
    QProgressBar {{
        background-color: {COLORS['bg_tertiary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        text-align: center;
        color: {COLORS['text_primary']};
        font-weight: bold;
    }}
    
    QProgressBar::chunk {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {COLORS['accent_purple']},
            stop: 1 {COLORS['accent_blue']}
        );
        border-radius: 6px;
    }}
    
    /* ===== COMBO BOX ===== */
    QComboBox {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QComboBox:hover {{
        border-color: {COLORS['accent_blue']};
    }}
    
    QComboBox::drop-down {{
        border: none;
    }}
    
    /* ===== SCROLL BAR ===== */
    QScrollBar:vertical {{
        background: {COLORS['bg_secondary']};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {COLORS['bg_tertiary']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {COLORS['border_active']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    /* ===== STATUS BAR ===== */
    QStatusBar {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_secondary']};
        border-top: 1px solid {COLORS['border']};
    }}
    
    /* ===== MENU BAR ===== */
    QMenuBar {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    QMenuBar::item {{
        background: transparent;
        padding: 6px 12px;
    }}
    
    QMenuBar::item:selected {{
        background-color: {COLORS['bg_tertiary']};
        border-radius: 4px;
    }}
    
    QMenu {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
    }}
    
    QMenu::item {{
        padding: 6px 20px;
    }}
    
    QMenu::item:selected {{
        background-color: {COLORS['bg_tertiary']};
    }}
    """


def get_stat_card_style(value_type: str = 'neutral') -> str:
    """
    Get stylesheet for stat cards with color coding.
    
    Args:
        value_type: Type of value ('profit', 'loss', 'neutral')
    
    Returns:
        QSS stylesheet for the card
    """
    color = COLORS.get(value_type, COLORS['neutral'])
    
    return f"""
    QFrame {{
        background-color: {COLORS['bg_secondary']};
        border: 2px solid {color};
        border-radius: 10px;
        padding: 15px;
    }}
    
    QLabel#valueLabel {{
        color: {color};
        font-size: 28px;
        font-weight: bold;
    }}
    """


def get_signal_indicator_style(signal_type: str = 'hold') -> str:
    """
    Get stylesheet for signal indicators.
    
    Args:
        signal_type: Type of signal ('buy', 'sell', 'hold', 'filtered')
    
    Returns:
        QSS stylesheet for the indicator
    """
    colors_map = {
        'buy': COLORS['accent_green'],
        'sell': COLORS['accent_red'],
        'hold': COLORS['hold'],
        'filtered': COLORS['filtered'],
    }
    
    color = colors_map.get(signal_type, COLORS['neutral'])
    
    return f"""
    QLabel {{
        background-color: {color};
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px 20px;
    }}
    """
