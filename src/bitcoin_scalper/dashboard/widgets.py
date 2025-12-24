"""
Custom widgets for the trading dashboard.

Includes:
- CandlestickChart: Real-time candlestick chart with pyqtgraph
- LogConsole: Scrolling log display
- StatCard: Metric display cards
- MetaConfidencePanel: Meta-labeling visualization
"""

from typing import Optional, List, Tuple
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QFrame, QSlider, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np

from .styles import COLORS, get_stat_card_style, get_signal_indicator_style


class CandlestickChart(QWidget):
    """
    Real-time candlestick chart using pyqtgraph.
    
    Displays OHLCV data with buy/sell markers.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.timestamps: List[int] = []
        self.opens: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []
        self.buy_signals: List[Tuple[int, float]] = []  # (timestamp, price)
        self.sell_signals: List[Tuple[int, float]] = []  # (timestamp, price)
        
        self.max_points = 200  # Keep last 200 candles
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget(background=COLORS['bg_chart'])
        self.plot_widget.setLabel('left', 'Price', units='$', color=COLORS['text_primary'])
        self.plot_widget.setLabel('bottom', 'Time', color=COLORS['text_primary'])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Style the axes
        axis_pen = pg.mkPen(color=COLORS['text_secondary'], width=1)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        self.plot_widget.getAxis('left').setTextPen(COLORS['text_primary'])
        self.plot_widget.getAxis('bottom').setTextPen(COLORS['text_primary'])
        
        layout.addWidget(self.plot_widget)
        
        # Initialize plot items
        self.candle_items = []
        self.buy_scatter = pg.ScatterPlotItem(
            size=15, brush=pg.mkBrush(COLORS['accent_green']),
            symbol='t1', name='Buy'
        )
        self.sell_scatter = pg.ScatterPlotItem(
            size=15, brush=pg.mkBrush(COLORS['accent_red']),
            symbol='t', name='Sell'
        )
        
        self.plot_widget.addItem(self.buy_scatter)
        self.plot_widget.addItem(self.sell_scatter)
        
        # Add legend
        self.plot_widget.addLegend(offset=(10, 10))
    
    @pyqtSlot(float, float, float, float, float, int)
    def update_candle(self, open_: float, high: float, low: float, 
                      close: float, volume: float, timestamp: int):
        """
        Add or update a candlestick.
        
        Args:
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            timestamp: Unix timestamp
        """
        # Add new candle data
        self.timestamps.append(timestamp)
        self.opens.append(open_)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)
        
        # Keep only max_points
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            self.opens = self.opens[-self.max_points:]
            self.highs = self.highs[-self.max_points:]
            self.lows = self.lows[-self.max_points:]
            self.closes = self.closes[-self.max_points:]
            self.volumes = self.volumes[-self.max_points:]
        
        self._redraw_candles()
    
    def _redraw_candles(self):
        """Redraw all candlesticks."""
        # Clear old candles
        for item in self.candle_items:
            self.plot_widget.removeItem(item)
        self.candle_items.clear()
        
        # Draw new candles
        for i in range(len(self.timestamps)):
            open_ = self.opens[i]
            high = self.highs[i]
            low = self.lows[i]
            close = self.closes[i]
            x = i
            
            # Determine color
            color = COLORS['accent_green'] if close >= open_ else COLORS['accent_red']
            
            # Draw wick (high-low line)
            wick = pg.PlotDataItem(
                [x, x], [low, high],
                pen=pg.mkPen(color, width=1)
            )
            self.plot_widget.addItem(wick)
            self.candle_items.append(wick)
            
            # Draw body (open-close rectangle)
            body_height = abs(close - open_)
            body_y = min(open_, close)
            
            # Use a narrow rectangle for the body
            body_width = 0.6
            body = pg.QtWidgets.QGraphicsRectItem(
                x - body_width/2, body_y, body_width, body_height
            )
            body.setBrush(pg.mkBrush(color))
            body.setPen(pg.mkPen(color, width=1))
            
            self.plot_widget.addItem(body)
            self.candle_items.append(body)
        
        # Update buy/sell markers
        self._redraw_signals()
    
    def _redraw_signals(self):
        """Redraw buy/sell signal markers."""
        if self.buy_signals:
            buy_x = [self._timestamp_to_index(ts) for ts, _ in self.buy_signals]
            buy_y = [price for _, price in self.buy_signals]
            self.buy_scatter.setData(buy_x, buy_y)
        
        if self.sell_signals:
            sell_x = [self._timestamp_to_index(ts) for ts, _ in self.sell_signals]
            sell_y = [price for _, price in self.sell_signals]
            self.sell_scatter.setData(sell_x, sell_y)
    
    def _timestamp_to_index(self, timestamp: int) -> int:
        """Convert timestamp to chart index."""
        try:
            return self.timestamps.index(timestamp)
        except ValueError:
            # If exact timestamp not found, find closest
            if not self.timestamps:
                return 0
            closest_idx = min(range(len(self.timestamps)),
                            key=lambda i: abs(self.timestamps[i] - timestamp))
            return closest_idx
    
    @pyqtSlot(str, float, float)
    def add_trade_marker(self, side: str, price: float, volume: float):
        """
        Add a buy/sell marker to the chart.
        
        Args:
            side: 'buy' or 'sell'
            price: Execution price
            volume: Trade volume
        """
        if not self.timestamps:
            return
        
        # Use latest timestamp
        timestamp = self.timestamps[-1]
        
        if side == 'buy':
            self.buy_signals.append((timestamp, price))
        elif side == 'sell':
            self.sell_signals.append((timestamp, price))
        
        # Keep only signals within visible window
        if len(self.buy_signals) > self.max_points:
            self.buy_signals = self.buy_signals[-self.max_points:]
        if len(self.sell_signals) > self.max_points:
            self.sell_signals = self.sell_signals[-self.max_points:]
        
        self._redraw_signals()


class LogConsole(QPlainTextEdit):
    """
    Scrolling log console widget.
    
    Displays timestamped log messages with auto-scroll.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)  # Keep last 1000 lines
        
        # Set monospace font
        font = QFont("SF Mono", 11)
        if not font.exactMatch():
            font = QFont("Monaco", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        self.setFont(font)
    
    @pyqtSlot(str)
    def append_log(self, message: str):
        """
        Append a log message with timestamp.
        
        Args:
            message: Log message to append
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        # Append and auto-scroll
        self.appendPlainText(formatted)
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )


class StatCard(QFrame):
    """
    Metric display card widget.
    
    Shows a label and value with optional color coding.
    """
    
    def __init__(self, title: str, initial_value: str = "0",
                 value_type: str = 'neutral', parent=None):
        super().__init__(parent)
        
        self.value_type = value_type
        self._init_ui(title, initial_value)
    
    def _init_ui(self, title: str, initial_value: str):
        """Initialize the card UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Value label
        self.value_label = QLabel(initial_value)
        self.value_label.setObjectName("valueLabel")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        
        # Apply style
        self.setStyleSheet(get_stat_card_style(self.value_type))
    
    @pyqtSlot(str)
    def update_value(self, value: str):
        """Update the displayed value."""
        self.value_label.setText(value)
    
    def set_value_type(self, value_type: str):
        """Update the value type and re-apply style."""
        self.value_type = value_type
        self.setStyleSheet(get_stat_card_style(value_type))


class MetaConfidencePanel(QFrame):
    """
    Meta-labeling confidence visualization panel.
    
    Shows current confidence, threshold, and signal status.
    """
    
    def __init__(self, initial_threshold: float = 0.6, parent=None):
        super().__init__(parent)
        
        self.current_confidence = 0.0
        self.current_signal = 'hold'
        self.threshold = initial_threshold
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Meta-Labeling Brain")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Confidence progress bar
        conf_label = QLabel("Confidence:")
        conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(conf_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%p%")
        self.confidence_bar.setMinimumHeight(30)
        layout.addWidget(self.confidence_bar)
        
        # Current confidence value
        self.conf_value_label = QLabel("0.00")
        self.conf_value_label.setObjectName("valueLabel")
        self.conf_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_value_label)
        
        # Signal status indicator
        self.signal_label = QLabel("HOLD")
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signal_label.setStyleSheet(get_signal_indicator_style('hold'))
        layout.addWidget(self.signal_label)
        
        # Threshold control
        threshold_label = QLabel("Meta Threshold:")
        layout.addWidget(threshold_label)
        
        threshold_layout = QHBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self._on_slider_changed)
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(self.threshold)
        self.threshold_spinbox.valueChanged.connect(self._on_spinbox_changed)
        
        threshold_layout.addWidget(self.threshold_slider, 3)
        threshold_layout.addWidget(self.threshold_spinbox, 1)
        
        layout.addLayout(threshold_layout)
        
        layout.addStretch()
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        threshold = value / 100.0
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(threshold)
        self.threshold_spinbox.blockSignals(False)
        self.threshold = threshold
        self._update_signal_status()
    
    def _on_spinbox_changed(self, value: float):
        """Handle spinbox value change."""
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(value * 100))
        self.threshold_slider.blockSignals(False)
        self.threshold = value
        self._update_signal_status()
    
    @pyqtSlot(str, float)
    def update_signal(self, signal: str, confidence: float):
        """
        Update the displayed signal and confidence.
        
        Args:
            signal: Signal type ('buy', 'sell', 'hold', etc.)
            confidence: Confidence value (0.0 to 1.0)
        """
        self.current_signal = signal
        self.current_confidence = confidence if confidence is not None else 0.0
        
        # Update confidence bar
        conf_pct = int(self.current_confidence * 100)
        self.confidence_bar.setValue(conf_pct)
        self.conf_value_label.setText(f"{self.current_confidence:.2f}")
        
        self._update_signal_status()
    
    def _update_signal_status(self):
        """Update signal status indicator based on confidence and threshold."""
        # Determine if signal is filtered or passed
        if self.current_confidence >= self.threshold and self.current_signal in ['buy', 'sell']:
            status = self.current_signal  # 'buy' or 'sell'
            text = self.current_signal.upper()
        else:
            status = 'filtered' if self.current_signal in ['buy', 'sell'] else 'hold'
            text = "FILTERED" if status == 'filtered' else "HOLD"
        
        self.signal_label.setText(text)
        self.signal_label.setStyleSheet(get_signal_indicator_style(status))
    
    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.threshold


# Aliases for naming consistency with problem statement
ControlPanel = MetaConfidencePanel  # The control panel with meta_threshold slider
ChartWidget = CandlestickChart      # The chart widget
