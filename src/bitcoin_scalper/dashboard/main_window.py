"""
Main window for the Bitcoin Scalper Trading Dashboard.

Assembles all components into a professional trading interface.
"""

from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QStatusBar, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QIcon

from bitcoin_scalper.core.config import TradingConfig

from .styles import get_main_stylesheet, COLORS
from .widgets import (
    CandlestickChart, LogConsole, StatCard, MetaConfidencePanel
)
from .worker import TradingWorker


class MainWindow(QMainWindow):
    """
    Main trading dashboard window.
    
    Provides a "Mission Control" interface for monitoring and controlling
    the trading bot with real-time data visualization.
    """
    
    def __init__(self, config: TradingConfig, model_path: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.model_path = model_path
        self.worker: Optional[TradingWorker] = None
        
        self._init_ui()
        self._connect_signals()
        
        # Apply stylesheet
        self.setStyleSheet(get_main_stylesheet())
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Bitcoin Scalper - Trading Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel: Metrics
        left_panel = self._create_metrics_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Center panel: Chart and logs
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 4)
        
        # Right panel: Meta-labeling control
        right_panel = self._create_meta_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self._create_status_bar()
    
    def _create_metrics_panel(self) -> QWidget:
        """Create the left metrics panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 Metrics")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Metric cards
        self.balance_card = StatCard("Balance", "$10,000.00", 'neutral')
        layout.addWidget(self.balance_card)
        
        self.pnl_card = StatCard("P&L", "$0.00", 'neutral')
        layout.addWidget(self.pnl_card)
        
        self.trades_card = StatCard("Trades", "0", 'neutral')
        layout.addWidget(self.trades_card)
        
        self.winrate_card = StatCard("Win Rate", "0%", 'neutral')
        layout.addWidget(self.winrate_card)
        
        layout.addStretch()
        
        # Control buttons
        self.start_button = QPushButton("▶ Start")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumHeight(50)
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏸ Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_button)
        
        return panel
    
    def _create_center_panel(self) -> QWidget:
        """Create the center chart and log panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Chart
        chart_label = QLabel("📈 Price Chart")
        chart_label.setObjectName("titleLabel")
        layout.addWidget(chart_label)
        
        self.chart = CandlestickChart()
        layout.addWidget(self.chart, 3)
        
        # Logs
        log_label = QLabel("📝 Logs")
        log_label.setObjectName("titleLabel")
        layout.addWidget(log_label)
        
        self.log_console = LogConsole()
        layout.addWidget(self.log_console, 1)
        
        return panel
    
    def _create_meta_panel(self) -> QWidget:
        """Create the right meta-labeling panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Meta confidence panel
        self.meta_panel = MetaConfidencePanel(
            initial_threshold=self.config.meta_threshold
        )
        layout.addWidget(self.meta_panel)
        
        # Configuration info
        config_label = QLabel("⚙️ Configuration")
        config_label.setObjectName("titleLabel")
        config_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(config_label)
        
        config_text = QLabel(
            f"Symbol: {self.config.symbol}\n"
            f"Timeframe: {self.config.timeframe}\n"
            f"Mode: {self.config.mode.upper()}\n"
            f"Model: {Path(self.model_path).name if self.model_path else 'None'}"
        )
        config_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        config_text.setWordWrap(True)
        layout.addWidget(config_text)
        
        layout.addStretch()
        
        return panel
    
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.status_bar.showMessage("Dashboard initialized")
    
    def _connect_signals(self):
        """Connect all signal handlers."""
        # Signals will be connected when worker is created
        pass
    
    @pyqtSlot()
    def _on_start_clicked(self):
        """Handle start button click."""
        try:
            self.log_console.append_log("🚀 Starting trading engine...")
            
            # Create and start worker
            self.worker = TradingWorker(self.config, self.model_path)
            
            # Connect worker signals
            self.worker.log_message.connect(self.log_console.append_log)
            self.worker.price_update.connect(self.chart.update_candle)
            self.worker.signal_generated.connect(self.meta_panel.update_signal)
            self.worker.trade_executed.connect(self.chart.add_trade_marker)
            self.worker.metric_update.connect(self._on_metric_update)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.status_changed.connect(self._on_status_changed)
            
            # Connect meta threshold changes
            self.meta_panel.threshold_slider.valueChanged.connect(
                lambda v: self.worker.update_meta_threshold(v / 100.0)
            )
            
            # Start worker thread
            self.worker.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_bar.showMessage("Trading engine started")
            
        except Exception as e:
            self.log_console.append_log(f"❌ Failed to start: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start trading engine:\n{str(e)}")
    
    @pyqtSlot()
    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self.worker:
            self.log_console.append_log("🛑 Stopping trading engine...")
            self.worker.stop()
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_bar.showMessage("Trading engine stopped")
    
    @pyqtSlot(str, object)
    def _on_metric_update(self, metric_name: str, value):
        """Handle metric updates from worker."""
        try:
            if metric_name == 'balance':
                self.balance_card.update_value(f"${value:,.2f}")
            
            elif metric_name == 'pnl':
                self.pnl_card.update_value(f"${value:,.2f}")
                # Update card color
                if value > 0:
                    self.pnl_card.set_value_type('profit')
                elif value < 0:
                    self.pnl_card.set_value_type('loss')
                else:
                    self.pnl_card.set_value_type('neutral')
            
            elif metric_name == 'trades':
                self.trades_card.update_value(str(int(value)))
            
            elif metric_name == 'winrate':
                self.winrate_card.update_value(f"{value:.1f}%")
                # Update card color
                if value >= 60:
                    self.winrate_card.set_value_type('profit')
                elif value < 40:
                    self.winrate_card.set_value_type('loss')
                else:
                    self.winrate_card.set_value_type('neutral')
        
        except Exception as e:
            self.log_console.append_log(f"⚠️  Metric update error: {str(e)}")
    
    @pyqtSlot(str)
    def _on_error(self, error_message: str):
        """Handle error messages from worker."""
        self.log_console.append_log(f"❌ ERROR: {error_message}")
        self.status_bar.showMessage(f"Error: {error_message}", 5000)
    
    @pyqtSlot(str)
    def _on_status_changed(self, status: str):
        """Handle status changes from worker."""
        self.status_label.setText(f"Status: {status}")
        self.status_bar.showMessage(f"Engine status: {status}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Trading engine is still running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait(5000)  # Wait up to 5 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
