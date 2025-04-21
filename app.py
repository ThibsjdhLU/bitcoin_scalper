import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QTableWidget, QTableWidgetItem, QTabWidget,
                            QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
                            QMessageBox, QFrame, QLineEdit, QTextEdit, QSplitter,
                            QHeaderView, QStyle, QLayout, QScrollArea)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QGuiApplication
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
import logging
import pyqtgraph as pg
from queue import Queue
import numpy as np

from api_connector import APIConnector
from strategy import ScalperStrategy
from position_manager import PositionManager
from risk_manager import RiskManager
from data_manager import DataManager
from mt5_connector import MT5Connector
from styles import STYLE_SHEET
from config_manager import ConfigManager

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class ScalperBotApp(QMainWindow):
    """
    Application principale du bot de scalping
    """
    
    # Signaux personnalisés
    error_signal = Signal(str)
    status_signal = Signal(str)
    data_update_signal = Signal(tuple)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Scalper Bot")
        self.setGeometry(100, 100, 1200, 800)
        
        # Configuration de la mise à l'échelle pour Windows
        if sys.platform == 'win32':
            # Utilisation des nouveaux attributs Qt pour la mise à l'échelle
            QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        
        # Style global amélioré pour Windows
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial;
                font-size: 10pt;
            }
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
                font-size: 11pt;
            }
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
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 10px;
                color: #cdd6f4;
                min-width: 140px;
                font-size: 11pt;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
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
            QTextEdit {
                background-color: #181825;
                border: 2px solid #313244;
                border-radius: 8px;
                color: #cdd6f4;
                font-family: 'Consolas', monospace;
                font-size: 11pt;
                padding: 10px;
            }
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
        """)
        
        # Initialisation des composants
        self.strategy = None
        self.strategy_thread = None
        self.mt5_connector = MT5Connector()
        self.data_manager = None
        self.logger = logging.getLogger(__name__)
        
        # File d'attente pour les logs
        self.log_queue = Queue()
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_log_queue)
        self.log_timer.start(100)  # Vérifie la file d'attente toutes les 100ms
        
        # Timer pour la mise à jour de l'interface
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Met à jour l'interface chaque seconde
        
        # Configuration par défaut
        self.config = {
            'symbol': 'BTCUSD',
            'timeframe': '1m',
            'min_volume': 1000000,
            'min_volatility': 0.001,
            'signal_strength_threshold': 2,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'atr_multiplier': 2,
            'trading_hours': {
                'start': '00:00',
                'end': '23:59',
                'timezone': 'UTC'
            }
        }
        
        # Initialisation de l'interface
        self.init_ui()
        
        # Chargement de la configuration sauvegardée
        self.load_config()
        
        # Vérification de la connexion MT5
        self.check_mt5_connection()
        
    def check_mt5_connection(self):
        """Vérifie la connexion MT5 et affiche un message d'erreur si nécessaire"""
        try:
            if not self.mt5_connector.initialized:
                if not self.mt5_connector.initialize():
                    self.handle_error("Échec de l'initialisation MT5")
                    return False
                    
            if not self.mt5_connector.connected:
                self.log_message("Connexion à MT5...")
                # Récupération des informations de connexion depuis les variables d'environnement
                login = os.getenv('MT5_LOGIN')
                self.log_message(f"MT5_LOGIN lu: {login}")
                if not login:
                    raise Exception("MT5_LOGIN manquant dans le fichier .env")
                try:
                    login_int = int(login)
                    self.log_message(f"MT5_LOGIN converti en entier: {login_int}")
                except (ValueError, TypeError) as e:
                    raise Exception(f"MT5_LOGIN invalide dans le fichier .env: {login}. Erreur: {str(e)}")
                
                password = os.getenv('MT5_PASSWORD')
                self.log_message(f"MT5_PASSWORD lu: {'*' * len(password) if password else 'None'}")
                if not password:
                    raise Exception("MT5_PASSWORD manquant dans le fichier .env")
                
                server = os.getenv('MT5_SERVER')
                self.log_message(f"MT5_SERVER lu: {server}")
                if not server:
                    raise Exception("MT5_SERVER manquant dans le fichier .env")
                
                if not self.mt5_connector.login(login=login_int, password=password, server=server):
                    raise Exception(f"Échec de la connexion MT5: {self.mt5_connector.get_last_error()}")
            
            return True
            
        except Exception as e:
            self.handle_error(f"Erreur lors de la vérification de la connexion MT5: {str(e)}")
            return False
        
    def init_ui(self):
        """
        Initialise l'interface utilisateur avec un design moderne
        """
        # Widget central avec taille minimale
        central_widget = QWidget()
        central_widget.setMinimumSize(1200, 800)
        self.setCentralWidget(central_widget)
        
        # Layout principal vertical avec taille minimale
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        
        # Barre d'état supérieure avec taille minimale
        status_bar = QWidget()
        status_bar.setMinimumHeight(80)
        status_bar.setStyleSheet("""
            QWidget {
                background-color: #181825;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(15, 10, 15, 10)
        status_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        
        # Statut du bot avec icône et style amélioré
        status_container = QWidget()
        status_container_layout = QHBoxLayout(status_container)
        status_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_icon = QLabel()
        self.status_icon.setPixmap(self.style().standardPixmap(QStyle.StandardPixmap.SP_DialogNoButton))
        self.status_icon.setFixedSize(24, 24)
        status_container_layout.addWidget(self.status_icon)
        
        self.status_label = QLabel("Statut: Arrêté")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet("color: #f38ba8;")
        status_container_layout.addWidget(self.status_label)
        
        status_layout.addWidget(status_container)
        
        # Prix actuel avec icône et style amélioré
        price_container = QWidget()
        price_layout = QHBoxLayout(price_container)
        price_layout.setContentsMargins(0, 0, 0, 0)
        
        price_icon = QLabel()
        price_icon.setPixmap(self.style().standardPixmap(QStyle.StandardPixmap.SP_ArrowRight))
        price_icon.setFixedSize(24, 24)
        price_layout.addWidget(price_icon)
        
        self.price_label = QLabel("Prix BTC/USD: 0.00")
        self.price_label.setObjectName("priceLabel")
        price_layout.addWidget(self.price_label)
        
        status_layout.addWidget(price_container)
        
        # Profit/Perte avec icône et style amélioré
        pnl_container = QWidget()
        pnl_layout = QHBoxLayout(pnl_container)
        pnl_layout.setContentsMargins(0, 0, 0, 0)
        
        pnl_icon = QLabel()
        pnl_icon.setPixmap(self.style().standardPixmap(QStyle.StandardPixmap.SP_ArrowUp))
        pnl_icon.setFixedSize(24, 24)
        pnl_layout.addWidget(pnl_icon)
        
        self.pnl_label = QLabel("P/L: 0.00")
        self.pnl_label.setObjectName("pnlLabel")
        pnl_layout.addWidget(self.pnl_label)
        
        status_layout.addWidget(pnl_container)
        
        status_layout.addStretch()
        
        # Dernière mise à jour avec style amélioré
        self.last_update_label = QLabel("Dernière mise à jour: Jamais")
        self.last_update_label.setStyleSheet("color: #6c7086;")
        status_layout.addWidget(self.last_update_label)
        
        main_layout.addWidget(status_bar)
        
        # Création du widget d'onglets
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
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
        """)
        
        # Onglet Configuration
        config_tab = QWidget()
        config_tab.setStyleSheet("background-color: #181825;")
        config_scroll = QScrollArea()
        config_scroll.setWidget(config_tab)
        config_scroll.setWidgetResizable(True)
        config_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #181825;
            }
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
        """)
        tab_widget.addTab(config_scroll, "Configuration")
        
        # Layout pour l'onglet Configuration
        config_layout = QVBoxLayout(config_tab)
        config_layout.setSpacing(20)
        config_layout.setContentsMargins(20, 20, 20, 20)
        
        # Ajout des widgets de configuration existants
        config_layout.addWidget(self.create_config_panel())
        
        # Onglet Graphiques
        charts_tab = QWidget()
        charts_tab.setStyleSheet("background-color: #181825;")
        charts_scroll = QScrollArea()
        charts_scroll.setWidget(charts_tab)
        charts_scroll.setWidgetResizable(True)
        charts_scroll.setStyleSheet(config_scroll.styleSheet())
        tab_widget.addTab(charts_scroll, "Graphiques")
        
        # Layout pour l'onglet Graphiques
        charts_layout = QVBoxLayout(charts_tab)
        charts_layout.setSpacing(20)
        charts_layout.setContentsMargins(20, 20, 20, 20)
        
        # Ajout des graphiques existants
        charts_layout.addWidget(self.create_charts_panel())
        
        # Onglet Logs
        logs_tab = QWidget()
        logs_tab.setStyleSheet("background-color: #181825;")
        logs_scroll = QScrollArea()
        logs_scroll.setWidget(logs_tab)
        logs_scroll.setWidgetResizable(True)
        logs_scroll.setStyleSheet(config_scroll.styleSheet())
        tab_widget.addTab(logs_scroll, "Logs")
        
        # Layout pour l'onglet Logs
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.setSpacing(20)
        logs_layout.setContentsMargins(20, 20, 20, 20)
        
        # Ajout des logs existants
        logs_layout.addWidget(self.create_logs_panel())
        
        main_layout.addWidget(tab_widget)
        
        # Définition de la taille minimale de la fenêtre
        self.setMinimumSize(1400, 1000)
        
        # Initialisation des listes pour l'historique
        self.price_history = []
        self.time_history = []
        
        # Connexion des signaux
        self.error_signal.connect(self.handle_error)
        self.status_signal.connect(self.handle_status)
        self.data_update_signal.connect(self.handle_data_update)

    def create_config_panel(self):
        """
        Crée le panneau de configuration
        """
        config_panel = QWidget()
        config_panel.setStyleSheet("""
            QWidget {
                background-color: #181825;
                border-radius: 12px;
            }
            QGroupBox {
                margin-top: 1.5em;
                padding: 1.5em;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                min-width: 200px;
                min-height: 30px;
            }
            QPushButton {
                min-width: 150px;
                min-height: 40px;
            }
        """)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setSpacing(20)
        config_layout.setContentsMargins(20, 20, 20, 20)
        
        # Groupe de configuration principal
        config_group = QGroupBox("Configuration du Bot")
        config_form = QFormLayout()
        config_form.setSpacing(15)
        config_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Symboles avec placeholder
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['BTC/USD', 'ETH/USD', 'BNB/USD'])
        self.symbol_combo.setToolTip("Sélectionnez la paire de trading")
        self.symbol_combo.setPlaceholderText("Sélectionnez une paire")
        config_form.addRow("Symbole:", self.symbol_combo)
        
        # Timeframe avec placeholder
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
        self.timeframe_combo.setToolTip("Intervalle de temps pour l'analyse technique")
        self.timeframe_combo.setPlaceholderText("Sélectionnez un timeframe")
        config_form.addRow("Timeframe:", self.timeframe_combo)
        
        config_group.setLayout(config_form)
        config_layout.addWidget(config_group)
        
        # Paramètres RSI
        rsi_group = QGroupBox("Paramètres RSI")
        rsi_layout = QFormLayout()
        rsi_layout.setSpacing(10)
        
        self.rsi_period_spin = QSpinBox()
        self.rsi_period_spin.setRange(2, 100)
        self.rsi_period_spin.setValue(14)
        self.rsi_period_spin.setToolTip("Nombre de périodes pour le calcul du RSI")
        rsi_layout.addRow("Période:", self.rsi_period_spin)
        
        self.rsi_overbought_spin = QSpinBox()
        self.rsi_overbought_spin.setRange(50, 100)
        self.rsi_overbought_spin.setValue(70)
        self.rsi_overbought_spin.setToolTip("Niveau de surachat")
        rsi_layout.addRow("Suracheté:", self.rsi_overbought_spin)
        
        self.rsi_oversold_spin = QSpinBox()
        self.rsi_oversold_spin.setRange(0, 50)
        self.rsi_oversold_spin.setValue(30)
        self.rsi_oversold_spin.setToolTip("Niveau de survente")
        rsi_layout.addRow("Survente:", self.rsi_oversold_spin)
        
        rsi_group.setLayout(rsi_layout)
        config_layout.addWidget(rsi_group)
        
        # Paramètres MACD
        macd_group = QGroupBox("Paramètres MACD")
        macd_layout = QFormLayout()
        macd_layout.setSpacing(10)
        
        self.macd_fast_spin = QSpinBox()
        self.macd_fast_spin.setRange(2, 100)
        self.macd_fast_spin.setValue(12)
        self.macd_fast_spin.setToolTip("Période rapide du MACD")
        macd_layout.addRow("Rapide:", self.macd_fast_spin)
        
        self.macd_slow_spin = QSpinBox()
        self.macd_slow_spin.setRange(2, 100)
        self.macd_slow_spin.setValue(26)
        self.macd_slow_spin.setToolTip("Période lente du MACD")
        macd_layout.addRow("Lent:", self.macd_slow_spin)
        
        self.macd_signal_spin = QSpinBox()
        self.macd_signal_spin.setRange(2, 100)
        self.macd_signal_spin.setValue(9)
        self.macd_signal_spin.setToolTip("Période du signal MACD")
        macd_layout.addRow("Signal:", self.macd_signal_spin)
        
        macd_group.setLayout(macd_layout)
        config_layout.addWidget(macd_group)
        
        # Paramètres Bollinger Bands
        bb_group = QGroupBox("Paramètres Bollinger Bands")
        bb_layout = QFormLayout()
        bb_layout.setSpacing(10)
        
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(2, 100)
        self.bb_period_spin.setValue(20)
        self.bb_period_spin.setToolTip("Période pour les bandes de Bollinger")
        bb_layout.addRow("Période:", self.bb_period_spin)
        
        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(0.1, 5.0)
        self.bb_std_spin.setValue(2.0)
        self.bb_std_spin.setSingleStep(0.1)
        self.bb_std_spin.setToolTip("Nombre d'écarts-types")
        bb_layout.addRow("Écart-type:", self.bb_std_spin)
        
        bb_group.setLayout(bb_layout)
        config_layout.addWidget(bb_group)
        
        # Paramètres ATR
        atr_group = QGroupBox("Paramètres ATR")
        atr_layout = QFormLayout()
        atr_layout.setSpacing(10)
        
        self.atr_period_spin = QSpinBox()
        self.atr_period_spin.setRange(2, 100)
        self.atr_period_spin.setValue(14)
        self.atr_period_spin.setToolTip("Période pour l'ATR")
        atr_layout.addRow("Période:", self.atr_period_spin)
        
        self.atr_multiplier_spin = QDoubleSpinBox()
        self.atr_multiplier_spin.setRange(0.1, 10.0)
        self.atr_multiplier_spin.setValue(2.0)
        self.atr_multiplier_spin.setSingleStep(0.1)
        self.atr_multiplier_spin.setToolTip("Multiplicateur ATR")
        atr_layout.addRow("Multiplicateur:", self.atr_multiplier_spin)
        
        atr_group.setLayout(atr_layout)
        config_layout.addWidget(atr_group)
        
        # Boutons de contrôle
        control_group = QGroupBox("Contrôle")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        
        self.start_button = QPushButton("Démarrer")
        self.start_button.setObjectName("startButton")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_bot)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.clicked.connect(self.pause_bot)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)
        
        control_group.setLayout(control_layout)
        config_layout.addWidget(control_group)
        
        return config_panel

    def create_charts_panel(self):
        """
        Crée le panneau des graphiques
        """
        charts_panel = QWidget()
        charts_panel.setStyleSheet("""
            QWidget {
                background-color: #181825;
                border-radius: 12px;
            }
            QGroupBox {
                margin-top: 1.5em;
                padding: 1.5em;
            }
        """)
        charts_layout = QVBoxLayout(charts_panel)
        charts_layout.setSpacing(20)
        charts_layout.setContentsMargins(20, 20, 20, 20)
        
        # Groupe des graphiques
        charts_group = QGroupBox("Graphiques")
        charts_inner_layout = QVBoxLayout()
        charts_inner_layout.setSpacing(15)
        charts_inner_layout.setContentsMargins(15, 15, 15, 15)
        
        # Création des graphiques
        self.price_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.price_canvas.setMinimumHeight(200)
        self.rsi_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.rsi_canvas.setMinimumHeight(150)
        self.macd_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.macd_canvas.setMinimumHeight(150)
        
        # Style des graphiques
        for canvas in [self.price_canvas, self.rsi_canvas, self.macd_canvas]:
            canvas.figure.patch.set_facecolor('#181825')
            canvas.axes.set_facecolor('#181825')
            canvas.axes.tick_params(colors='#cdd6f4')
            canvas.axes.spines['bottom'].set_color('#cdd6f4')
            canvas.axes.spines['top'].set_color('#cdd6f4')
            canvas.axes.spines['left'].set_color('#cdd6f4')
            canvas.axes.spines['right'].set_color('#cdd6f4')
        
        self.price_canvas.axes.set_title('Prix BTC/USD', color='#cdd6f4', pad=10)
        self.price_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
        self.price_canvas.axes.set_ylabel('Prix (USD)', color='#cdd6f4')
        
        self.rsi_canvas.axes.set_title('RSI', color='#cdd6f4', pad=10)
        self.rsi_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
        self.rsi_canvas.axes.set_ylabel('RSI', color='#cdd6f4')
        
        self.macd_canvas.axes.set_title('MACD', color='#cdd6f4', pad=10)
        self.macd_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
        self.macd_canvas.axes.set_ylabel('MACD', color='#cdd6f4')
        
        charts_inner_layout.addWidget(self.price_canvas)
        charts_inner_layout.addWidget(self.rsi_canvas)
        charts_inner_layout.addWidget(self.macd_canvas)
        
        charts_group.setLayout(charts_inner_layout)
        charts_layout.addWidget(charts_group)
        
        return charts_panel

    def create_logs_panel(self):
        """
        Crée le panneau des logs
        """
        logs_panel = QWidget()
        logs_panel.setStyleSheet("""
            QWidget {
                background-color: #181825;
                border-radius: 12px;
            }
            QGroupBox {
                margin-top: 1.5em;
                padding: 1.5em;
            }
            QTextEdit {
                min-height: 200px;
            }
        """)
        logs_layout = QVBoxLayout(logs_panel)
        logs_layout.setSpacing(20)
        logs_layout.setContentsMargins(20, 20, 20, 20)
        
        # Groupe des logs
        logs_group = QGroupBox("Logs")
        logs_inner_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 11pt;
            }
        """)
        logs_inner_layout.addWidget(self.log_text)
        
        logs_group.setLayout(logs_inner_layout)
        logs_layout.addWidget(logs_group)
        
        return logs_panel

    def validate_config(self):
        """
        Valide les paramètres de configuration
        
        Returns:
            tuple: (bool, str) - (validation réussie, message d'erreur)
        """
        try:
            # Validation des paramètres RSI
            if not (0 < self.config['rsi_period'] <= 100):
                return False, "La période RSI doit être comprise entre 1 et 100"
                
            if not (0 <= self.config['rsi_oversold'] < self.config['rsi_overbought'] <= 100):
                return False, "Les niveaux RSI doivent être : 0 <= survente < surachat <= 100"
            
            # Validation des paramètres MACD
            if not (self.config['macd_fast'] < self.config['macd_slow']):
                return False, "La période rapide MACD doit être inférieure à la période lente"
                
            if not (0 < self.config['macd_signal'] <= 100):
                return False, "La période du signal MACD doit être comprise entre 1 et 100"
            
            # Validation des paramètres Bollinger
            if not (0 < self.config['bb_period'] <= 100):
                return False, "La période des bandes de Bollinger doit être comprise entre 1 et 100"
                
            if not (0 < self.config['bb_std'] <= 10):
                return False, "L'écart-type des bandes de Bollinger doit être compris entre 0 et 10"
            
            # Validation des paramètres ATR
            if not (0 < self.config['atr_period'] <= 100):
                return False, "La période ATR doit être comprise entre 1 et 100"
                
            if not (0 < self.config['atr_multiplier'] <= 10):
                return False, "Le multiplicateur ATR doit être compris entre 0 et 10"
            
            return True, "Configuration valide"
            
        except Exception as e:
            return False, f"Erreur lors de la validation de la configuration: {str(e)}"

    def update_config(self):
        """
        Met à jour la configuration à partir des valeurs de l'interface
        """
        try:
            # Mise à jour de la configuration
            self.config.update({
                'symbol': self.symbol_combo.currentText(),
                'timeframe': self.timeframe_combo.currentText(),
                'rsi_period': self.rsi_period_spin.value(),
                'rsi_overbought': self.rsi_overbought_spin.value(),
                'rsi_oversold': self.rsi_oversold_spin.value(),
                'macd_fast': self.macd_fast_spin.value(),
                'macd_slow': self.macd_slow_spin.value(),
                'macd_signal': self.macd_signal_spin.value(),
                'bb_period': self.bb_period_spin.value(),
                'bb_std': self.bb_std_spin.value(),
                'atr_period': self.atr_period_spin.value(),
                'atr_multiplier': self.atr_multiplier_spin.value()
            })
            
            # Validation des paramètres
            is_valid, message = self.validate_config()
            if not is_valid:
                raise ValueError(message)
                
            # Sauvegarde de la configuration
            self.save_config()
            
            # Si le bot est en cours d'exécution, mettre à jour la stratégie
            if self.strategy and self.strategy.running:
                self.strategy.update_config(self.config)
                self.log_message("Configuration mise à jour en temps réel")
            
            self.log_message("Configuration mise à jour avec succès")
            
        except Exception as e:
            self.handle_error(f"Erreur lors de la mise à jour de la configuration: {str(e)}")

    def save_config(self):
        """
        Sauvegarde la configuration dans un fichier JSON
        """
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.log_message("Configuration sauvegardée")
        except Exception as e:
            self.handle_error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")

    def load_config(self):
        """
        Charge la configuration depuis le fichier JSON
        """
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                    
                # Mise à jour de l'interface
                self.symbol_combo.setCurrentText(self.config['symbol'])
                self.timeframe_combo.setCurrentText(self.config['timeframe'])
                self.rsi_period_spin.setValue(self.config['rsi_period'])
                self.rsi_overbought_spin.setValue(self.config['rsi_overbought'])
                self.rsi_oversold_spin.setValue(self.config['rsi_oversold'])
                
                self.log_message("Configuration chargée avec succès")
        except Exception as e:
            self.handle_error(f"Erreur lors du chargement de la configuration: {str(e)}")

    def log_message(self, message: str, level: str = "info"):
        """
        Ajoute un message à la file d'attente des logs
        
        Args:
            message (str): Message à ajouter
            level (str): Niveau de log ('info', 'warning', 'error')
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_message = f"{timestamp} - [{level.upper()}] - {message}"
            self.log_queue.put(formatted_message)
            
            # Mise à jour de la couleur du message selon le niveau
            color = {
                'info': '#cdd6f4',    # Blanc
                'warning': '#f9e2af',  # Jaune
                'error': '#f38ba8'     # Rouge
            }.get(level.lower(), '#cdd6f4')
            
            # Ajout du message au widget de logs avec la couleur appropriée
            self.log_text.append(f'<span style="color: {color}">{formatted_message}</span>')
            
            # Faire défiler automatiquement vers le bas
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
            
        except Exception as e:
            print(f"Erreur lors de l'ajout du message aux logs: {str(e)}")
        
    def process_log_queue(self):
        """
        Traite la file d'attente des logs de manière thread-safe
        """
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.log_text.append(message)
                # Faire défiler automatiquement vers le bas
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )
        except Exception as e:
            print(f"Erreur lors du traitement des logs: {str(e)}")

    def handle_error(self, error_message):
        """
        Gère les erreurs de manière plus robuste
        """
        self.log_message(f"ERREUR: {error_message}", level="ERROR")
        self.status_label.setText("Erreur")
        self.status_label.setStyleSheet("color: #f44336;")  # Rouge pour erreur
        
        # Réinitialisation de l'interface en cas d'erreur
        self.reset_interface()
        
        # Affichage d'une boîte de dialogue d'erreur
        QMessageBox.critical(
            self,
            "Erreur",
            f"Une erreur est survenue:\n{error_message}\n\nLe bot a été arrêté pour des raisons de sécurité.",
            QMessageBox.Ok
        )

    def handle_status(self, status: str):
        """
        Gère les changements de statut
        """
        try:
            if status == "running":
                self.status_label.setText("Bot en cours d'exécution")
                self.status_label.setStyleSheet("color: #4caf50;")  # Vert
            elif status == "stopped":
                self.status_label.setText("Bot arrêté")
                self.status_label.setStyleSheet("color: #f44336;")  # Rouge
            elif status == "paused":
                self.status_label.setText("Bot en pause")
                self.status_label.setStyleSheet("color: #ff9800;")  # Orange
        except Exception as e:
            self.handle_error(f"Erreur lors de la mise à jour du statut: {str(e)}")

    def handle_data_update(self, data: dict):
        """
        Gère la mise à jour des données reçues de la stratégie
        """
        try:
            if not data:
                self.log_message("Aucune donnée reçue pour la mise à jour", "warning")
                return
                
            # Mise à jour des labels avec vérification des données
            if 'price' in data:
                self.price_label.setText(f"BTC/USD: {data['price']:.2f}")
            if 'pnl' in data:
                self.pnl_label.setText(f"P/L: {data['pnl']:.2f}%")
                self.pnl_label.setStyleSheet(
                    "color: #4caf50;" if data['pnl'] >= 0 else "color: #f44336;"
                )
            if 'time' in data:
                self.last_update_label.setText(f"Dernière mise à jour: {data['time']}")
            
            # Mise à jour des graphiques avec les nouvelles données
            self.update_charts(data)
            
            # Mise à jour du statut si présent
            if 'status' in data:
                self.status_label.setText(data['status'])
                if data['status'] == "En attente d'opportunité":
                    self.status_label.setStyleSheet("color: #ff9800;")
                elif data['status'] == "Position ouverte":
                    self.status_label.setStyleSheet("color: #4caf50;")
                else:
                    self.status_label.setStyleSheet("color: #2196f3;")
                    
        except Exception as e:
            self.handle_error(f"Erreur lors de la mise à jour des données: {str(e)}")
            self.log_message(f"Données reçues: {data}", "error")

    def update_charts(self, data: dict):
        """
        Met à jour les graphiques avec les nouvelles données
        """
        try:
            if not data:
                self.log_message("Aucune donnée reçue pour la mise à jour des graphiques", "warning")
                return
                
            # Initialisation des historiques si nécessaire
            if not hasattr(self, 'price_history'):
                self.price_history = []
                self.time_history = []
                self.rsi_history = []
                self.macd_history = []
                self.macd_signal_history = []
                self.bb_upper_history = []
                self.bb_lower_history = []
                self.bb_middle_history = []
                
            # Vérification et ajout des nouvelles données aux historiques
            if 'price' in data and 'time' in data:
                self.price_history.append(float(data['price']))
                self.time_history.append(data['time'])
                
                # Récupération des indicateurs depuis le dictionnaire
                indicators = data.get('indicators', {})
                self.rsi_history.append(float(indicators.get('rsi', 0)))
                self.macd_history.append(float(indicators.get('macd', 0)))
                self.macd_signal_history.append(float(indicators.get('macd_signal', 0)))
                self.bb_upper_history.append(float(indicators.get('bb_upper', 0)))
                self.bb_lower_history.append(float(indicators.get('bb_lower', 0)))
                self.bb_middle_history.append(float(indicators.get('bb_middle', 0)))
                
                # Limitation de l'historique à 100 points
                max_points = 100
                if len(self.price_history) > max_points:
                    self.price_history = self.price_history[-max_points:]
                    self.time_history = self.time_history[-max_points:]
                    self.rsi_history = self.rsi_history[-max_points:]
                    self.macd_history = self.macd_history[-max_points:]
                    self.macd_signal_history = self.macd_signal_history[-max_points:]
                    self.bb_upper_history = self.bb_upper_history[-max_points:]
                    self.bb_lower_history = self.bb_lower_history[-max_points:]
                    self.bb_middle_history = self.bb_middle_history[-max_points:]
                
                # Mise à jour du graphique des prix
                self.price_canvas.axes.clear()
                self.price_canvas.axes.plot(self.time_history, self.price_history, 'b-', label='Prix', linewidth=1)
                self.price_canvas.axes.plot(self.time_history, self.bb_upper_history, 'r--', label='BB Supérieure', linewidth=1)
                self.price_canvas.axes.plot(self.time_history, self.bb_lower_history, 'g--', label='BB Inférieure', linewidth=1)
                self.price_canvas.axes.plot(self.time_history, self.bb_middle_history, 'y--', label='BB Moyenne', linewidth=1)
                self.price_canvas.axes.set_title('Prix BTC/USD', color='#cdd6f4', pad=10)
                self.price_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
                self.price_canvas.axes.set_ylabel('Prix', color='#cdd6f4')
                self.price_canvas.axes.grid(True, alpha=0.3)
                self.price_canvas.axes.legend(loc='upper left')
                
                # Mise à jour du graphique RSI
                self.rsi_canvas.axes.clear()
                self.rsi_canvas.axes.plot(self.time_history, self.rsi_history, 'b-', label='RSI', linewidth=1)
                self.rsi_canvas.axes.axhline(y=self.config['rsi_overbought'], color='r', linestyle='--', label='Surachat')
                self.rsi_canvas.axes.axhline(y=self.config['rsi_oversold'], color='g', linestyle='--', label='Survente')
                self.rsi_canvas.axes.set_title('RSI', color='#cdd6f4', pad=10)
                self.rsi_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
                self.rsi_canvas.axes.set_ylabel('RSI', color='#cdd6f4')
                self.rsi_canvas.axes.grid(True, alpha=0.3)
                self.rsi_canvas.axes.legend(loc='upper left')
                
                # Mise à jour du graphique MACD
                self.macd_canvas.axes.clear()
                self.macd_canvas.axes.plot(self.time_history, self.macd_history, 'b-', label='MACD', linewidth=1)
                self.macd_canvas.axes.plot(self.time_history, self.macd_signal_history, 'r-', label='Signal', linewidth=1)
                self.macd_canvas.axes.set_title('MACD', color='#cdd6f4', pad=10)
                self.macd_canvas.axes.set_xlabel('Temps', color='#cdd6f4')
                self.macd_canvas.axes.set_ylabel('MACD', color='#cdd6f4')
                self.macd_canvas.axes.grid(True, alpha=0.3)
                self.macd_canvas.axes.legend(loc='upper left')
                
                # Ajustement automatique des limites et rafraîchissement
                for canvas in [self.price_canvas, self.rsi_canvas, self.macd_canvas]:
                    canvas.axes.relim()
                    canvas.axes.autoscale_view()
                    canvas.figure.tight_layout()
                    canvas.draw()
            
        except Exception as e:
            self.handle_error(f"Erreur lors de la mise à jour des graphiques: {str(e)}")
            self.log_message(f"Données reçues pour les graphiques: {data}", "error")
        
    def start_bot(self):
        """
        Démarre le bot de trading avec une meilleure gestion des erreurs
        """
        try:
            self.log_message("Démarrage du bot de trading...")
            
            # Vérification si le bot est déjà en cours d'exécution
            if hasattr(self, 'strategy') and self.strategy and self.strategy.running:
                self.log_message("Le bot est déjà en cours d'exécution")
                return
            
            # Vérification de la connexion MT5
            if not hasattr(self, 'mt5_connector') or not self.mt5_connector:
                self.log_message("Création du connecteur MT5...")
                self.mt5_connector = MT5Connector()
            
            if not self.mt5_connector.initialized:
                self.log_message("Initialisation de la connexion MT5...")
                if not self.mt5_connector.initialize():
                    raise Exception("Échec de l'initialisation MT5")
            
            if not self.mt5_connector.connected:
                self.log_message("Connexion à MT5...")
                # Récupération des informations de connexion depuis les variables d'environnement
                login = os.getenv('MT5_LOGIN')
                password = os.getenv('MT5_PASSWORD')
                server = os.getenv('MT5_SERVER')
                
                if not all([login, password, server]):
                    raise Exception("Informations de connexion MT5 manquantes dans le fichier .env")
                
                if not self.mt5_connector.login(login=int(login), password=password, server=server):
                    raise Exception(f"Échec de la connexion MT5: {self.mt5_connector.get_last_error()}")
            
            # Mise à jour de la configuration
            self.update_config()
            
            # Création de la stratégie
            self.log_message("Création de la stratégie de trading...")
            self.strategy = ScalperStrategy(self.config)
            
            # Connexion des signaux
            self.strategy.log_message.connect(self.log_message)
            self.strategy.error_occurred.connect(self.handle_error)
            self.strategy.data_updated.connect(self.handle_data_update)
            self.strategy.finished.connect(self.on_strategy_finished)
            
            # Démarrage de la stratégie
            self.strategy.start_trading()
            
            # Mise à jour de l'interface
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.status_label.setText("Bot en cours d'exécution")
            self.status_label.setStyleSheet("color: #4caf50;")  # Vert pour en cours
            
            self.log_message("Bot démarré avec succès")
            
        except Exception as e:
            self.handle_error(f"Erreur lors du démarrage du bot: {str(e)}")
            self.reset_interface()

    def stop_bot(self):
        """
        Arrête le bot de trading
        """
        try:
            if not self.strategy:
                return
                
            self.log_message("Arrêt du bot de trading...")
            self.strategy.stop_trading()
            
            # Mise à jour de l'interface
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.status_label.setText("Bot arrêté")
            self.status_label.setStyleSheet("color: #f44336;")  # Rouge pour arrêté
            
            self.log_message("Bot arrêté avec succès")
            
        except Exception as e:
            self.handle_error(f"Erreur lors de l'arrêt du bot: {str(e)}")
            self.reset_interface()

    def pause_bot(self):
        """
        Met en pause ou reprend le bot de trading
        """
        try:
            if self.strategy:
                if self.strategy.pause:
                    # Reprise du bot
                    self.strategy.resume_trading()
                    self.pause_button.setText("Pause")
                    self.log_message("Bot repris")
                else:
                    # Mise en pause du bot
                    self.strategy.pause_trading()
                    self.pause_button.setText("Reprendre")
                    self.log_message("Bot en pause")
                    
        except Exception as e:
            self.handle_error(f"Erreur lors de la mise en pause/reprise du bot: {str(e)}")
            
    def closeEvent(self, event):
        """
        Gère la fermeture de l'application
        """
        try:
            # Arrêt du bot si en cours d'exécution
            if self.strategy and self.strategy.running:
                self.stop_bot()
                
            # Fermeture de la connexion MT5
            if self.mt5_connector:
                self.mt5_connector.shutdown()
                
            event.accept()
            
        except Exception as e:
            self.handle_error(f"Erreur lors de la fermeture de l'application: {str(e)}")
            event.accept()
            
    def resizeEvent(self, event):
        """
        Gère le redimensionnement de la fenêtre de manière simplifiée
        """
        try:
            # Appel de la méthode parente
            super().resizeEvent(event)
            
            # Pas besoin de calculs complexes, Qt gère déjà le redimensionnement
            # via les layouts et les splitters
            
            # Mise à jour des graphiques si des données sont disponibles
            if hasattr(self, 'price_history') and self.price_history:
                self.update_charts({
                    'price': self.price_history[-1],
                    'time': self.time_history[-1],
                    'rsi': self.rsi_history[-1] if hasattr(self, 'rsi_history') else 0,
                    'macd': self.macd_history[-1] if hasattr(self, 'macd_history') else 0,
                    'macd_signal': self.macd_signal_history[-1] if hasattr(self, 'macd_signal_history') else 0,
                    'bb_upper': self.bb_upper_history[-1] if hasattr(self, 'bb_upper_history') else 0,
                    'bb_lower': self.bb_lower_history[-1] if hasattr(self, 'bb_lower_history') else 0,
                    'bb_middle': self.bb_middle_history[-1] if hasattr(self, 'bb_middle_history') else 0
                })
            
        except Exception as e:
            self.log_message(f"Erreur lors du redimensionnement de la fenêtre: {str(e)}")
            # En cas d'erreur, on essaie de restaurer une taille minimale acceptable
            self.setMinimumSize(1200, 800)

    def reset_interface(self):
        """
        Réinitialise l'interface utilisateur
        """
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.status_label.setText("En attente")
        self.status_label.setStyleSheet("color: #ff9800;")  # Orange pour en attente
        
        # Ne pas réinitialiser les graphiques, juste mettre à jour leur état
        if hasattr(self, 'price_history') and self.price_history:
            # Mise à jour des graphiques avec les dernières données
            self.update_charts({
                'price': self.price_history[-1],
                'time': self.time_history[-1],
                'indicators': {
                    'rsi': self.rsi_history[-1] if self.rsi_history else 0,
                    'macd': self.macd_history[-1] if self.macd_history else 0,
                    'macd_signal': self.macd_signal_history[-1] if self.macd_signal_history else 0,
                    'bb_upper': self.bb_upper_history[-1] if self.bb_upper_history else 0,
                    'bb_lower': self.bb_lower_history[-1] if self.bb_lower_history else 0,
                    'bb_middle': self.bb_middle_history[-1] if self.bb_middle_history else 0
                }
            })
        
        # Nettoyage de la stratégie
        if hasattr(self, 'strategy') and self.strategy:
            self.strategy.stop_trading()
            self.strategy = None

    def clear_charts(self):
        """
        Réinitialise les graphiques
        """
        for canvas in [self.price_canvas, self.rsi_canvas, self.macd_canvas]:
            if canvas:
                canvas.figure.clear()
                canvas.draw()
        
        # Réinitialisation des données
        self.price_history = []
        self.time_history = []
        self.rsi_history = []
        self.macd_history = []
        self.macd_signal_history = []
        self.bb_upper_history = []
        self.bb_lower_history = []
        self.bb_middle_history = []

    def update_ui(self):
        """Met à jour l'interface utilisateur avec les dernières données de manière thread-safe"""
        try:
            if self.strategy and self.strategy.running:
                # Mise à jour du statut
                self.status_label.setText("Bot en cours d'exécution")
                self.status_label.setStyleSheet("color: #4caf50;")  # Vert pour en cours
                
                # Les autres mises à jour sont gérées par handle_data_update
                
        except Exception as e:
            self.log_message(f"Erreur lors de la mise à jour de l'interface: {str(e)}")

    def on_strategy_finished(self):
        """
        Gère la fin de la stratégie de trading
        """
        try:
            self.log_message("La stratégie de trading a terminé son exécution")
            
            # Réinitialisation de l'interface
            self.reset_interface()
            
            # Nettoyage
            if self.strategy:
                self.strategy.deleteLater()
                self.strategy = None
            
        except Exception as e:
            self.handle_error(f"Erreur lors de la fin de la stratégie: {str(e)}")
            self.reset_interface()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = ScalperBotApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 