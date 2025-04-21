#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module d'interface utilisateur pour le bot de scalping
Fournit une interface graphique moderne pour contrôler et surveiller le bot
"""

import os
import sys
import logging
import json
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QToolTip, QStyle,
    QStyleFactory, QFrame, QSplitter, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
import pyqtgraph as pg
import numpy as np
import pandas as pd

class TradingUI(QMainWindow):
    """
    Interface utilisateur moderne du bot de scalping
    """
    
    def __init__(self, bot):
        """
        Initialisation de l'interface utilisateur
        
        Args:
            bot (ScalperBot): Instance du bot de scalping
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.bot = bot
        
        # Données pour l'UI
        self.market_data = None
        self.indicators = None
        self.price_history = []
        self.indicator_history = {
            'ema_short': [],
            'ema_long': [],
            'rsi': []
        }
        self.order_history = []
        
        # État de l'UI
        self.running = False
        self.refresh_interval = 1000  # Intervalle de rafraîchissement en millisecondes
        
        # Configuration de l'interface
        self.setWindowTitle("Scalper Adaptatif BTC/USD")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QGroupBox {
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                margin-top: 1em;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #0d47a1;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #424242;
            }
            QPushButton#stopButton {
                background-color: #b71c1c;
            }
            QPushButton#stopButton:hover {
                background-color: #d32f2f;
            }
            QTableWidget {
                border: 1px solid #3e3e3e;
                gridline-color: #3e3e3e;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                padding: 5px;
                border: 1px solid #3e3e3e;
            }
            QTextEdit {
                border: 1px solid #3e3e3e;
                background-color: #2d2d2d;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        
        # Initialisation de l'interface
        self.init_ui()
        
        self.logger.info("Interface utilisateur initialisée")
    
    def init_ui(self):
        """
        Initialise l'interface utilisateur
        """
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Barre d'état supérieure
        status_bar = QHBoxLayout()
        
        # Statut du bot
        self.status_label = QLabel("Statut: Arrêté")
        self.status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
        status_bar.addWidget(self.status_label)
        
        # Prix actuel
        self.price_label = QLabel("Prix BTC/USD: 0.00")
        self.price_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_bar.addWidget(self.price_label)
        
        # Profit/Perte
        self.pnl_label = QLabel("P/L: 0.00")
        self.pnl_label.setStyleSheet("font-weight: bold;")
        status_bar.addWidget(self.pnl_label)
        
        status_bar.addStretch()
        
        # Dernière mise à jour
        self.last_update_label = QLabel("Dernière mise à jour: Jamais")
        status_bar.addWidget(self.last_update_label)
        
        main_layout.addLayout(status_bar)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Zone supérieure (Configuration et Contrôle)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Groupe de configuration
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()
        
        # Symboles
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['BTC/USD', 'ETH/USD', 'BNB/USD'])
        config_layout.addRow("Symbole:", self.symbol_combo)
        
        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
        config_layout.addRow("Timeframe:", self.timeframe_combo)
        
        # Paramètres RSI
        self.rsi_period_spin = QSpinBox()
        self.rsi_period_spin.setRange(2, 100)
        self.rsi_period_spin.setValue(14)
        config_layout.addRow("Période RSI:", self.rsi_period_spin)
        
        self.rsi_overbought_spin = QSpinBox()
        self.rsi_overbought_spin.setRange(50, 100)
        self.rsi_overbought_spin.setValue(70)
        config_layout.addRow("RSI Suracheté:", self.rsi_overbought_spin)
        
        self.rsi_oversold_spin = QSpinBox()
        self.rsi_oversold_spin.setRange(0, 50)
        self.rsi_oversold_spin.setValue(30)
        config_layout.addRow("RSI Survente:", self.rsi_oversold_spin)
        
        config_group.setLayout(config_layout)
        top_layout.addWidget(config_group)
        
        # Groupe de contrôle
        control_group = QGroupBox("Contrôle")
        control_layout = QVBoxLayout()
        
        # Boutons de contrôle
        self.start_button = QPushButton("Démarrer")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.on_start)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.clicked.connect(self.on_pause)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)
        
        control_group.setLayout(control_layout)
        top_layout.addWidget(control_group)
        
        main_splitter.addWidget(top_widget)
        
        # Zone inférieure (Graphiques et Données)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Onglets pour les graphiques et données
        tabs = QTabWidget()
        
        # Onglet des graphiques
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        
        # Graphique des prix et EMAs
        self.price_plot = pg.PlotWidget()
        self.price_plot.setBackground('#2d2d2d')
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', 'Prix (USD)')
        self.price_plot.setLabel('bottom', 'Temps')
        
        # Graphique du RSI
        self.rsi_plot = pg.PlotWidget()
        self.rsi_plot.setBackground('#2d2d2d')
        self.rsi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rsi_plot.setLabel('left', 'RSI')
        self.rsi_plot.setLabel('bottom', 'Temps')
        self.rsi_plot.setYRange(0, 100)
        
        # Lignes de référence RSI
        self.rsi_plot.addLine(y=70, pen='r', label='Surachat')
        self.rsi_plot.addLine(y=30, pen='g', label='Survente')
        
        chart_layout.addWidget(self.price_plot)
        chart_layout.addWidget(self.rsi_plot)
        
        tabs.addTab(chart_tab, "Graphiques")
        
        # Onglet des positions
        positions_tab = QWidget()
        positions_layout = QVBoxLayout(positions_tab)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels([
            "ID", "Type", "Prix d'entrée", "Stop Loss", "Take Profit", "Profit/Perte"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        positions_layout.addWidget(self.positions_table)
        
        tabs.addTab(positions_tab, "Positions")
        
        # Onglet de l'historique
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Date/Heure", "Type", "Prix d'Entrée", "Prix de Sortie", "Profit"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        history_layout.addWidget(self.history_table)
        
        tabs.addTab(history_tab, "Historique")
        
        # Onglet des indicateurs
        indicators_tab = QWidget()
        indicators_layout = QVBoxLayout(indicators_tab)
        
        self.indicators_table = QTableWidget()
        self.indicators_table.setColumnCount(3)
        self.indicators_table.setHorizontalHeaderLabels([
            "Indicateur", "Valeur", "Signal"
        ])
        self.indicators_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        indicators_layout.addWidget(self.indicators_table)
        
        tabs.addTab(indicators_tab, "Indicateurs")
        
        # Onglet des logs
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)
        
        tabs.addTab(logs_tab, "Logs")
        
        bottom_layout.addWidget(tabs)
        
        main_splitter.addWidget(bottom_widget)
        
        main_layout.addWidget(main_splitter)
        
        # Timer pour les mises à jour
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(self.refresh_interval)
    
    def update_ui(self):
        """
        Met à jour l'interface utilisateur périodiquement
        """
        if not self.running:
            return
        
        try:
            # Vérification des messages dans la file d'attente
            while not self.bot.ui_queue.empty():
                action, args = self.bot.ui_queue.get_nowait()
                if action == 'update':
                    market_data, indicators = args
                    self.update_data(market_data, indicators)
                elif action == 'notify_order':
                    signal, order_result = args
                    self.notify_order(signal, order_result)
            
            # Mise à jour de l'horodatage
            self.last_update_label.setText(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de l'UI: {str(e)}")
    
    def update_data(self, market_data, indicators):
        """
        Met à jour les données de marché et les indicateurs
        
        Args:
            market_data (pd.DataFrame): Données de marché
            indicators (TechnicalIndicators): Instance des indicateurs techniques
        """
        try:
            if market_data is not None and len(market_data) > 0:
                self.market_data = market_data
                self.indicators = indicators
                
                # Mise à jour du prix actuel
                current_price = market_data['close'].iloc[-1]
                self.price_label.setText(f"Prix BTC/USD: {current_price:.2f}")
                
                # Mise à jour de l'historique des prix
                self.price_history.append(current_price)
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]
                
                # Mise à jour des indicateurs
                latest_values = indicators.get_latest_values()
                if latest_values:
                    for indicator, value in latest_values.items():
                        if indicator in self.indicator_history:
                            self.indicator_history[indicator].append(value)
                            if len(self.indicator_history[indicator]) > 100:
                                self.indicator_history[indicator] = self.indicator_history[indicator][-100:]
                
                # Mise à jour des graphiques
                self.update_charts()
                
                # Mise à jour des tableaux
                self.update_indicators_table()
                self.update_positions_table()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des données: {str(e)}")
    
    def update_charts(self):
        """
        Met à jour les graphiques
        """
        try:
            if not self.price_history:
                return
            
            # Nettoyage des graphiques
            self.price_plot.clear()
            self.rsi_plot.clear()
            
            # Tracé du prix
            x = range(len(self.price_history))
            self.price_plot.plot(x, self.price_history, pen='w', name="Prix")
            
            # Tracé des EMAs
            if 'ema_short' in self.indicator_history and self.indicator_history['ema_short']:
                self.price_plot.plot(
                    x[-len(self.indicator_history['ema_short']):],
                             self.indicator_history['ema_short'], 
                    pen='b',
                    name="EMA Court"
                )
            
            if 'ema_long' in self.indicator_history and self.indicator_history['ema_long']:
                self.price_plot.plot(
                    x[-len(self.indicator_history['ema_long']):],
                             self.indicator_history['ema_long'], 
                    pen='r',
                    name="EMA Long"
                )
            
            # Tracé du RSI
            if 'rsi' in self.indicator_history and self.indicator_history['rsi']:
                self.rsi_plot.plot(
                    x[-len(self.indicator_history['rsi']):],
                             self.indicator_history['rsi'], 
                    pen='y',
                    name="RSI"
                )
            
            # Ajout des légendes
            self.price_plot.addLegend()
            self.rsi_plot.addLegend()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des graphiques: {str(e)}")
    
    def update_indicators_table(self):
        """
        Met à jour le tableau des indicateurs
        """
        try:
            # Effacement des données existantes
            self.indicators_table.setRowCount(0)
            
            if self.indicators is None:
                return
            
            # Récupération des dernières valeurs
            latest_values = self.indicators.get_latest_values()
            if not latest_values:
                return
            
            # Mise à jour des indicateurs
            indicators_data = []
            
            # EMAs
            if 'ema_short' in latest_values:
                indicators_data.append(("EMA Court", f"{latest_values['ema_short']:.2f}", ""))
            if 'ema_long' in latest_values:
                indicators_data.append(("EMA Long", f"{latest_values['ema_long']:.2f}", ""))
            
            # RSI
            if 'rsi' in latest_values and latest_values['rsi'] is not None:
                rsi_value = latest_values['rsi']
                rsi_signal = self._get_rsi_signal(rsi_value)
                indicators_data.append(("RSI", f"{rsi_value:.2f}", rsi_signal))
            
            # MACD
            if all(k in latest_values for k in ['macd', 'macd_signal']):
                macd_value = latest_values['macd']
                macd_signal = latest_values['macd_signal']
                if macd_value is not None and macd_signal is not None:
                    macd_signal_text = self._get_macd_signal(latest_values)
                    indicators_data.append(("MACD", f"{macd_value:.5f}", macd_signal_text))
                    indicators_data.append(("Signal MACD", f"{macd_signal:.5f}", ""))
            
            # Stochastique
            if all(k in latest_values for k in ['stoch_k', 'stoch_d']):
                stoch_k = latest_values['stoch_k']
                stoch_d = latest_values['stoch_d']
                if stoch_k is not None and stoch_d is not None:
                    stoch_signal = self._get_stoch_signal(latest_values)
                    indicators_data.append(("Stochastique %K", f"{stoch_k:.2f}", stoch_signal))
                    indicators_data.append(("Stochastique %D", f"{stoch_d:.2f}", ""))
            
            # ATR
            if 'atr' in latest_values and latest_values['atr'] is not None:
                indicators_data.append(("ATR", f"{latest_values['atr']:.5f}", ""))
            
            # Ajout des données dans le tableau
            self.indicators_table.setRowCount(len(indicators_data))
            for i, (name, value, signal) in enumerate(indicators_data):
                self.indicators_table.setItem(i, 0, QTableWidgetItem(name))
                self.indicators_table.setItem(i, 1, QTableWidgetItem(value))
                self.indicators_table.setItem(i, 2, QTableWidgetItem(signal))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du tableau des indicateurs: {str(e)}")
    
    def update_positions_table(self):
        """
        Met à jour le tableau des positions ouvertes
        """
        try:
            # Nettoyage de la table
            self.positions_table.setRowCount(0)
            
            # Récupération des positions ouvertes via l'API
            positions = self.bot.api.client.positions_get(symbol=self.bot.api.symbol)
            
            if not positions:
                return
            
            # Remplissage de la table
            self.positions_table.setRowCount(len(positions))
            for i, position in enumerate(positions):
                self.positions_table.setItem(i, 0, QTableWidgetItem(str(position.ticket)))
                self.positions_table.setItem(i, 1, QTableWidgetItem("Achat" if position.type == 0 else "Vente"))
                self.positions_table.setItem(i, 2, QTableWidgetItem(f"{position.price_open:.2f}"))
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"{position.sl:.2f}" if position.sl != 0 else "N/A"))
                self.positions_table.setItem(i, 4, QTableWidgetItem(f"{position.tp:.2f}" if position.tp != 0 else "N/A"))
                self.positions_table.setItem(i, 5, QTableWidgetItem(f"{position.profit:.6f}"))
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des positions: {e}")
    
    def _get_rsi_signal(self, rsi):
        """Retourne le signal RSI"""
        if rsi is None:
            return ""
        if rsi > 70:
            return "Surachat"
        elif rsi < 30:
            return "Survente"
        return "Neutre"
    
    def _get_macd_signal(self, latest):
        """Retourne le signal MACD"""
        try:
            if latest['macd'] is None or latest['macd_signal'] is None:
                return ""
            if latest['macd'] > latest['macd_signal']:
                return "Haussier"
            elif latest['macd'] < latest['macd_signal']:
                return "Baissier"
            return "Neutre"
        except (KeyError, TypeError):
            return ""
    
    def _get_stoch_signal(self, latest):
        """Retourne le signal Stochastique"""
        try:
            if latest['stoch_k'] is None or latest['stoch_d'] is None:
                return ""
            k = latest['stoch_k']
            d = latest['stoch_d']
            if k > 80 and d > 80:
                return "Surachat"
            elif k < 20 and d < 20:
                return "Survente"
            elif k > d:
                return "Haussier"
            elif k < d:
                return "Baissier"
            return "Neutre"
        except (KeyError, TypeError):
            return ""
    
    def notify_order(self, signal, order_result):
        """
        Notifie l'UI d'un nouvel ordre exécuté
        
        Args:
            signal (dict): Signal de trading
            order_result (dict): Résultat de l'exécution de l'ordre
        """
        # Ajout à l'historique des ordres
        self.order_history.append({
            'type': signal['type'],
            'price': signal['price'],
            'index': len(self.price_history) - 1
        })
        
        # Limitation de l'historique
        if len(self.order_history) > 20:
            self.order_history.pop(0)
        
        # Notification visuelle
        msg = f"Ordre exécuté: {signal['type'].upper()} à {signal['price']:.2f}"
        self.log_text.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
        
        # Mise à jour immédiate des tableaux
        self.update_positions_table()
        
        # Popup de notification
        QMessageBox.information(self, "Ordre Exécuté", msg)
    
    def on_start(self):
        """
        Gère l'événement de démarrage du bot
        """
        if self.bot.running:
            QMessageBox.information(self, "Information", "Le bot est déjà en cours d'exécution.")
            return
        
        # Mise à jour de l'UI
        self.status_label.setText("Statut: En cours d'exécution")
        self.status_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        
        # Démarrage du bot dans un thread séparé
        threading.Thread(target=self.bot.start, daemon=True).start()
    
    def on_stop(self):
        """
        Gère l'événement d'arrêt du bot
        """
        if not self.bot.running:
            return
        
        # Confirmation
        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Êtes-vous sûr de vouloir arrêter le bot ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Arrêt du bot
            self.bot.stop()
            
            # Mise à jour de l'UI
            self.status_label.setText("Statut: Arrêté")
            self.status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
    
    def on_pause(self):
        """
        Gère l'événement de pause du bot
        """
        if not self.bot.running:
            return
        
        # Mise à jour de l'UI
        if self.pause_button.text() == "Pause":
            self.pause_button.setText("Reprendre")
            self.status_label.setText("Statut: En pause")
            self.status_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        else:
            self.pause_button.setText("Pause")
            self.status_label.setText("Statut: En cours d'exécution")
            self.status_label.setStyleSheet("font-weight: bold; color: #4caf50;")
    
    def closeEvent(self, event):
        """
        Gère l'événement de fermeture de l'application
        """
        if self.bot.running:
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Le bot est en cours d'exécution. Voulez-vous l'arrêter et quitter ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.bot.stop()
                self.running = False
                event.accept()
            else:
                event.ignore()
        else:
            self.running = False
            event.accept()