import os
import sys
import json
import logging
from queue import Queue
import threading
import time
from datetime import datetime

# Bibliothèques de données et analyse
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Qt imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox,
    QTabWidget, QGroupBox, QScrollArea, QScrollBar, QFormLayout,
    QLayout, QStyle
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QFont, QGuiApplication

# Visualisation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg

# Modules internes
from bitcoin_scalper.api_connector import APIConnector
from bitcoin_scalper.mt5_connector import MT5Connector
from bitcoin_scalper.strategies.bitcoin_scalper import ScalperStrategy
from bitcoin_scalper.config.scalper_config import DEFAULT_CONFIG
from bitcoin_scalper.utils.logger import logger
from bitcoin_scalper.risk_manager import RiskManager
from bitcoin_scalper.data_manager import DataManager
from bitcoin_scalper.styles import STYLE_SHEET
from bitcoin_scalper.config_manager import ConfigManager

# ... reste du code inchangé ... 