from PyQt6.QtWidgets import QMainWindow, QDockWidget, QTableView, QTextEdit, QMenuBar, QMessageBox, QLabel, QGraphicsRectItem
from PyQt6.QtGui import QAction, QBrush
from PyQt6.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
# import pyqtgraph.financial as pgf
import pandas as pd
import numpy as np

from .signal_panel import SignalPanel
from .risk_panel import RiskPanel
from .account_info_panel import AccountInfoPanel
from .position_delegate import PositionDelegate

class MainWindow(QMainWindow):
    start_trading = pyqtSignal()
    stop_trading = pyqtSignal()
    reload_settings = pyqtSignal()

    def __init__(self, logger, settings, positions_model):
        super().__init__()
        self.setWindowTitle("Bitcoin Scalper Demo (PyQt6)")
        self.resize(1200, 800)
        self.logger = logger
        self.settings = settings
        self.positions_model = positions_model
        self._init_menu()
        self._init_docks()
        self._ohlcv_history_df = pd.DataFrame()
        self.graph_widget = pg.PlotWidget()
        self.ohlcv_item = pg.PlotDataItem()
        self.graph_widget.addItem(self.ohlcv_item)
        self.apply_dark_theme()
        # Ajout de la barre d'état
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Déconnecté", 5000)
        # Bandeau d'alerte critique (overlay)
        self.alert_banner = QLabel("")
        self.alert_banner.setObjectName("global_alert_banner")
        self.alert_banner.setStyleSheet("background:#f44336;color:white;font-weight:bold;padding:8px 24px;border-radius:6px;font-size:16px;")
        self.alert_banner.setVisible(False)
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.centralWidget())  # S'assure que centralWidget existe
        self.layout().insertWidget(0, self.alert_banner)

    def _init_menu(self):
        menubar = self.menuBar()
        trading_menu = menubar.addMenu("Trading")
        start_action = QAction("Démarrer", self)
        stop_action = QAction("Arrêter", self)
        quit_action = QAction("Quitter", self)
        reload_action = QAction("Recharger config", self)
        trading_menu.addAction(start_action)
        trading_menu.addAction(stop_action)
        trading_menu.addAction(reload_action)
        trading_menu.addSeparator()
        trading_menu.addAction(quit_action)
        start_action.triggered.connect(self.start_trading)
        stop_action.triggered.connect(self.stop_trading)
        reload_action.triggered.connect(self.reload_settings)
        quit_action.triggered.connect(self.close)

    def _init_docks(self):
        # Infos compte/statut dock (gauche, haut)
        self.account_info_panel = AccountInfoPanel()
        account_info_dock = QDockWidget("Compte & Statut", self)
        account_info_dock.setWidget(self.account_info_panel)
        account_info_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, account_info_dock)

        # Positions dock (gauche, bas)
        self.positions_view = QTableView()
        self.positions_view.setModel(self.positions_model)
        self.positions_view.setObjectName("positions_view")
        self.positions_view.setProperty("role", "positions")
        self.positions_view.setItemDelegate(PositionDelegate(self.positions_view))
        positions_dock = QDockWidget("Positions", self)
        positions_dock.setWidget(self.positions_view)
        positions_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, positions_dock)
        self.splitDockWidget(account_info_dock, positions_dock, Qt.Orientation.Vertical)

        # Graph dock (centre)
        self.graph_widget = pg.PlotWidget(title="Ticker BTCUSD")
        graph_dock = QDockWidget("Graphique", self)
        graph_dock.setWidget(self.graph_widget)
        graph_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, graph_dock)

        # Contrôles, signaux, risque (droite)
        self.signal_panel = SignalPanel()
        signal_dock = QDockWidget("Signal de Trading", self)
        signal_dock.setWidget(self.signal_panel)
        signal_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, signal_dock)

        self.risk_panel = RiskPanel()
        risk_dock = QDockWidget("Gestion du Risque", self)
        risk_dock.setWidget(self.risk_panel)
        risk_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, risk_dock)
        self.tabifyDockWidget(signal_dock, risk_dock)
        signal_dock.raise_()

        # Log dock (bas)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setObjectName("log_console")
        self.log_console.setProperty("role", "log")
        log_dock = QDockWidget("Logs", self)
        log_dock.setWidget(self.log_console)
        log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def apply_dark_theme(self):
        """Charge la feuille de style sombre personnalisée."""
        try:
            with open("ui/styles/dark_theme.qss", "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            self.logger.error(f"Erreur chargement thème sombre : {e}")

    def update_graph(self, df):
        # df: pandas DataFrame OHLCV avec colonnes 'open', 'high', 'low', 'close', 'timestamp'
        try:
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                self.append_log(f"[INFO] update_graph: DataFrame reçu avec {len(df)} lignes.")
                # Assurer que les colonnes nécessaires sont présentes et numériques
                required_cols = ['open', 'high', 'low', 'close', 'timestamp']
                if not all(col in df.columns for col in required_cols):
                    self.append_log("[ERROR] Impossible d'afficher les bougies OHLCV : colonnes manquantes.")
                    self.graph_widget.clear()
                    self.graph_widget.setBackground('dimgrey')
                    self.graph_widget.addItem(pg.TextItem("Colonnes manquantes pour OHLCV", color='r', anchor=(0.5,0.5)))
                    return
                # Convertir le timestamp si nécessaire et le définir comme index
                if pd.api.types.is_integer_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df_plot = df.set_index('timestamp').sort_index()

                # Concaténer avec l'historique pour affichage continu
                self._ohlcv_history_df = pd.concat([self._ohlcv_history_df, df_plot]).drop_duplicates().sort_index()

                # Nettoyer tous les items du graphique sauf axes
                self.graph_widget.clear()
                self.graph_widget.setBackground('dimgrey')  # Fond gris pour vérifier la visibilité

                # Préparer les données pour BarGraphItem (affichage des bougies)
                x = np.arange(len(self._ohlcv_history_df))
                open_prices = self._ohlcv_history_df['open'].values
                high_prices = self._ohlcv_history_df['high'].values
                low_prices = self._ohlcv_history_df['low'].values
                close_prices = self._ohlcv_history_df['close'].values
                width = 0.6
                y = np.minimum(open_prices, close_prices)
                height = close_prices - open_prices
                # Couleurs : vert si hausse, rouge si baisse
                brushes = [pg.mkBrush('g') if h >= 0 else pg.mkBrush('r') for h in height]
                # Logs détaillés pour le diagnostic
                self.append_log(f"[DEBUG] open min={open_prices.min()}, max={open_prices.max()}")
                self.append_log(f"[DEBUG] close min={close_prices.min()}, max={close_prices.max()}")
                self.append_log(f"[DEBUG] high min={high_prices.min()}, max={high_prices.max()}")
                self.append_log(f"[DEBUG] low min={low_prices.min()}, max={low_prices.max()}")
                self.append_log(f"[DEBUG] height min={height.min()}, max={height.max()}, values={height}")

                # Un seul BarGraphItem pour toutes les bougies
                bar_item = pg.BarGraphItem(x=x, height=height, width=width, y=y, brushes=brushes)
                self.graph_widget.addItem(bar_item)
                # Ajouter les mèches (lignes high-low)
                for i in range(len(x)):
                    self.graph_widget.addItem(pg.PlotDataItem(x=[x[i], x[i]], y=[low_prices[i], high_prices[i]], pen=pg.mkPen('w')))

                self.graph_widget.setTitle(f"Ticker BTCUSD ({len(self._ohlcv_history_df)} bougies)")

                # Zoom automatique sur les 10 dernières bougies (plage de prix réelle)
                if len(x) > 0:
                    start_x = max(0, x[-1] - 9)
                    end_x = x[-1] + 1
                    # Utiliser les indices pour sélectionner les bons prix
                    min_y = np.min(low_prices[start_x:end_x])
                    max_y = np.max(high_prices[start_x:end_x])
                    self.graph_widget.setXRange(start_x, end_x)
                    self.graph_widget.setYRange(min_y, max_y)

                # MAJ statut bot et dernier prix
                if hasattr(self, 'account_info_panel'):
                    self.account_info_panel.set_status("running")
                    self.account_info_panel.set_last_price(float(close_prices[-1]))
            else:
                self.append_log("[WARNING] Données OHLCV vides ou invalides pour le graphique.")
                self.graph_widget.clear()
                self.graph_widget.setBackground('dimgrey')
                self.graph_widget.addItem(pg.TextItem("Aucune donnée OHLCV reçue", color='w', anchor=(0.5,0.5)))
        except Exception as e:
            self.append_log(f"[ERROR] Erreur update_graph : {e}")

    def append_log(self, message):
        # Ajoute un log coloré selon le niveau
        import re
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        # Détection du niveau
        level = "INFO"
        if message.startswith("[DEBUG]"):
            color = "#b0b0b0"; level = "DEBUG"
        elif message.startswith("[ERROR]"):
            color = "#f44336"; level = "ERROR"
        elif message.startswith("[WARNING]"):
            color = "#ffa726"; level = "WARNING"
        else:
            color = "#e0e0e3"; level = "INFO"
        # Mise en forme HTML
        msg_html = f'<span style="color:{color};"><b>[{ts}]</b> {message}</span>'
        self.log_console.append(msg_html)

    def update_positions(self, positions):
        # positions est une liste de dictionnaires
        # Mettre à jour le modèle QAbstractTableModel avec ces données
        self.positions_model.update_data(positions)
        if len(positions) > 0 or (hasattr(self, '_last_positions_count') and len(positions) != self._last_positions_count):
            self.append_log(f"[UI] {len(positions)} positions mises à jour.")
        self._last_positions_count = len(positions)

    def on_settings_reloaded(self):
        QMessageBox.information(self, "Config", "Configuration rechargée.")

    def on_worker_finished(self):
        QMessageBox.information(self, "Worker", "Le worker de trading s'est arrêté.")

    # Slots pour les nouveaux signaux métier
    def update_signal_display(self, signal):
        # signal est une chaîne: 'buy', 'sell', 'hold'
        self.signal_panel.set_signal(signal)

    def update_risk_display(self, risk_metrics):
        # risk_metrics est un dictionnaire
        self.risk_panel.set_metrics(risk_metrics)

    def show_global_alert(self, message: str) -> None:
        """Affiche un bandeau d'alerte critique en haut de la fenêtre."""
        self.alert_banner.setText(message)
        self.alert_banner.setVisible(True)

    def hide_global_alert(self) -> None:
        """Masque le bandeau d'alerte critique."""
        self.alert_banner.setVisible(False) 