from PyQt6.QtCore import QThread, pyqtSignal
import random
import time
from queue import Queue
import pandas as pd
import os
import joblib
from bitcoin_scalper.core.config import SecureConfig
from bitcoin_scalper.core.data_cleaner import DataCleaner
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.timescaledb_client import TimescaleDBClient
from bitcoin_scalper.core.dvc_manager import DVCManager
from bitcoin_scalper.core.export import load_objects
from bitcoin_scalper.core.modeling import predict
from bitcoin_scalper.core.order_algos import execute_iceberg, execute_vwap
from bot.connectors.mt5_rest_client import MT5RestClient
from scripts.prepare_features import generate_signal

class TradingWorker(QThread):
    new_ohlcv = pyqtSignal(object)
    features_ready = pyqtSignal(object)
    prediction_ready = pyqtSignal(str)
    order_executed = pyqtSignal(object)
    risk_update = pyqtSignal(object)
    log_message = pyqtSignal(str)
    positions_updated = pyqtSignal(list)
    account_info_updated = pyqtSignal(dict)
    risk_metrics_updated = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._interrupted = False
        self.config = None
        self.aes_key = None
        self.symbol = "BTCUSD"
        self.timeframe = "M1"
        self.order_volume = 0.01
        self.rsi_period = 14
        self.loop_interval = 10
        self.mt5_client = None
        self.cleaner = None
        self.fe = None
        self.db_client = None
        self.dvc = None
        self.ml_pipe = None
        self.ml_loaded = False
        self.risk = None
        self.positions = []

    def set_config(self, aes_key, config_path="config.enc"):
        self.aes_key = aes_key
        self.config = SecureConfig(config_path, aes_key)

    def run(self):
        """
        Boucle principale du worker de trading (live ou backtest).
        Loggue toute erreur de features manquantes et applique un fallback automatique (stratégie alternative) si besoin.
        """
        try:
            if not self.config:
                self.log_message.emit("[Worker] Erreur : config non initialisée.")
                self.finished.emit()
                return
            self.log_message.emit("[Worker] Initialisation des modules métier...")
            mt5_url = self.config.get("MT5_REST_URL")
            mt5_api_key = self.config.get("MT5_REST_API_KEY")
            self.mt5_client = MT5RestClient(mt5_url, api_key=mt5_api_key)
            self.cleaner = DataCleaner()
            self.fe = FeatureEngineering()
            db_host = self.config.get("TSDB_HOST")
            db_port = int(self.config.get("TSDB_PORT", 5432))
            db_name = self.config.get("TSDB_NAME")
            db_user = self.config.get("TSDB_USER")
            db_password = self.config.get("TSDB_PASSWORD")
            db_sslmode = self.config.get("TSDB_SSLMODE", "require")
            self.db_client = TimescaleDBClient(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password,
                sslmode=db_sslmode
            )
            self.db_client.create_schema()
            self.dvc = DVCManager()
            ml_model_path = self.config.get("ML_MODEL_PATH", "model_rf")
            try:
                objects = load_objects(ml_model_path)
                self.ml_pipe = objects.get("model")
                self.ml_loaded = self.ml_pipe is not None
                if self.ml_loaded:
                    self.log_message.emit(f"[Worker] Modèle ML chargé depuis {ml_model_path}")
                else:
                    self.log_message.emit(f"[Worker] Aucun modèle ML trouvé à {ml_model_path}. Fallback sur stratégie algo projet.")
            except Exception as e:
                self.log_message.emit(f"[Worker] Erreur chargement modèle ML : {e}. Fallback sur stratégie algo projet.")
                self.ml_loaded = False
            self.risk = RiskManager(self.mt5_client)
            self.log_message.emit("[Worker] Boucle de trading démarrée.")
            positions_api_missing = False
            while not self._interrupted:
                try:
                    # --- Récupération infos de compte ---
                    try:
                        account_info = self.mt5_client._request("GET", "/account")
                        self.account_info_updated.emit(account_info)
                    except Exception as e:
                        self.log_message.emit(f"[Worker] Erreur récupération infos compte : {e}")
                    # --- Fin récupération infos de compte ---
                    # --- Récupération positions ---
                    try:
                        positions = self.mt5_client.get_positions()
                        self.positions_updated.emit(positions)
                    except Exception as e:
                        self.log_message.emit(f"[Worker] Erreur récupération positions : {e}")
                        positions = []
                    # --- Calcul dynamique des métriques de risque ---
                    risk_metrics = self.compute_risk_metrics(account_info, positions)
                    self.risk_metrics_updated.emit(risk_metrics)
                    # --- Fin calcul risque ---
                    ohlcv = self.mt5_client.get_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.rsi_period+1)
                    if len(ohlcv) < self.rsi_period+1:
                        self.log_message.emit("[Worker] Pas assez de données OHLCV pour calculer RSI.")
                        time.sleep(self.loop_interval)
                        continue
                    cleaned = self.cleaner.clean_ohlcv(ohlcv)
                    df = pd.DataFrame(cleaned)
                    self.new_ohlcv.emit(df)
                    required_cols = {'close', 'open', 'high', 'low'}
                    if not required_cols.issubset(df.columns):
                        self.log_message.emit(f"[Worker] Colonnes manquantes dans OHLCV: {set(required_cols) - set(df.columns)}. Data brute: {df}")
                        time.sleep(self.loop_interval)
                        continue
                    df = self.fe.add_indicators(df)
                    self.features_ready.emit(df)
                    signal = None
                    if self.ml_loaded:
                        try:
                            features_list_path = "features_list.pkl"
                            if os.path.exists(features_list_path):
                                features_list = joblib.load(features_list_path)
                                missing_cols = [col for col in features_list if col not in df.columns]
                                if missing_cols:
                                    self.log_message.emit(f"[Worker][ERREUR] Colonnes de features manquantes pour la prédiction ML : {missing_cols}. Fallback automatique sur stratégie alternative.")
                                    # Fallback immédiat
                                    self.ml_loaded = False
                                    raise RuntimeError("Features manquantes : fallback sur stratégie alternative.")
                                X_pred = df[[col for col in features_list if col in df.columns]]
                            else:
                                self.log_message.emit("[Worker] features_list.pkl introuvable. Fallback sur stratégie algo projet.")
                                self.ml_loaded = False
                                raise RuntimeError("features_list.pkl manquant : fallback sur stratégie algo projet.")
                            pred = predict(self.ml_pipe, X_pred)[-1]
                            if pred == 1:
                                signal = "buy"
                            elif pred == -1:
                                signal = "sell"
                            else:
                                signal = None
                            self.prediction_ready.emit(signal or "hold")
                        except Exception as e:
                            self.log_message.emit(f"[Worker] Erreur prédiction ML : {e}. Fallback sur stratégie algo projet.")
                            self.ml_loaded = False
                    if not self.ml_loaded:
                        df_signal = generate_signal(df)
                        pred = df_signal['signal'].iloc[-1]
                        if pred == 1:
                            signal = "buy"
                        elif pred == -1:
                            signal = "sell"
                        else:
                            signal = None
                        self.prediction_ready.emit(signal or "hold")
                    if signal:
                        close_price = df["close"].iloc[-1]
                        atr = df["atr"].iloc[-1] if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else None
                        sl, tp = None, None
                        if atr and not pd.isna(atr) and atr > 0:
                            sl_mult = float(self.config.get("SL_ATR_MULT", 2.0))
                            tp_mult = float(self.config.get("TP_ATR_MULT", 3.0))
                            if signal == "buy":
                                sl = close_price - sl_mult * atr
                                tp = close_price + tp_mult * atr
                            elif signal == "sell":
                                sl = close_price + sl_mult * atr
                                tp = close_price - tp_mult * atr
                        else:
                            sl_pct = float(self.config.get("DEFAULT_SL_PCT", 0.01))
                            tp_pct = float(self.config.get("DEFAULT_TP_PCT", 0.02))
                            if signal == "buy":
                                sl = close_price * (1 - sl_pct)
                                tp = close_price * (1 + tp_pct)
                            elif signal == "sell":
                                sl = close_price * (1 + sl_pct)
                                tp = close_price * (1 - tp_pct)
                        risk_check = self.risk.can_open_position(self.symbol, self.order_volume)
                        self.risk_update.emit(risk_check)
                        if risk_check["allowed"]:
                            self.log_message.emit(f"[INFO] Passage d'un ordre {signal} ({self.order_volume} {self.symbol}) en cours (algo: {self.config.get('EXEC_ALGO', 'market')})...")
                            try:
                                exec_algo = self.config.get("EXEC_ALGO", "market").lower()
                                if exec_algo == "iceberg":
                                    res = execute_iceberg(self.order_volume, max_child=0.005, price=close_price, send_order_fn=lambda **kwargs: self.mt5_client.send_order(self.symbol, signal, kwargs['qty'], price=kwargs['price'], sl=sl, tp=tp))
                                elif exec_algo == "vwap":
                                    res = execute_vwap(self.order_volume, price_series=list(df["close"].tail(10)), send_order_fn=lambda **kwargs: self.mt5_client.send_order(self.symbol, signal, kwargs['qty'], price=kwargs['price'], sl=sl, tp=tp))
                                else:
                                    res = [self.mt5_client.send_order(self.symbol, signal, self.order_volume, price=close_price, sl=sl, tp=tp)]
                                self.order_executed.emit(res)
                                self.log_message.emit(f"[INFO] Ordre {signal} exécuté via {exec_algo} | SL={sl:.2f}, TP={tp:.2f}, prix={close_price:.2f}")
                                self.dvc.add(f"data/features/{self.symbol}_{self.timeframe}.csv")
                                self.dvc.commit(f"data/features/{self.symbol}_{self.timeframe}.csv")
                            except Exception as e:
                                self.log_message.emit(f"[ERROR] Erreur exécution ordre : {e}")
                        else:
                            self.log_message.emit(f"[WARNING] Ordre refusé par gestion du risque : {risk_check['reason']}")
                    # Stockage TimescaleDB
                    try:
                        self.db_client.insert_ohlcv(df.to_dict(orient="records"))
                    except Exception as e:
                        self.log_message.emit(f"[Worker] Erreur stockage TimescaleDB : {e}")
                    # Versioning DVC
                    try:
                        features_path = f"data/features/{self.symbol}_{self.timeframe}.csv"
                        df.to_csv(features_path, index=False)
                        self.dvc.add(features_path)
                        self.dvc.commit(features_path)
                    except Exception as e:
                        self.log_message.emit(f"[Worker] Erreur versioning DVC : {e}")
                    time.sleep(self.loop_interval)
                except Exception as e:
                    self.log_message.emit(f"[Worker] Erreur dans la boucle principale : {e}")
                    time.sleep(self.loop_interval)
            self.log_message.emit("[Worker] Arrêt propre du worker.")
            self.finished.emit()
        except Exception as e:
            self.log_message.emit(f"[Worker] Erreur critique à l'initialisation : {e}")
            self.finished.emit()

    def requestInterruption(self):
        self._interrupted = True

    def start_trading(self):
        if not self.isRunning():
            self._interrupted = False
            self.start()

    def stop_trading(self):
        self.requestInterruption()

    def compute_risk_metrics(self, account_info, positions):
        """Calcule drawdown, PnL journalier, peak balance, capital actuel à partir des vraies données."""
        try:
            if not account_info:
                return {}
            balance = account_info.get("balance")
            equity = account_info.get("equity")
            profit = account_info.get("profit")
            # Historique du capital (à améliorer si historique disponible)
            peak_balance = balance  # approximation : le solde actuel
            last_balance = balance
            # PnL journalier (à améliorer si historique disponible)
            daily_pnl = profit
            # Drawdown (approximé)
            drawdown = 0.0
            if peak_balance and last_balance:
                drawdown = (last_balance - peak_balance) / peak_balance if peak_balance > 0 else 0.0
            return {
                "drawdown": drawdown,
                "daily_pnl": daily_pnl,
                "peak_balance": peak_balance,
                "last_balance": last_balance
            }
        except Exception as e:
            self.log_message.emit(f"[Worker] Erreur calcul métriques risque : {e}")
            return {} 