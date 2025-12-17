from PyQt6.QtCore import QThread, pyqtSignal
import random
import time
from queue import Queue
import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime
from bitcoin_scalper.core.config import SecureConfig
from bitcoin_scalper.core.data_cleaner import DataCleaner
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.timescaledb_client import TimescaleDBClient
from bitcoin_scalper.core.dvc_manager import DVCManager
from bitcoin_scalper.core.export import load_objects
from bitcoin_scalper.core.modeling import predict
from bitcoin_scalper.core.order_algos import execute_iceberg, execute_vwap
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient
from bitcoin_scalper.scripts.prepare_features import generate_signal
# ✅ PHASE 5: INFERENCE & SAFETY modules
from bitcoin_scalper.core.inference_safety import InferenceSafetyGuard, DynamicRiskManager
from bitcoin_scalper.core.monitoring import DriftMonitor

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
        
        # ✅ PHASE 5: Initialize safety guards
        self.safety_guard = InferenceSafetyGuard(
            max_latency_ms=200.0,
            max_entropy=0.8,
            max_consecutive_errors=5,
            error_window_seconds=60
        )
        self.dynamic_risk = DynamicRiskManager(
            high_confidence_threshold=0.8,
            sl_atr_mult_confident=2.0,
            sl_atr_mult_uncertain=1.5
        )
        self.drift_monitor = None  # Initialized after loading training data
        self.drift_check_counter = 0
        self.drift_check_interval = 100  # Check drift every 100 ticks

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
                    
                    # ✅ PHASE 5: Initialize Drift Monitor with training data reference
                    try:
                        # Try to load reference training data for drift monitoring
                        train_data_path = "models/train_reference.pkl"
                        if os.path.exists(train_data_path):
                            train_ref = joblib.load(train_data_path)
                            # Select key features for drift monitoring (top 5 most important)
                            # In a real scenario, you'd load feature importance from model
                            key_features = list(train_ref.columns[:5]) if len(train_ref.columns) >= 5 else list(train_ref.columns)
                            self.drift_monitor = DriftMonitor(
                                reference_data=train_ref,
                                key_features=key_features,
                                p_value_threshold=0.05
                            )
                            self.log_message.emit(f"[Worker] ✅ Drift Monitor initialized with {len(key_features)} key features")
                        else:
                            self.log_message.emit(f"[Worker] ⚠️ No training reference data found for drift monitoring")
                    except Exception as drift_e:
                        self.log_message.emit(f"[Worker] ⚠️ Could not initialize drift monitor: {drift_e}")
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
                        print("DEBUG WORKER ACCOUNT INFO:", account_info)  # Log temporaire pour diagnostic
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
                    
                    # ✅ PHASE 5: Drift Monitor (Le Radar)
                    # Périodiquement, un test de Kolmogorov-Smirnov (KS-Test) compare les données live aux données d'entraînement
                    self.drift_check_counter += 1
                    if self.drift_monitor and self.drift_check_counter >= self.drift_check_interval:
                        try:
                            drift_report = self.drift_monitor.check_drift(df)
                            if drift_report["drift_detected"]:
                                self.log_message.emit(f"[Worker] 🚨 DRIFT DETECTED: Model may need retraining! Details: {drift_report['details']}")
                                # Optional: pause trading or send alert
                        except Exception as e:
                            self.log_message.emit(f"[Worker] Drift check error: {e}")
                        self.drift_check_counter = 0
                    
                    signal = None
                    model_confidence = 0.5  # Default confidence
                    probabilities = None
                    tick_timestamp = datetime.now()  # Get timestamp of current tick
                    
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
                                    self.safety_guard.record_error()
                                    raise RuntimeError("Features manquantes : fallback sur stratégie alternative.")
                                X_pred = df[[col for col in features_list if col in df.columns]]
                            else:
                                self.log_message.emit("[Worker] features_list.pkl introuvable. Fallback sur stratégie algo projet.")
                                self.ml_loaded = False
                                self.safety_guard.record_error()
                                raise RuntimeError("features_list.pkl manquant : fallback sur stratégie algo projet.")
                            
                            # Get prediction and probabilities
                            pred = predict(self.ml_pipe, X_pred)[-1]
                            
                            # Get probabilities if available
                            if hasattr(self.ml_pipe, 'predict_proba'):
                                try:
                                    probabilities = self.ml_pipe.predict_proba(X_pred)[-1]
                                    model_confidence = np.max(probabilities)
                                    self.log_message.emit(f"[Worker] Model confidence: {model_confidence:.3f}")
                                except Exception:
                                    probabilities = None
                                    model_confidence = 0.5
                            
                            # ✅ PHASE 5: Full Safety Check before trading
                            if probabilities is not None:
                                safe, safety_report = self.safety_guard.full_safety_check(tick_timestamp, probabilities)
                                
                                if not safe:
                                    self.log_message.emit(f"[Worker] ⛔ SAFETY CHECK FAILED: {safety_report}")
                                    self.safety_guard.record_error()
                                    signal = None  # Abort trade
                                    self.prediction_ready.emit("hold - safety abort")
                                else:
                                    # Safety checks passed
                                    self.safety_guard.record_success()
                                    if pred == 1:
                                        signal = "buy"
                                    elif pred == -1:
                                        signal = "sell"
                                    else:
                                        signal = None
                                    self.prediction_ready.emit(signal or "hold")
                            else:
                                # No probabilities available, skip safety checks but still predict
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
                            self.safety_guard.record_error()
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
                        
                        # ✅ PHASE 5: Risk Management Dynamique
                        # Le SL n'est pas fixe. Il est calculé via l'ATR actuel.
                        # Règle: Si Confiance Modèle > 0.8 → SL Large (ATR x 2). Sinon → SL Serré.
                        if atr and not pd.isna(atr) and atr > 0 and model_confidence is not None:
                            sl, tp, risk_info = self.dynamic_risk.calculate_sl_tp(
                                signal, close_price, atr, model_confidence
                            )
                            self.log_message.emit(
                                f"[Worker] Dynamic Risk: confidence={model_confidence:.2f}, "
                                f"SL={risk_info['sl_multiplier']}×ATR, TP={risk_info['tp_multiplier']}×ATR"
                            )
                        elif atr and not pd.isna(atr) and atr > 0:
                            # Fallback to config-based multipliers if no confidence available
                            sl_mult = float(self.config.get("SL_ATR_MULT", 2.0))
                            tp_mult = float(self.config.get("TP_ATR_MULT", 3.0))
                            if signal == "buy":
                                sl = close_price - sl_mult * atr
                                tp = close_price + tp_mult * atr
                            elif signal == "sell":
                                sl = close_price + sl_mult * atr
                                tp = close_price - tp_mult * atr
                        else:
                            # Fallback to percentage-based SL/TP if ATR unavailable
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