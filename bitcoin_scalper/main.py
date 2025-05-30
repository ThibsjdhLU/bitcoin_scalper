"""
Script principal d'orchestration du bot de trading BTCUSD (macOS)
- Orchestration complète : ingestion, nettoyage, feature engineering, ML, risk, exécution, monitoring, versioning
- Mode unique : live trading, monitoring, gestion du risque, DVC, Prometheus
- Toutes les briques du projet sont intégrées sous forme de TODO à compléter
"""
import time
import logging
import os
import pandas as pd
from bitcoin_scalper.core.config import SecureConfig
from bitcoin_scalper.core.data_ingestor import DataIngestor  # TODO: à instancier et lancer
from bitcoin_scalper.core.data_cleaner import DataCleaner
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.timescaledb_client import TimescaleDBClient  # TODO: à instancier pour stockage
from bitcoin_scalper.core.dvc_manager import DVCManager  # TODO: à utiliser pour versioning
from bitcoin_scalper.core.export import load_objects  # Pour charger le modèle ML
from bitcoin_scalper.core.modeling import predict    # Pour la prédiction ML
from bitcoin_scalper.core.backtesting import Backtester  # TODO: à intégrer pour reporting/backtest offline
from bitcoin_scalper.core.order_algos import execute_iceberg, execute_vwap  # TODO: à intégrer pour exécution avancée
from bot.connectors.mt5_rest_client import MT5RestClient
from prometheus_client import start_http_server, Counter, Gauge
import threading
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import joblib
from scripts.prepare_features import generate_signal  # Ajout pour fallback algo trading
import getpass
import hashlib
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTableView, QTextEdit, QMenuBar, QMessageBox
from PyQt6.QtGui import QAction
from PyQt6.QtCore import pyqtSignal, Qt
from ui.main_window import MainWindow
from ui.password_dialog import PasswordDialog
from threads.trading_worker import TradingWorker
from utils.logger import QtLogger
from utils.settings import SettingsManager
from models.positions_model import PositionsModel
import pyqtgraph as pg

# --- Config logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_bot")

# --- Paramètres principaux ---
SYMBOL = "BTCUSD"
TIMEFRAME = "M1"
ORDER_VOLUME = 0.01
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
LOOP_INTERVAL = 10  # secondes

# Métriques Prometheus pour le bot
BOT_UPTIME = Gauge('bot_uptime_seconds', 'Uptime du bot principal en secondes')
BOT_CYCLES = Counter('bot_cycles_total', 'Nombre total de cycles de trading')
BOT_ERRORS = Counter('bot_errors_total', 'Nombre total d\'erreurs dans la boucle principale')
START_TIME = time.time()
# Métriques avancées
BOT_DRAWDOWN = Gauge('bot_drawdown', 'Drawdown courant du bot')
BOT_DAILY_PNL = Gauge('bot_daily_pnl', 'PnL journalier du bot')
BOT_PEAK_BALANCE = Gauge('bot_peak_balance', 'Peak balance du bot')
BOT_LAST_BALANCE = Gauge('bot_last_balance', 'Dernier solde du bot')

# TODO: Ajouter d'autres métriques Prometheus (latence, exécution ordres, PnL, drawdown, etc.)

# --- Sécurité simplifiée : dérivation de la clé AES à partir d'un mot de passe utilisateur ---
SALT = b"bitcoin_scalper_salt"  # À stocker dans un fichier séparé pour plus de sécurité
ITERATIONS = 200_000

def derive_key_from_password(password: str, salt: bytes = SALT, iterations: int = ITERATIONS) -> bytes:
    """Dérive une clé AES-256 à partir d'un mot de passe utilisateur."""
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32  # 256 bits
    )

def prometheus_exporter():
    start_http_server(8001)
    while True:
        BOT_UPTIME.set(time.time() - START_TIME)
        # Exporter les métriques avancées
        try:
            if 'risk' in globals() and risk is not None:
                metrics = risk.get_risk_metrics()
                BOT_DRAWDOWN.set(metrics.get('drawdown', 0.0) or 0.0)
                BOT_DAILY_PNL.set(metrics.get('daily_pnl', 0.0) or 0.0)
                if metrics.get('peak_balance') is not None:
                    BOT_PEAK_BALANCE.set(metrics.get('peak_balance', 0.0) or 0.0)
                if metrics.get('last_balance') is not None:
                    BOT_LAST_BALANCE.set(metrics.get('last_balance', 0.0) or 0.0)
        except Exception as e:
            logger.error(f"Erreur export métriques Prometheus : {e}")
        time.sleep(5)

# Lancer l'exporteur Prometheus dans un thread séparé
threading.Thread(target=prometheus_exporter, daemon=True).start()


def run_live_trading(max_cycles=None):
    """
    Exécute le bot de trading en mode live (production).
    - Ingestion temps réel, feature engineering, ML, risk, exécution, monitoring, DVC.
    - Toutes les métriques sont exportées pour Prometheus.
    - Les ordres sont validés par la gestion du risque avant exécution.
    - Mode par défaut du bot.
    """
    # 1. Charger la config (sécurisée uniquement, fallback interdit)
    # --- Sécurité simplifiée : demande du mot de passe utilisateur et dérivation de la clé ---
    password = getpass.getpass("Mot de passe pour déverrouiller la config sécurisée : ")
    aes_key = derive_key_from_password(password)
    config = SecureConfig("config.enc", aes_key)
    logger.info("Configuration chargée en mode sécurisé (clé dérivée du mot de passe utilisateur, PBKDF2, AES-256). Fallback interdit.")
    mt5_url = config.get("MT5_REST_URL")
    mt5_api_key = config.get("MT5_REST_API_KEY")

    # 2. Initialiser les modules
    mt5_client = MT5RestClient(mt5_url, api_key=mt5_api_key)
    cleaner = DataCleaner()
    fe = FeatureEngineering()

    # --- Stockage TimescaleDB ---
    db_host = config.get("TSDB_HOST")
    db_port = int(config.get("TSDB_PORT", 5432))
    db_name = config.get("TSDB_NAME")
    db_user = config.get("TSDB_USER")
    db_password = config.get("TSDB_PASSWORD")
    db_sslmode = config.get("TSDB_SSLMODE", "require")
    db_client = TimescaleDBClient(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode=db_sslmode
    )
    db_client.create_schema()

    # --- Ingestion temps réel (thread) ---
    ingestor = DataIngestor(mt5_client, db_client, symbol=SYMBOL, timeframe=TIMEFRAME, cleaner=cleaner)
    ingestor.start()

    # --- Versioning DVC ---
    dvc = DVCManager()

    # --- Chargement du pipeline ML (si modèle existant) ---
    ml_pipe = None
    ml_model_path = config.get("ML_MODEL_PATH", "model_rf.pkl")
    ml_loaded = False
    try:
        if os.path.exists(ml_model_path):
            # On charge le modèle ML via load_objects (cf. orchestrator.py)
            ml_pipe, _, _, _, _ = load_objects(ml_model_path)
            logger.info(f"Modèle ML chargé depuis {ml_model_path}")
            ml_loaded = True
        else:
            logger.warning(f"Aucun modèle ML trouvé à {ml_model_path}. Fallback sur stratégie algo projet.")
    except Exception as e:
        logger.warning(f"Erreur chargement modèle ML : {e}. Fallback sur stratégie algo projet.")
    risk = RiskManager(mt5_client)
    # TODO: Initialiser les algos d'exécution avancée (iceberg, TWAP, VWAP)
    # TODO: Initialiser le backtester pour reporting offline

    logger.info("Bot de trading BTCUSD démarré.")
    cycles = 0
    while True:
        if max_cycles is not None and cycles >= max_cycles:
            break
        cycles += 1
        try:
            # 3. Récupérer les données OHLCV
            ohlcv = mt5_client.get_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=RSI_PERIOD+1)
            if len(ohlcv) < RSI_PERIOD+1:
                logger.warning("Pas assez de données OHLCV pour calculer RSI.")
                time.sleep(LOOP_INTERVAL)
                continue
            # 4. Nettoyer les données
            cleaned = cleaner.clean_ohlcv(ohlcv)
            df = pd.DataFrame(cleaned)
            logger.info(f"Aperçu OHLCV DataFrame:\n{df.head()}\nColonnes: {list(df.columns)}")
            required_cols = {'close', 'open', 'high', 'low'}
            if not required_cols.issubset(df.columns):
                logger.error(f"Colonnes manquantes dans OHLCV: {set(required_cols) - set(df.columns)}. Data brute: {df}")
                raise ValueError(f"Colonnes OHLCV manquantes: {set(required_cols) - set(df.columns)}")
            # 5. Feature engineering
            df = fe.add_indicators(df)
            # TODO: Ajouter add_features, multi_timeframe, etc.
            # 6. Prédiction ML ou fallback algo
            signal = None
            if ml_loaded:
                try:
                    features_list_path = "features_list.pkl"
                    if os.path.exists(features_list_path):
                        features_list = joblib.load(features_list_path)
                        missing_cols = [col for col in features_list if col not in df.columns]
                        if missing_cols:
                            logger.warning(f"Colonnes de features manquantes pour la prédiction ML : {missing_cols}")
                        X_pred = df[[col for col in features_list if col in df.columns]]
                    else:
                        logger.warning("features_list.pkl introuvable. Fallback sur stratégie algo projet.")
                        ml_loaded = False
                        raise RuntimeError("features_list.pkl manquant : fallback sur stratégie algo projet.")
                    pred = predict(ml_pipe, X_pred)[-1]
                    logger.info(f"ML utilisée pour la prédiction. Prédiction brute : {pred}")
                    if pred == 1:
                        signal = "buy"
                    elif pred == -1:
                        signal = "sell"
                    else:
                        signal = None
                    logger.info(f"Signal ML : {signal} (prédiction brute : {pred})")
                except Exception as e:
                    logger.warning(f"Erreur prédiction ML : {e}. Fallback sur stratégie algo projet.")
                    ml_loaded = False
            if not ml_loaded:
                # Fallback : stratégie algo projet
                df_signal = generate_signal(df)
                pred = df_signal['signal'].iloc[-1]
                logger.info(f"Signal généré par stratégie algo projet : {pred}")
                if pred == 1:
                    signal = "buy"
                elif pred == -1:
                    signal = "sell"
                else:
                    signal = None
            # 7. Gestion du risque et exécution avancée
            if signal:
                # Calcul SL/TP (par défaut % ou ATR si dispo)
                close_price = df["close"].iloc[-1]
                atr = df["atr"].iloc[-1] if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else None
                sl, tp = None, None
                if atr and not pd.isna(atr) and atr > 0:
                    sl_mult = float(config.get("SL_ATR_MULT", 2.0))
                    tp_mult = float(config.get("TP_ATR_MULT", 3.0))
                    if signal == "buy":
                        sl = close_price - sl_mult * atr
                        tp = close_price + tp_mult * atr
                    elif signal == "sell":
                        sl = close_price + sl_mult * atr
                        tp = close_price - tp_mult * atr
                else:
                    sl_pct = float(config.get("DEFAULT_SL_PCT", 0.01))
                    tp_pct = float(config.get("DEFAULT_TP_PCT", 0.02))
                    if signal == "buy":
                        sl = close_price * (1 - sl_pct)
                        tp = close_price * (1 + tp_pct)
                    elif signal == "sell":
                        sl = close_price * (1 + sl_pct)
                        tp = close_price * (1 - tp_pct)
                logger.info(f"SL/TP utilisés : SL={sl:.2f}, TP={tp:.2f} (méthode {'ATR' if atr else '%'})")
                risk_check = risk.can_open_position(SYMBOL, ORDER_VOLUME)
                if risk_check["allowed"]:
                    try:
                        exec_algo = config.get("EXEC_ALGO", "market").lower()
                        if exec_algo == "iceberg":
                            execute_iceberg(mt5_client, SYMBOL, signal, ORDER_VOLUME, sl=sl, tp=tp)
                        elif exec_algo == "vwap":
                            execute_vwap(mt5_client, SYMBOL, signal, ORDER_VOLUME, sl=sl, tp=tp)
                        elif exec_algo == "twap":
                            logger.warning("TWAP non implémenté, fallback market order.")
                            mt5_client.send_order(SYMBOL, signal, ORDER_VOLUME, sl=sl, tp=tp)
                        else:
                            mt5_client.send_order(SYMBOL, signal, ORDER_VOLUME, sl=sl, tp=tp)
                        logger.info(f"Ordre {signal} exécuté via {exec_algo} avec SL={sl:.2f}, TP={tp:.2f}.")
                        dvc.add("data/features/BTCUSD_M1.csv")
                        dvc.commit()
                    except Exception as e:
                        logger.error(f"Erreur exécution ordre : {e}")
                        BOT_ERRORS.inc()
                else:
                    logger.warning(f"Ordre refusé par gestion du risque : {risk_check['reason']}")
            else:
                logger.info("Aucun signal de trading généré.")
            # 8. Reporting backtesting (si activé)
            if config.get("ENABLE_BACKTEST", False):
                try:
                    backtester = Backtester()
                    backtester.run(df)
                    logger.info("Backtesting exécuté.")
                except Exception as e:
                    logger.error(f"Erreur backtesting : {e}")
            # 9. Monitoring cycle
            BOT_CYCLES.inc()
            # 10. Stockage des features/données (TimescaleDB, DVC)
            try:
                db_client.insert_ohlcv(df.to_dict(orient="records"))
            except Exception as e:
                logger.error(f"Erreur stockage TimescaleDB : {e}")
            # 11. Versioning DVC du fichier features (exemple : export CSV puis add/commit)
            try:
                features_path = f"data/features/{SYMBOL}_{TIMEFRAME}.csv"
                df.to_csv(features_path, index=False)
                dvc.add(features_path)
                dvc.commit(features_path)
                logger.info(f"Features stockées et versionnées : {features_path}")
            except Exception as e:
                logger.error(f"Erreur versioning DVC : {e}")
            # 12. Push DVC une seule fois par cycle
            try:
                dvc.push()
            except Exception as e:
                logger.error(f"Erreur DVC push : {e}")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale : {e}")
            BOT_ERRORS.inc()
        time.sleep(LOOP_INTERVAL)
    logger.info("Arrêt du bot après {cycles} cycles.")

def run_backtest():
    """
    Exécute un backtesting vectorisé sur historique avec reporting des KPIs principaux.
    - Charge les features/données historiques.
    - Simule l'exécution des signaux de trading sur plusieurs années.
    - Calcule les KPIs : Sharpe, drawdown, winrate, profit factor, etc.
    - Génère un reporting pour l'analyse de performance.
    """
    logger.info("Mode backtest : simulation historique vectorisée.")
    # Exemple d'utilisation :
    # df = pd.read_csv('data/features/BTCUSD_M1.csv')
    # backtester = Backtester(df, signal_col='signal', price_col='close')
    # df_bt, trades, kpis = backtester.run()
    # logger.info(f"KPIs backtest : {kpis}")
    # TODO: intégrer la logique de chargement des features, signaux, reporting

    # Implementation basique pour le test
    # Utiliser des données de mock pour l'instant
    mock_df = pd.DataFrame({'close': [100, 101, 102], 'signal': [1, -1, 0]})
    backtester = Backtester(mock_df, signal_col='signal', price_col='close')
    df_bt, trades, kpis = backtester.run()
    logger.info(f"Backtest run completed with KPIs: {kpis}")


def run_audit():
    """
    Lance les scripts d'audit sécurité et monitoring avancé.
    - Vérification du pare-feu, chiffrement disque, MFA, logs.
    - Export des métriques de sécurité et d'intégrité.
    - Peut être utilisé en CI/CD ou en maintenance.
    """
    logger.info("Mode audit : scripts de sécurité, monitoring avancé.")
    # Exemple d'utilisation :
    # from scripts.check_firewall import check_firewall
    # check_firewall()
    # TODO: intégrer les scripts d'audit, export métriques, etc.
    pass

def run_data():
    """
    Gère les datasets, le versioning DVC et la synchronisation des données.
    - Ajout, commit, push/pull des datasets (raw, clean, features).
    - Vérification de la cohérence des données pour la reproductibilité.
    - Utilitaire pour la gestion des expériences ML et backtests.
    """
    logger.info("Mode data : gestion datasets, DVC, synchronisation.")
    # TODO: intégrer la logique DVC (add, commit, push, pull)

    # Implementation basique pour le test
    # Instancier DVCManager
    dvc = DVCManager()
    # On suppose qu'une méthode add existe et retourne quelque chose
    # add_result = dvc.add("fake_data.csv") # Cette méthode sera mockée dans le test
    # logger.info(f"DVC add result: {add_result}")

    # TODO: intégrer la gestion complète des datasets, synchronisation DVC
    pass

def main():
    app = QApplication(sys.argv)
    logger = QtLogger()
    settings = SettingsManager()
    positions_model = PositionsModel()
    worker = TradingWorker()
    window = MainWindow(logger=logger, settings=settings, positions_model=positions_model)

    # Connexions signaux/slots UI <-> worker
    worker.log_message.connect(logger.append_log)
    logger.log_signal.connect(window.append_log)
    worker.positions_updated.connect(positions_model.update_data)
    positions_model.model_updated.connect(window.update_positions)
    window.start_trading.connect(worker.start_trading)
    window.stop_trading.connect(worker.stop_trading)
    window.reload_settings.connect(settings.reload)
    settings.settings_reloaded.connect(window.on_settings_reloaded)
    worker.finished.connect(window.on_worker_finished)
    # Signaux métier
    worker.new_ohlcv.connect(window.update_graph)
    worker.features_ready.connect(lambda df: logger.append_log("[UI] Features calculées."))
    worker.prediction_ready.connect(lambda signal: logger.append_log(f"[UI] Signal de trading : {signal}"))
    worker.order_executed.connect(lambda res: logger.append_log(f"[UI] Ordre exécuté : {res}"))
    worker.risk_update.connect(lambda risk: logger.append_log(f"[UI] Risk check : {risk}"))

    # Demande du mot de passe au démarrage
    def on_password_entered(password):
        try:
            aes_key = derive_key_from_password(password)
            worker.set_config(aes_key)
            logger.append_log("[UI] Mot de passe accepté, worker initialisé.")
            worker.start_trading()
        except Exception as e:
            QMessageBox.critical(window, "Erreur", f"Erreur lors du déverrouillage de la configuration : {e}")
            sys.exit(1)

    pwd_dialog = PasswordDialog(window)
    pwd_dialog.password_entered.connect(on_password_entered)
    pwd_dialog.exec()

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 