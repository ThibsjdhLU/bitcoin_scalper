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
from bitcoin_scalper.core.ml_pipeline import MLPipeline  # TODO: à charger pour prédiction ML
from bitcoin_scalper.core.backtesting import Backtester  # TODO: à intégrer pour reporting/backtest offline
from bitcoin_scalper.core.order_algos import execute_iceberg, execute_vwap  # TODO: à intégrer pour exécution avancée
from bot.connectors.mt5_rest_client import MT5RestClient
from prometheus_client import start_http_server, Counter, Gauge
import threading
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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
    # 1. Charger la config sécurisée
    aes_key = os.environ.get("CONFIG_AES_KEY")
    if not aes_key:
        raise RuntimeError("La variable d'environnement CONFIG_AES_KEY doit être définie (clé AES-256 hex)")
    config = SecureConfig("config.enc", bytes.fromhex(aes_key))
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
    db_client = TimescaleDBClient(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
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
    try:
        if os.path.exists(ml_model_path):
            ml_pipe = MLPipeline(model_type="random_forest")  # TODO: adapter le type si besoin
            ml_pipe.load(ml_model_path)
            logger.info(f"Modèle ML chargé depuis {ml_model_path}")
        else:
            logger.info(f"Aucun modèle ML trouvé à {ml_model_path}, fallback RSI")
    except Exception as e:
        logger.error(f"Erreur chargement modèle ML : {e}")
        ml_pipe = None

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
            # 6. Prédiction ML (si modèle chargé)
            signal = None
            if ml_pipe:
                try:
                    # On suppose que le modèle prédit 1=buy, -1=sell, 0=hold
                    pred = ml_pipe.predict(df)[-1]
                    if pred == 1:
                        signal = "buy"
                    elif pred == -1:
                        signal = "sell"
                    else:
                        signal = None
                    logger.info(f"Signal ML : {signal} (prédiction brute : {pred})")
                except Exception as e:
                    logger.error(f"Erreur prédiction ML, fallback RSI : {e}")
                    ml_pipe = None  # On désactive le ML si erreur
            if not ml_pipe:
                # Fallback RSI
                last_rsi = df["rsi"].iloc[-1]
                logger.info(f"Dernier RSI: {last_rsi:.2f}")
                if last_rsi < RSI_OVERSOLD:
                    signal = "buy"
                elif last_rsi > RSI_OVERBOUGHT:
                    signal = "sell"
                else:
                    signal = None
            # 8. Gestion du risque
            if signal:
                # Validation du risque avant envoi d'ordre
                res = risk.can_open_position(SYMBOL, ORDER_VOLUME)
                if not res["allowed"]:
                    logger.info(f"Signal {signal} refusé par gestion du risque: {res['reason']}")
                    signal = None
            # 9. Exécution de l'ordre
            if signal:
                logger.info(f"Envoi ordre {signal} {ORDER_VOLUME} {SYMBOL}")
                try:
                    # TODO: Utiliser les algos avancés si besoin (ex: execute_iceberg, execute_vwap)
                    # Exemple d'appel :
                    # res = execute_iceberg(ORDER_VOLUME, max_child=0.005, price=None, send_order_fn=mt5_client.send_order, symbol=SYMBOL, action=signal)
                    res = mt5_client.send_order(SYMBOL, action=signal, volume=ORDER_VOLUME)
                    logger.info(f"Ordre envoyé: {res}")
                except Exception as e:
                    logger.error(f"Erreur envoi ordre: {e}")
            # 10. Stockage des features/données (TimescaleDB, DVC)
            try:
                # Stockage OHLCV enrichi dans TimescaleDB
                db_client.insert_ohlcv(df.to_dict(orient="records"))
                # Versioning DVC du fichier features (exemple : export CSV puis add/commit/push)
                features_path = f"data/features/{SYMBOL}_{TIMEFRAME}.csv"
                df.to_csv(features_path, index=False)
                dvc.add(features_path)
                dvc.commit(features_path)
                dvc.push()
                logger.info(f"Features stockées et versionnées : {features_path}")
            except Exception as e:
                logger.error(f"Erreur stockage/versioning features : {e}")
            # 11. Monitoring avancé (Prometheus, alertes, dashboard)
            # TODO: exporter d'autres métriques, hooks alertes
            BOT_CYCLES.inc()
            time.sleep(LOOP_INTERVAL)
        except Exception as e:
            BOT_ERRORS.inc()
            logger.error(f"Erreur dans la boucle principale: {e}")
            time.sleep(LOOP_INTERVAL)

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

def run_ml():
    """
    Entraîne, tune et explique un modèle ML/Deep Learning pour le trading.

    Ce pipeline effectue les étapes suivantes :
    
    1. Chargement des features et du label cible à partir d'un fichier CSV.
    2. Séparation temporelle des données en ensembles d'entraînement et de validation (pas de shuffle).
    3. Entraînement d'un modèle ML (RandomForest, XGBoost, DNN, LSTM, Transformer, etc.) via MLPipeline.
    4. Évaluation du modèle sur l'ensemble de validation (Accuracy, F1, Precision, Recall, AUC).
    5. Sauvegarde du modèle entraîné sur disque.
    6. Versioning du modèle avec DVC (Data Version Control).
    7. Tuning des hyperparamètres (GridSearchCV ou Optuna).
    8. Explicabilité du modèle (SHAP).

    :raises FileNotFoundError: Si le fichier de features n'est pas trouvé.
    :raises ValueError: Si la colonne de label n'est pas présente dans les données.
    :raises Exception: Pour toute erreur lors de l'entraînement, de la sauvegarde ou du versioning.
    
    :return: None. Les métriques et résultats sont logués.
    """
    logger.info("Mode ML training/tuning : pipeline complet.")
    # 1. Définir le chemin des features
    features_file = "/Users/thibaultleray-beer/bitcoin_scalper/data/features/BTCUSD_M1_features_phoenix.csv"  # TODO: Rendre configurable (config.enc)

    # 2. Charger les données
    if not os.path.exists(features_file):
        logger.error(f"Fichier de features non trouvé: {features_file}. Veuillez générer les features d'abord.")
        return
    try:
        df = pd.read_csv(features_file)
        logger.info(f"Features chargées depuis {features_file}. Aperçu:\n{df.head()}")
        logger.info(f"Colonnes disponibles: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des features: {e}")
        return

    # 3. Préparer les données (Features X et Labels y)
    label_col = 'signal'  # Le label à prédire (doit exister dans le CSV)
    if label_col not in df.columns:
        logger.error(f"Colonne de label '{label_col}' non trouvée dans les features.")
        return
    # Colonnes à exclure des features (timestamp, label, etc.)
    exclude_cols = [label_col, 'timestamp', 'date', 'time']
    features = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
    labels = df[label_col]
    logger.info(f"Séparation features/labels : X shape={features.shape}, y shape={labels.shape}")

    # 4. Séparer en train/validation (split temporel, pas de shuffle)
    split_ratio = 0.8  # 80% train, 20% val
    split_index = int(len(df) * split_ratio)
    X_train, X_val = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_val = labels.iloc[:split_index], labels.iloc[split_index:]
    logger.info(f"Split temporel : Train ({len(X_train)}), Val ({len(X_val)})")

    # 5. Instancier le pipeline ML
    model_type = "random_forest"  # Ou "xgboost", "lstm", etc.
    try:
        ml_pipe = MLPipeline(model_type=model_type, random_state=42)
        logger.info(f"Pipeline ML instancié avec modèle: {model_type}")
    except ValueError as e:
        logger.error(f"Erreur d'instanciation du pipeline ML: {e}")
        return

    # 6. Entraîner le modèle
    logger.info("Lancement de l'entraînement...")
    try:
        training_metrics = ml_pipe.fit(X_train, y_train, val_split=len(X_val)/len(df), epochs=100, batch_size=64)
        logger.info(f"Entraînement ML terminé. Métriques: {training_metrics}")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
        return

    # 7. Évaluer le modèle sur l'ensemble de validation
    try:
        val_predictions = ml_pipe.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        # Utiliser average='weighted' pour les métriques multiclasses pour gérer les déséquilibres de classes
        f1 = f1_score(y_val, val_predictions, average='weighted', zero_division=0)
        precision = precision_score(y_val, val_predictions, average='weighted', zero_division=0)
        recall = recall_score(y_val, val_predictions, average='weighted', zero_division=0)
        try:
            # Adapter roc_auc_score pour multiclasse (méthode 'ovr' one-vs-rest)
            # predict_proba doit renvoyer les probabilités pour chaque classe
            # Assurez-vous que predict_proba est bien implémenté dans MLPipeline pour les modèles non-sklearn si besoin.
            val_probabilities = ml_pipe.predict_proba(X_val)
            if val_probabilities is not None and val_probabilities.shape[1] > 2:
                 auc = roc_auc_score(y_val, val_probabilities, multi_class='ovr', average='weighted')
            elif val_probabilities is not None and val_probabilities.shape[1] == 2:
                 # Cas binaire géré pour compatibilité, bien que les données soient multiclasses ici.
                 auc = roc_auc_score(y_val, val_probabilities[:, 1])
            else:
                 auc = None # Impossible de calculer l'AUC sans probabilités ou pour un cas non géré.

        except Exception as auc_e:
            logger.error(f"Erreur lors du calcul de l'AUC: {auc_e}")
            auc = None
        logger.info(f"Performance sur validation: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc}")
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {e}")

    # 8. Sauvegarder le modèle entraîné
    model_save_path = "model_rf.pkl"  # TODO: Rendre configurable (config.enc)
    try:
        ml_pipe.save(model_save_path)
        logger.info(f"Modèle sauvegardé sous: {model_save_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
        return

    # 9. Versionner le modèle sauvegardé avec DVC
    try:
        dvc_manager = DVCManager()
        dvc_manager.add(model_save_path)
        dvc_manager.commit(model_save_path, f"Trained {model_type} model")
        dvc_manager.push()
        logger.info(f"Modèle versionné avec DVC: {model_save_path}.dvc")
    except Exception as e:
        logger.error(f"Erreur lors du versioning DVC du modèle: {e}")

    # 10. Tuning hyperparamètres (GridSearchCV/Optuna)
    try:
        param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 10]}
        logger.info("Lancement du tuning hyperparamètres (GridSearchCV)...")
        tuning_metrics = ml_pipe.tune(X_train, y_train, param_grid, cv=3, use_optuna=False)
        logger.info(f"Tuning terminé. Best params: {tuning_metrics.get('best_params')}, Best score: {tuning_metrics.get('best_score')}")
    except Exception as e:
        logger.error(f"Erreur lors du tuning hyperparamètres: {e}")

    # 11. Explicabilité (SHAP)
    try:
        logger.info("Calcul des valeurs SHAP pour l'explicabilité...")
        shap_values = ml_pipe.explain(X_val, method="shap", nsamples=100)
        logger.info(f"Valeurs SHAP calculées (shape: {np.array(shap_values).shape})")
    except Exception as e:
        logger.error(f"Erreur lors du calcul des valeurs SHAP: {e}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur universel du bot BTCUSD")
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'backtest', 'ml', 'audit', 'data'], help='Mode d\'exécution du bot')
    args = parser.parse_args()

    if args.mode == 'live':
        run_live_trading()
    elif args.mode == 'backtest':
        run_backtest()
    elif args.mode == 'ml':
        run_ml()
    elif args.mode == 'audit':
        run_audit()
    elif args.mode == 'data':
        run_data() 