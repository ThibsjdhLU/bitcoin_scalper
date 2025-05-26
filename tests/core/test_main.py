import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import app.main as main_mod
import sys
from unittest import mock
import os # Import os for path checking in ML test

class DummyConfig:
    def get(self, key, default=None):
        if key == "MT5_REST_URL":
            return "http://localhost:8000"
        if key == "MT5_REST_API_KEY":
            return "testkey"
        if key == "TSDB_HOST":
            return "localhost"
        if key == "TSDB_PORT":
            return 5432
        if key == "TSDB_NAME":
            return "testdb"
        if key == "TSDB_USER":
            return "user"
        if key == "TSDB_PASSWORD":
            return "pass"
        # Mock un chemin de modèle ML existant pour le test ML
        if key == "ML_MODEL_PATH":
            return "/fake/path/model_rf.pkl"
        return default

# --- Tests pour le mode live trading ---

def test_main_live_trading_cycle():
    """
    Teste un cycle complet du mode live trading.
    Vérifie l'appel des modules d'ingestion, nettoyage, FE, risk, envoi ordre, DB, DVC.
    """
    # Mocks spécifiques pour ce test
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch('app.main.MT5RestClient') as mock_mt5_client_cls,
        patch('app.main.DataCleaner') as mock_cleaner_cls,
        patch('app.main.FeatureEngineering') as mock_fe_cls,
        patch('app.main.RiskManager') as mock_risk_cls,
        patch('app.main.TimescaleDBClient') as mock_db_client_cls,
        patch('app.main.DataIngestor') as mock_ingestor_cls,
        patch('app.main.DVCManager') as mock_dvc_cls,
        patch('app.main.MLPipeline') as mock_ml_pipe_cls,
        patch('threading.Thread') as mock_thread,
        patch("time.sleep"),
        patch("builtins.print"),
        patch.object(main_mod.logger, "info") as mock_logger_info,
        patch.object(main_mod, "LOOP_INTERVAL", 0)
    ):

        # Instances mockées
        mt5_client = mock_mt5_client_cls.return_value
        cleaner = mock_cleaner_cls.return_value
        fe = mock_fe_cls.return_value
        risk = mock_risk_cls.return_value
        db_client = mock_db_client_cls.return_value
        ingestor = mock_ingestor_cls.return_value
        dvc = mock_dvc_cls.return_value
        ml_pipe = mock_ml_pipe_cls.return_value

        # Configurer les retours des mocks
        mt5_client.get_ohlcv.return_value = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 1, "timestamp": "2024-01-01T00:00:00Z"}] * 15
        df = pd.DataFrame({"close": [100]*15, "open": [99]*15, "high": [101]*15, "low": [98]*15, "volume": [1]*15, "timestamp": ["2024-01-01T00:00:00Z"]*15, "rsi": [10]*15})
        cleaner.clean_ohlcv.return_value = df
        fe.add_indicators.return_value = df
        risk.can_open_position.return_value = {"allowed": True, "reason": "OK"}
        # Simuler l'absence de modèle ML pour forcer le fallback RSI
        mock_ml_pipe_cls.side_effect = FileNotFoundError # Empêche l'instanciation/chargement si chemin faux

        # Forcer l'arrêt après une itération
        def stop_loop(*a, **kw): raise StopIteration
        mock_logger_info.side_effect = stop_loop

        # Exécuter la fonction
        with pytest.raises(StopIteration):
            main_mod.run_live_trading()

        # Vérifier les appels
        mock_mt5_client_cls.assert_called()
        mock_cleaner_cls.assert_called()
        mock_fe_cls.assert_called()
        mock_risk_cls.assert_called_with(mt5_client)
        mock_db_client_cls.assert_called()
        mock_ingestor_cls.assert_called_with(mt5_client, db_client, symbol=main_mod.SYMBOL, timeframe=main_mod.TIMEFRAME, cleaner=cleaner)
        ingestor.start.assert_called_once()
        mock_dvc_cls.assert_called()
        # Vérifier le fallback RSI (MLPipeline ne doit pas être chargé ou appelé pour la prédiction)
        mock_ml_pipe_cls.assert_not_called()
        # Vérifier les appels dans la boucle
        mt5_client.get_ohlcv.assert_called()
        cleaner.clean_ohlcv.assert_called_with(mt5_client.get_ohlcv.return_value)
        fe.add_indicators.assert_called_with(cleaner.clean_ohlcv.return_value)
        risk.can_open_position.assert_called_with(main_mod.SYMBOL, main_mod.ORDER_VOLUME)
        mt5_client.send_order.assert_called_with(main_mod.SYMBOL, action="buy", volume=main_mod.ORDER_VOLUME)
        db_client.insert_ohlcv.assert_called_with(df.to_dict(orient="records"))
        dvc.add.assert_called_with(f"data/features/{main_mod.SYMBOL}_{main_mod.TIMEFRAME}.csv")
        dvc.commit.assert_called_with(f"data/features/{main_mod.SYMBOL}_{main_mod.TIMEFRAME}.csv")
        dvc.push.assert_called_once() # Push une seule fois par cycle dans le main actuel


def test_main_live_trading_risk_block():
    """
    Teste que le mode live ne passe pas d'ordre si le risk manager bloque.
    """
    # Mocks spécifiques pour ce test
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch('app.main.MT5RestClient') as mock_mt5_client_cls,
        patch('app.main.DataCleaner') as mock_cleaner_cls,
        patch('app.main.FeatureEngineering') as mock_fe_cls,
        patch('app.main.RiskManager') as mock_risk_cls,
        patch('app.main.TimescaleDBClient') as mock_db_client_cls,
        patch('app.main.DataIngestor') as mock_ingestor_cls,
        patch('app.main.DVCManager') as mock_dvc_cls,
        patch('app.main.MLPipeline') as mock_ml_pipe_cls,
        patch('threading.Thread') as mock_thread,
        patch("time.sleep"),
        patch("builtins.print"),
        patch.object(main_mod.logger, "info") as mock_logger_info,
        patch.object(main_mod, "LOOP_INTERVAL", 0)
    ):

        # Instances mockées
        mt5_client = mock_mt5_client_cls.return_value
        cleaner = mock_cleaner_cls.return_value
        fe = mock_fe_cls.return_value
        risk = mock_risk_cls.return_value
        db_client = mock_db_client_cls.return_value
        ingestor = mock_ingestor_cls.return_value
        dvc = mock_dvc_cls.return_value

        # Configurer les retours des mocks
        mt5_client.get_ohlcv.return_value = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 1, "timestamp": "2024-01-01T00:00:00Z"}] * 15
        df = pd.DataFrame({"close": [100]*15, "open": [99]*15, "high": [101]*15, "low": [98]*15, "volume": [1]*15, "timestamp": ["2024-01-01T00:00:00Z"]*15, "rsi": [10]*15})
        cleaner.clean_ohlcv.return_value = df
        fe.add_indicators.return_value = df
        # Configurer le risk manager pour bloquer l'ordre
        risk.can_open_position.return_value = {"allowed": False, "reason": "Drawdown max"}

        # Forcer l'arrêt après une itération
        def stop_loop(*a, **kw): raise StopIteration
        mock_logger_info.side_effect = stop_loop

        # Exécuter la fonction
        with pytest.raises(StopIteration):
            main_mod.run_live_trading()

        # Vérifier les appels
        risk.can_open_position.assert_called_with(main_mod.SYMBOL, main_mod.ORDER_VOLUME)
        mt5_client.send_order.assert_not_called() # L'ordre ne doit PAS être envoyé
        # Vérifier les autres appels similaires au cycle normal
        mock_mt5_client_cls.assert_called()
        mock_cleaner_cls.assert_called()
        mock_fe_cls.assert_called()
        mock_risk_cls.assert_called_with(mt5_client)
        mock_db_client_cls.assert_called()
        mock_ingestor_cls.assert_called_with(mt5_client, db_client, symbol=main_mod.SYMBOL, timeframe=main_mod.TIMEFRAME, cleaner=cleaner)
        ingestor.start.assert_called_once()
        mock_dvc_cls.assert_called()
        mt5_client.get_ohlcv.assert_called()
        cleaner.clean_ohlcv.assert_called_with(mt5_client.get_ohlcv.return_value)
        fe.add_indicators.assert_called_with(cleaner.clean_ohlcv.return_value)
        db_client.insert_ohlcv.assert_called_with(df.to_dict(orient="records"))
        dvc.add.assert_called_with(f"data/features/{main_mod.SYMBOL}_{main_mod.TIMEFRAME}.csv")
        dvc.commit.assert_called_with(f"data/features/{main_mod.SYMBOL}_{main_mod.TIMEFRAME}.csv")
        dvc.push.assert_called_once()


def test_main_live_trading_exception_handling():
    """
    Teste la gestion des exceptions dans la boucle live trading.
    Vérifie que les erreurs sont loguées et que la boucle continue (après sleep).
    """
    # Mocks spécifiques pour ce test
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch('app.main.MT5RestClient') as mock_mt5_client_cls,
        patch('app.main.DataCleaner') as mock_cleaner_cls,
        patch('app.main.FeatureEngineering') as mock_fe_cls,
        patch('app.main.RiskManager') as mock_risk_cls,
        patch('app.main.TimescaleDBClient') as mock_db_client_cls,
        patch('app.main.DataIngestor') as mock_ingestor_cls,
        patch('app.main.DVCManager') as mock_dvc_cls,
        patch('app.main.MLPipeline') as mock_ml_pipe_cls,
        patch('threading.Thread') as mock_thread,
        patch("time.sleep") as mock_time_sleep,
        patch("builtins.print"),
        patch.object(main_mod.logger, "info") as mock_logger_info,
        patch.object(main_mod.logger, "error") as mock_logger_error,
        patch.object(main_mod.BOT_ERRORS, "inc") as mock_errors_inc,
        patch.object(main_mod, "LOOP_INTERVAL", 0)
    ):

        # Configurer un mock pour lever une exception
        mock_mt5_client_cls.return_value.get_ohlcv.side_effect = Exception("Simulated MT5 Error")

        # Forcer l'arrêt après la première exception loguée
        def stop_on_error(*a, **kw):
            # Assurer que BOT_ERRORS.inc() est appelé avant de s'arrêter
            mock_errors_inc.assert_called_once()
            raise StopIteration
        mock_logger_error.side_effect = stop_on_error

        # Exécuter la fonction
        with pytest.raises(StopIteration):
            main_mod.run_live_trading()

        # Vérifier les appels
        mock_mt5_client_cls.return_value.get_ohlcv.assert_called()
        mock_logger_error.assert_called_with("Erreur dans la boucle principale: Simulated MT5 Error")
        mock_errors_inc.assert_called_once()
        mock_time_sleep.assert_called_with(main_mod.LOOP_INTERVAL) # Vérifie que le bot attend après l'erreur


# --- Tests pour les autres modes ---

def test_run_backtest_mode():
    """
    Teste l'exécution du mode backtest.
    Vérifie l'appel des fonctions d'audit (à implémenter dans run_audit).
    """
    # Mocks spécifiques
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch.object(main_mod, "Backtester") as mock_backtester_cls,
        patch.object(main_mod.logger, "info") as mock_logger_info
    ):

        mock_backtester_cls.return_value.run.return_value = (pd.DataFrame(), [], {"sharpe": 1.2})

        main_mod.run_backtest()

        mock_backtester_cls.assert_called()
        mock_backtester_cls.return_value.run.assert_called_once()
        mock_logger_info.assert_any_call("Mode backtest : simulation historique vectorisée.")


def test_run_ml_mode():
    """
    Teste l'exécution du mode ML training/tuning.
    Simule le chargement d'un modèle et l'appel de la méthode fit.
    """
    # Mocks spécifiques
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch.object(main_mod, "MLPipeline") as mock_ml_pipe_cls,
        patch.object(main_mod.logger, "info") as mock_logger_info,
        patch('os.path.exists', return_value=True) # Simuler l'existence du fichier modèle
    ):

        mock_ml_pipe_cls.return_value.fit.return_value = {"val_accuracy": 0.99}
        mock_ml_pipe_cls.return_value.load.return_value = None # Load ne retourne rien

        main_mod.run_ml()

        # Vérifier les appels
        mock_ml_pipe_cls.assert_called_once()
        # TODO: Tester le chargement du modèle une fois que la logique sera implémentée dans run_ml
        # mock_ml_pipe_cls.return_value.load.assert_called_once_with("/fake/path/model_rf.pkl")
        mock_ml_pipe_cls.return_value.fit.assert_called_once() # Supposons qu'on lance fit dans ce mode
        mock_logger_info.assert_any_call("Mode ML : entraînement/tuning/explicabilité.")
        # TODO: Ajouter des assertions sur le tuning, explicabilité, sauvegarde/versioning

# TODO: Ajouter un test pour le mode ML quand aucun modèle n'est trouvé (fallback)

def test_run_audit_mode():
    """
    Teste l'exécution du mode audit.
    Vérifie l'appel des fonctions d'audit (à implémenter dans run_audit).
    """
    # Mocks spécifiques
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch.object(main_mod.logger, "info") as mock_logger_info,
        patch('scripts.check_firewall.check_firewall') as mock_check_firewall,
        patch('scripts.check_filevault.check_filevault') as mock_check_filevault,
        patch('scripts.check_python_version.check_python_version') as mock_check_python_version # Mock les scripts d'audit
    ):

        main_mod.run_audit()

        # Vérifier les appels
        mock_logger_info.assert_any_call("Mode audit : scripts de sécurité, monitoring avancé.")
        # TODO: Ajouter des assertions sur les appels aux scripts d'audit une fois intégrés
        # mock_check_firewall.assert_called_once()
        # mock_check_filevault.assert_called_once()
        # mock_check_python_version.assert_called_once()


def test_run_data_mode():
    """
    Teste l'exécution du mode data.
    Vérifie l'appel des méthodes DVC.
    """
    # Mocks spécifiques
    with (
        patch('app.main.SecureConfig', return_value=DummyConfig()),
        patch.object(main_mod, "DVCManager") as mock_dvc_cls,
        patch.object(main_mod.logger, "info") as mock_logger_info
    ):

        mock_dvc_cls.return_value.add.return_value = True # Simule l'ajout DVC réussi

        main_mod.run_data()

        # Vérifier les appels
        mock_dvc_cls.assert_called_once()
        # TODO: Ajouter des assertions sur les appels DVC (add, commit, push, pull) une fois intégrés
        # mock_dvc_cls.return_value.add.assert_called_once()
        mock_logger_info.assert_any_call("Mode data : gestion datasets, DVC, synchronisation.")

# Suppression de l'ancien test paramétré et de la fixture générale
# @pytest.mark.parametrize("func", [main_mod.run_live_trading, main_mod.run_backtest, main_mod.run_ml, main_mod.run_audit, main_mod.run_data])
# def test_main_modes_no_exception(func):
#     """
#     Teste que chaque mode principal du main.py s'exécute sans lever d'exception fatale.
#     Les dépendances lourdes sont mockées pour éviter les effets de bord.
#     """
#     with mock.patch.dict(sys.modules, {
#         'bot.connectors.mt5_rest_client': mock.MagicMock(),
#         'app.core.timescaledb_client': mock.MagicMock(),
#         'app.core.dvc_manager': mock.MagicMock(),
#         'app.core.ml_pipeline': mock.MagicMock(),
#         'app.core.backtesting': mock.MagicMock(),
#         'app.core.data_ingestor': mock.MagicMock(),
#         'app.core.data_cleaner': mock.MagicMock(),
#         'app.core.feature_engineering': mock.MagicMock(),
#         'app.core.risk_management': mock.MagicMock(),
#     }):
#         try:
#             func()
#         except Exception as e:
#             pytest.fail(f"Le mode {func.__name__} a levé une exception : {e}") 