import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import builtins
import logging
from types import ModuleType

# Mock des modules manquants ou coûteux AVANT l'import de main
import sys
# On mock les modules Qt pour éviter les erreurs d'affichage
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["PyQt6.QtGui"] = MagicMock()
sys.modules["pyqtgraph"] = MagicMock()

# On doit aussi mocker les imports internes si nécessaire, mais ici on veut tester main
# qui importe ces modules. On va laisser main faire ses imports, mais on patchera
# les classes instanciées dans les tests.

import bitcoin_scalper.main as main_mod

class DummyConfig:
    def get(self, key, default=None):
        return default

@pytest.fixture
def mock_dependencies():
    with patch("bitcoin_scalper.main.SecureConfig") as mock_config, \
         patch("bitcoin_scalper.main.MT5RestClient") as mock_mt5, \
         patch("bitcoin_scalper.main.DataCleaner") as mock_cleaner, \
         patch("bitcoin_scalper.main.FeatureEngineering") as mock_fe, \
         patch("bitcoin_scalper.main.RiskManager") as mock_risk, \
         patch("bitcoin_scalper.main.TimescaleDBClient") as mock_db, \
         patch("bitcoin_scalper.main.DataIngestor") as mock_ingestor, \
         patch("bitcoin_scalper.main.DVCManager") as mock_dvc, \
         patch("bitcoin_scalper.main.Backtester") as mock_backtester, \
         patch("bitcoin_scalper.main.TradingWorker") as mock_worker, \
         patch("bitcoin_scalper.main.MainWindow") as mock_window, \
         patch("bitcoin_scalper.main.QApplication") as mock_qapp, \
         patch("bitcoin_scalper.main.PasswordDialog") as mock_pwd_dialog:

        mock_config.return_value.get.return_value = "dummy_value"
        yield {
            "config": mock_config,
            "mt5": mock_mt5,
            "cleaner": mock_cleaner,
            "fe": mock_fe,
            "risk": mock_risk,
            "db": mock_db,
            "ingestor": mock_ingestor,
            "dvc": mock_dvc,
            "backtester": mock_backtester,
            "worker": mock_worker,
            "window": mock_window,
            "qapp": mock_qapp,
            "pwd_dialog": mock_pwd_dialog
        }

def test_main_initialization(mock_dependencies):
    """Test que main() initialise correctement les composants UI et Worker."""
    with patch.dict(os.environ, {"CONFIG_AES_KEY": "dummy_key"}):
        with pytest.raises(SystemExit): # main() appelle sys.exit()
             main_mod.main()

        mock_dependencies["qapp"].assert_called_once()
        mock_dependencies["window"].assert_called_once()
        mock_dependencies["worker"].assert_called_once()
        mock_dependencies["pwd_dialog"].assert_called_once()


def test_run_backtest_mode():
    """
    Teste l'exécution du mode backtest via run_backtest_offline.
    """
    with patch('bitcoin_scalper.main.Backtester') as mock_backtester_cls:
        # Configure the mock instance returned by the mocked class
        mock_backtester_instance = mock_backtester_cls.return_value
        mock_backtester_instance.run.return_value = (pd.DataFrame(), [], {"sharpe": 1.2})

        df = pd.DataFrame({"close": [1, 2, 3], "signal": [0, 1, 0]})
        main_mod.run_backtest_offline(df)

        # Assert that the Backtester class was called (instantiated)
        mock_backtester_cls.assert_called_once()
