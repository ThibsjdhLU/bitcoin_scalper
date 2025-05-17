@echo off
REM Script de nettoyage des fichiers/dossiers inutiles pour app.py

REM -- FICHIERS À SUPPRIMER --
del /Q check_mt5.py
del /Q conftest.py
del /Q health_check.py
del /Q main.py
del /Q pytest.ini
del /Q requirements-test.txt
del /Q setup.py
del /Q test_app.py
del /Q test_integration.py
del /Q test_performance.py
del /Q test_security.py
del /Q test_services.py
del /Q test_strategies.py
del /Q test_utils.py
del /Q update_imports.py
del /Q update_source_imports.py
del /Q __init__.py

REM -- DOSSIERS À SUPPRIMER --
rmdir /S /Q .pytest_cache
rmdir /S /Q dashboard
rmdir /S /Q docs
rmdir /S /Q exchange
rmdir /S /Q logs
rmdir /S /Q scripts
rmdir /S /Q services
rmdir /S /Q tests
rmdir /S /Q trading
rmdir /S /Q utils
rmdir /S /Q config\archive
rmdir /S /Q config\__pycache__
rmdir /S /Q src\bitcoin_scalper.egg-info
rmdir /S /Q src\__pycache__
rmdir /S /Q src\bitcoin_scalper\analysis\__pycache__
rmdir /S /Q src\bitcoin_scalper\connectors\__pycache__
rmdir /S /Q src\bitcoin_scalper\core\__pycache__
rmdir /S /Q src\bitcoin_scalper\models\__pycache__
rmdir /S /Q src\bitcoin_scalper\optimization\__pycache__
rmdir /S /Q src\bitcoin_scalper\risk\__pycache__
rmdir /S /Q src\bitcoin_scalper\services\__pycache__
rmdir /S /Q src\bitcoin_scalper\strategies\__pycache__
rmdir /S /Q src\bitcoin_scalper\utils\__pycache__
rmdir /S /Q src\bitcoin_scalper\__pycache__
rmdir /S /Q src\backtest
rmdir /S /Q src\api
rmdir /S /Q src\analysis\__pycache__
rmdir /S /Q src\backtest\__pycache__

echo Nettoyage terminé ! Il ne reste plus que le strict nécessaire pour app.py.
pause