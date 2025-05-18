@echo off
REM ================================
REM     Finalise la réorganisation
REM ================================

REM -- Nettoyage du cache Python --
for /R %%d in (__pycache__) do (
    if exist "%%d" rmdir /S /Q "%%d"
)

REM -- Création des __init__.py manquants (si jamais oubliés) --
REM (optionnel, Python >=3.3 n'exige plus __init__.py partout, mais c'est plus propre)
for %%d in (
    "bot"
    "bot\core"
    "bot\connectors"
    "bot\models"
    "bot\optimization"
    "bot\risk"
    "bot\services"
    "bot\strategies"
    "bot\utils"
    "app"
) do (
    if not exist %%d\__init__.py type nul > %%d\__init__.py
)

REM -- Message d'information --
echo.
echo ===================================
echo  Projet reorganise proprement !
echo ===================================
echo.
echo 1. Relis bien les imports dans app\main.py et tout le code :
echo    from bot.services.dashboard_service import DashboardService
echo    from config.unified_config import config
echo.
echo 2. Supprime ce script ainsi que les anciens scripts de migration.
echo.
echo 3. Pense a ajuster ton requirements.txt si besoin.
echo.
echo 4. Pour packager, cree un setup.py si tu veux rendre le projet installable.
echo.
pause