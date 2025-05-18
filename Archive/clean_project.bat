@echo off
REM === Suppression de tous les fichiers README ===

for /R %%f in (*README*) do (
    del /F /Q "%%f"
)

REM === Création d'un nouveau README.md à la racine ===

echo # Bitcoin Scalper>README.md
echo.>>README.md
echo ## Structure du projet>>README.md
echo - ^`app/^` : Dashboard Streamlit ^(`python app/main.py^`)>>README.md
echo - ^`bot/^` : Code du bot (stratégies, services, etc)^>>README.md
echo - ^`config/^` : Configuration centrale>>README.md
echo - ^`logs/^` : Logs d'exécution ^(non versionnés^)>>README.md
echo - ^`data/^` : Données de backtest ou marché ^(non versionnées^)>>README.md
echo.>>README.md
echo ## Installation>>README.md
echo.>>README.md
echo ^```bash>>README.md
echo pip install -r requirements.txt>>README.md
echo ^```>>README.md
echo.>>README.md
echo ## Lancement du dashboard>>README.md
echo.>>README.md
echo ^```bash>>README.md
echo python app/main.py>>README.md
echo ^```>>README.md
echo.>>README.md
echo ## Nettoyage du projet>>README.md
echo.>>README.md
echo ^```bash>>README.md
echo final_clean_reset.bat>>README.md
echo ^```>>README.md
echo.>>README.md
echo _(README générique à personnaliser selon ton usage)_>>README.md

echo.
echo Tous les fichiers README ont été supprimés et un nouveau README.md standard a été créé.
pause