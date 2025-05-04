"""
Script de préparation de la release v1.0.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

def create_directory(path: str) -> None:
    """Crée un répertoire s'il n'existe pas."""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_files(src: str, dst: str, pattern: str = "*") -> None:
    """Copie les fichiers correspondant au pattern."""
    for file in Path(src).glob(pattern):
        if file.is_file():
            shutil.copy2(file, dst)

def main():
    """Point d'entrée principal."""
    # Créer le répertoire release
    release_dir = "release/v1.0.0"
    create_directory(release_dir)
    
    # Structure des dossiers
    dirs = [
        "core",
        "strategies",
        "backtest",
        "utils",
        "config",
        "docs",
        "tests",
        "logs"
    ]
    
    for dir_name in dirs:
        create_directory(f"{release_dir}/{dir_name}")
    
    # Copier les fichiers principaux
    main_files = [
        "main.py",
        "monitor.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in main_files:
        if os.path.exists(file):
            shutil.copy2(file, release_dir)
    
    # Copier les modules
    for dir_name in dirs:
        if os.path.exists(dir_name):
            copy_files(dir_name, f"{release_dir}/{dir_name}", "*.py")
            copy_files(dir_name, f"{release_dir}/{dir_name}", "*.json")
            copy_files(dir_name, f"{release_dir}/{dir_name}", "*.md")
    
    # Copier la documentation
    shutil.copy2("release/CHANGELOG.md", release_dir)
    shutil.copy2("release/GUIDE.md", release_dir)
    
    # Créer les fichiers de configuration exemple
    config_examples = {
        "config/config.example.json": {
            "broker": {
                "mt5": {
                    "server": "AvaTrade-Demo",
                    "login": "YOUR_LOGIN",
                    "password": "YOUR_PASSWORD",
                    "symbols": ["BTCUSD", "ETHUSD"]
                }
            }
        },
        "config/risk_config.example.json": {
            "general": {
                "initial_capital": 10000.0,
                "max_drawdown": 0.15,
                "daily_loss_limit": 0.05
            }
        }
    }
    
    import json
    for file, content in config_examples.items():
        target = f"{release_dir}/{file}"
        with open(target, 'w') as f:
            json.dump(content, f, indent=4)
    
    # Créer le fichier version
    version_info = {
        "version": "1.0.0",
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "python_version": "3.11+",
        "mt5_version": "5.0.0+"
    }
    
    with open(f"{release_dir}/version.json", 'w') as f:
        json.dump(version_info, f, indent=4)
    
    print(f"Release v1.0.0 préparée dans {release_dir}")
    print("\nÉtapes suivantes :")
    print("1. Vérifier les fichiers de configuration")
    print("2. Tester l'installation")
    print("3. Sauvegarder la release")

if __name__ == "__main__":
    main() 