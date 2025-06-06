import json
import os
from PyQt6.QtCore import QObject, pyqtSignal

class SettingsManager(QObject):
    """
    Gère le chargement et le rechargement des paramètres de l'application depuis un fichier JSON.
    Émet un signal Qt lors du rechargement des paramètres.
    """
    settings_reloaded = pyqtSignal()

    def __init__(self, config_path: str = "scalper_ui_config.json"):
        """
        Initialise le gestionnaire de paramètres.
        Args:
            config_path (str): Chemin du fichier de configuration JSON.
        """
        super().__init__()
        self.config_path: str = config_path
        self.settings: dict = {}
        self.load()

    def load(self) -> None:
        """
        Charge les paramètres depuis le fichier JSON si présent, sinon initialise un dictionnaire vide.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.settings = json.load(f)
        else:
            self.settings = {}

    def reload(self) -> None:
        """
        Recharge les paramètres et émet le signal settings_reloaded.
        """
        self.load()
        self.settings_reloaded.emit() 