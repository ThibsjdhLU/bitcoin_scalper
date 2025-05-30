import json
import os
from PyQt6.QtCore import QObject, pyqtSignal

class SettingsManager(QObject):
    settings_reloaded = pyqtSignal()

    def __init__(self, config_path="scalper_ui_config.json"):
        super().__init__()
        self.config_path = config_path
        self.settings = {}
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.settings = json.load(f)
        else:
            self.settings = {}

    def reload(self):
        self.load()
        self.settings_reloaded.emit() 