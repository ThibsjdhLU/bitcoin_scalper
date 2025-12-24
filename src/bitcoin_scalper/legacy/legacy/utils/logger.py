import logging
from PyQt6.QtCore import QObject, pyqtSignal, QMutex

class QtLogger(QObject):
    """
    Logger Qt thread-safe, émettant un signal à chaque message loggé.
    Permet d'afficher les logs dans l'UI et de les écrire dans un fichier.
    """
    log_signal = pyqtSignal(str)

    def __init__(self, logfile: str = "scalper_ui.log"):
        """
        Initialise le logger Qt.
        Args:
            logfile (str): Chemin du fichier de log à utiliser.
        """
        super().__init__()
        self.mutex = QMutex()
        self.logger = logging.getLogger("QtLogger")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def append_log(self, message: str) -> None:
        """
        Ajoute un message au log, émet le signal log_signal et garantit la thread-safety.
        Args:
            message (str): Message à logger.
        """
        self.mutex.lock()
        try:
            self.logger.info(message)
            self.log_signal.emit(message)
        finally:
            self.mutex.unlock() 