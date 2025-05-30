import logging
from PyQt6.QtCore import QObject, pyqtSignal, QMutex

class QtLogger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, logfile="scalper_ui.log"):
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

    def append_log(self, message):
        self.mutex.lock()
        try:
            self.logger.info(message)
            self.log_signal.emit(message)
        finally:
            self.mutex.unlock() 