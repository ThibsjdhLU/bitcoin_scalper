import pytest
from utils.logger import QtLogger
from PyQt6.QtCore import QCoreApplication

@pytest.fixture
def app():
    return QCoreApplication([])

def test_logger_signal(qtbot, app):
    logger = QtLogger(logfile='test_ui.log')
    received = []
    def on_log(msg):
        received.append(msg)
    logger.log_signal.connect(on_log)
    logger.append_log("Test log message")
    qtbot.waitSignal(logger.log_signal, timeout=1000)
    assert any("Test log message" in m for m in received) 