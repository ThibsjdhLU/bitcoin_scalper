import pytest
from utils.logger import QtLogger
from PyQt6.QtCore import QCoreApplication
import os
import tempfile
import threading
import logging

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

def test_logger_signal_levels(qtbot, app):
    logger = QtLogger(logfile='test_ui.log')
    received = []
    logger.log_signal.connect(received.append)
    messages = [
        ("[INFO] Info message", "INFO"),
        ("[ERROR] Error message", "ERROR"),
        ("[WARNING] Warning message", "WARNING"),
        ("[DEBUG] Debug message", "DEBUG"),
    ]
    for msg, _ in messages:
        logger.append_log(msg)
    qtbot.waitSignal(logger.log_signal, timeout=1000)
    assert any("Info message" in m for m in received)
    assert any("Error message" in m for m in received)
    assert any("Warning message" in m for m in received)
    assert any("Debug message" in m for m in received)

def test_logger_file_write(tmp_path, app):
    logfile = tmp_path / "test_ui.log"
    logger = QtLogger(logfile=str(logfile))
    logger.append_log("[INFO] File write test")
    with open(logfile, "r") as f:
        content = f.read()
    assert "File write test" in content

def test_logger_file_inaccessible(monkeypatch, tmp_path, app):
    # Simule un fichier de log inaccessible
    logfile = tmp_path / "forbidden.log"
    logfile.write_text("")
    os.chmod(logfile, 0o000)
    with pytest.raises(PermissionError):
        QtLogger(logfile=str(logfile)).append_log("[INFO] Should fail")
    os.chmod(logfile, 0o600)

def test_logger_thread_safety(tmp_path, app):
    logfile = tmp_path / "threaded.log"
    logger = QtLogger(logfile=str(logfile))
    def log_many():
        for i in range(100):
            logger.append_log(f"[INFO] Thread log {i}")
    threads = [threading.Thread(target=log_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(logfile, "r") as f:
        content = f.read()
    assert content.count("Thread log") >= 500

def test_logger_no_double_handler(tmp_path, app):
    logfile = tmp_path / "double.log"
    logger1 = QtLogger(logfile=str(logfile))
    logger1.append_log("[INFO] First instance")
    del logger1
    logger2 = QtLogger(logfile=str(logfile))
    logger2.append_log("[INFO] Second instance")
    with open(logfile, "r") as f:
        content = f.read()
    assert content.count("First instance") == 1
    assert content.count("Second instance") == 1 