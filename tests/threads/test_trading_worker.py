import pytest
from PyQt6.QtCore import QCoreApplication
from threads.trading_worker import TradingWorker
import time

@pytest.fixture
def app():
    return QCoreApplication([])

def test_worker_signals(qtbot, app):
    worker = TradingWorker()
    received = {"tick": False, "log": False, "pos": False, "finished": False}
    def on_tick(tick):
        received["tick"] = True
    def on_log(msg):
        received["log"] = True
    def on_pos(pos):
        received["pos"] = True
    def on_finished():
        received["finished"] = True
    worker.new_tick.connect(on_tick)
    worker.log_message.connect(on_log)
    worker.positions_updated.connect(on_pos)
    worker.finished.connect(on_finished)
    worker.start_trading()
    qtbot.wait(1200)
    worker.stop_trading()
    qtbot.wait(600)
    assert received["tick"]
    assert received["log"]
    assert received["pos"]
    assert received["finished"] 