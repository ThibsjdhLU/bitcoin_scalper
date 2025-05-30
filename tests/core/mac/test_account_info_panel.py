import pytest
from PyQt6.QtWidgets import QApplication
from ui.account_info_panel import AccountInfoPanel
import sys

@pytest.fixture(scope="module")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def panel(app):
    return AccountInfoPanel()

def test_default_status(panel):
    assert panel.status_label.text() == "Statut : Déconnecté"

def test_set_status_running(panel):
    panel.set_status("running")
    assert "En cours" in panel.status_label.text()
    assert "#4caf50" in panel.status_label.styleSheet()

def test_set_status_stopped(panel):
    panel.set_status("stopped")
    assert "Arrêté" in panel.status_label.text()
    assert "#f44336" in panel.status_label.styleSheet()

def test_set_balance(panel):
    panel.set_balance(12345.67)
    assert "12,345.67" in panel.balance_label.text()

def test_set_pnl_positive(panel):
    panel.set_pnl(100.5)
    assert "Profit/Perte" in panel.pnl_label.text()
    assert "#4caf50" in panel.pnl_label.text()

def test_set_pnl_negative(panel):
    panel.set_pnl(-42.42)
    assert "Profit/Perte" in panel.pnl_label.text()
    assert "#f44336" in panel.pnl_label.text()

def test_set_last_price(panel):
    panel.set_last_price(27350.99)
    assert "27,350.99" in panel.price_label.text()

def test_set_disconnected(panel):
    panel.set_status("running")
    panel.set_disconnected()
    assert panel.status_label.text() == "Statut : Déconnecté" 