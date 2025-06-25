import pytest
from PyQt6.QtWidgets import QApplication
from ui.dashboard_simple import DashboardSimple
import sys
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem

@pytest.fixture(scope="module")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def dash(app):
    return DashboardSimple()

def test_set_status_on(dash):
    dash.set_status(True)
    assert "ON" in dash.status_label.text()
    assert dash.status_label.property("state") == "on"

def test_set_status_off(dash):
    dash.set_status(False)
    assert "OFF" in dash.status_label.text()
    assert dash.status_label.property("state") == "off"

def test_set_pnl_positive(dash):
    dash.set_pnl(100.0, 2.5)
    assert "100.00" in dash.pnl_label.text()
    assert "2.50%" in dash.pnl_label.text()
    assert "#4caf50" in dash.pnl_label.text()

def test_set_pnl_negative(dash):
    dash.set_pnl(-50.0, -1.2)
    assert "-50.00" in dash.pnl_label.text()
    assert "-1.20%" in dash.pnl_label.text()
    assert "#f44336" in dash.pnl_label.text()

def test_set_capital(dash):
    dash.set_capital(12345.67)
    assert "12,345.67" in dash.capital_label.text()

def test_set_positions(dash):
    dash.set_positions(7)
    assert "7" in dash.positions_label.text()

def test_update_graph_empty(dash):
    dash.update_graph([])
    # Chercher le TextItem dans la scène du graphique
    items = dash.graph.scene().items()
    assert any(hasattr(item, 'toPlainText') and "Aucune donnée" in item.toPlainText() for item in items)

def test_update_graph_data(dash):
    dash.update_graph([10000, 10100, 10200])
    # Vérifier qu'un PlotDataItem (courbe) est présent dans la scène
    items = dash.graph.scene().items()
    assert any(isinstance(item, PlotDataItem) for item in items)

def test_alerts(dash):
    dash.show_alert("ALERTE!")
    assert dash.alert_banner.isVisible()
    assert "ALERTE" in dash.alert_banner.text()
    dash.hide_alert()
    assert not dash.alert_banner.isVisible()

def test_stop_signal(qtbot, dash):
    triggered = []
    dash.stop_requested.connect(lambda: triggered.append(True))
    dash._on_stop()
    assert triggered

def test_global_alert_banner(qtbot):
    from ui.main_window import MainWindow
    from models.positions_model import PositionsModel
    import logging
    mw = MainWindow(logger=logging.getLogger(), settings=None, positions_model=PositionsModel())
    qtbot.addWidget(mw)
    mw.show_global_alert("ALERTE CRITIQUE !")
    assert mw.alert_banner.isVisible()
    assert "ALERTE CRITIQUE" in mw.alert_banner.text()
    mw.hide_global_alert()
    assert not mw.alert_banner.isVisible() 