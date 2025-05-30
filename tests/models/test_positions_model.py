import pytest
from models.positions_model import PositionsModel
from PyQt6.QtCore import QCoreApplication

@pytest.fixture
def app():
    return QCoreApplication([])

def test_positions_model_update(qtbot, app):
    model = PositionsModel()
    updated = []
    def on_update(pos):
        updated.append(pos)
    model.model_updated.connect(on_update)
    positions = [
        {"id": 1, "symbol": "BTCUSD", "qty": 0.05, "price": 30100, "side": "buy"},
        {"id": 2, "symbol": "BTCUSD", "qty": 0.02, "price": 29900, "side": "sell"}
    ]
    model.update_data(positions)
    assert model.rowCount() == 2
    assert model.columnCount() == 5
    assert model.data(model.index(0, 1)) == "BTCUSD"
    assert updated and updated[0] == positions 