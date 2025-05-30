from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSignal

class PositionsModel(QAbstractTableModel):
    model_updated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.positions = []
        self.headers = ["ID", "Symbole", "Quantité", "Prix", "Sens"]

    def rowCount(self, parent=None):
        return len(self.positions)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        pos = self.positions[index.row()]
        mapping = ["id", "symbol", "qty", "price", "side"]
        return str(pos[mapping[index.column()]])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def update_data(self, positions):
        self.beginResetModel()
        self.positions = positions
        self.endResetModel()
        self.model_updated.emit(self.positions) 