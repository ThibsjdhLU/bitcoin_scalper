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
        # Mapping des colonnes vers les clés du dict MT5
        mapping = [
            ("ticket", "ID"),        # id
            ("symbol", "Symbole"),   # symbole
            ("volume", "Quantité"),  # qty
            ("price_open", "Prix"),  # prix
            ("type", "Sens")         # sens (0=achat, 1=vente)
        ]
        key, _ = mapping[index.column()]
        value = pos.get(key, "")
        # Pour la colonne Sens, afficher 'Achat' ou 'Vente'
        if key == "type":
            if value == 0:
                return "Achat"
            elif value == 1:
                return "Vente"
            else:
                return str(value)
        return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def update_data(self, positions):
        self.beginResetModel()
        self.positions = positions
        self.endResetModel()
        self.model_updated.emit(self.positions) 