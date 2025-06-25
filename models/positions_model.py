"""
PositionsModel : modèle de données pour la table des positions du bot de trading Bitcoin Scalper.
Permet l'affichage enrichi (couleurs, alignement, etc.) des positions ouvertes.
"""

from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor

class PositionsModel(QAbstractTableModel):
    """
    Modèle de données pour la table des positions du bot de trading Bitcoin Scalper.
    Fournit l'affichage enrichi (couleurs, alignement, etc.) des positions ouvertes.
    """
    model_updated = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.positions = []
        self.headers = ["ID", "Symbole", "Quantité", "Prix", "Sens"]

    def rowCount(self, parent=None) -> int:
        """Retourne le nombre de lignes (positions) du modèle."""
        return len(self.positions)

    def columnCount(self, parent=None) -> int:
        """Retourne le nombre de colonnes du modèle."""
        return len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Retourne la donnée à afficher ou le style selon le rôle Qt pour la cellule donnée."""
        if not index.isValid():
            return None
        pos = self.positions[index.row()]
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
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "type":
                if value == 0:
                    return "Achat"
                elif value == 1:
                    return "Vente"
                else:
                    return str(value)
            return str(value)
        # Coloration selon le sens (Achat=vert, Vente=rouge)
        if role == Qt.ItemDataRole.ForegroundRole and key == "type":
            if value == 0:
                return QBrush(QColor("#4caf50"))
            elif value == 1:
                return QBrush(QColor("#f44336"))
        # Alignement centré pour toutes les colonnes
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        # Placeholder pour icône (à compléter)
        # if role == Qt.ItemDataRole.DecorationRole and key == "type":
        #     ...
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """Retourne l'en-tête de colonne ou de ligne selon l'orientation et le rôle Qt."""
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def update_data(self, positions: list[dict]) -> None:
        """Met à jour la liste des positions et notifie la vue."""
        self.beginResetModel()
        self.positions = positions
        self.endResetModel()
        self.model_updated.emit(self.positions) 