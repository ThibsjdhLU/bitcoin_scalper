"""
Module pour le modèle de données des positions de trading pour l'interface PyQt.
"""
from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSignal
from typing import List, Dict, Any


class PositionsModel(QAbstractTableModel):
    """
    Modèle de données pour afficher les positions de trading dans un QTableView.
    """
    model_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.positions: List[Dict[str, Any]] = []
        self.headers = ['Ticket', 'Symbol', 'Type', 'Volume', 'Open Price', 'Current Price', 'SL', 'TP', 'Profit', 'Time']
    
    def rowCount(self, parent=None):
        """Retourne le nombre de lignes (positions)."""
        return len(self.positions)
    
    def columnCount(self, parent=None):
        """Retourne le nombre de colonnes."""
        return len(self.headers)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Retourne les données à afficher pour une cellule donnée."""
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        
        row = index.row()
        col = index.column()
        
        if row >= len(self.positions):
            return None
        
        position = self.positions[row]
        
        # Mapping des colonnes aux clés du dictionnaire
        column_keys = ['ticket', 'symbol', 'type', 'volume', 'open_price', 'current_price', 'sl', 'tp', 'profit', 'time']
        if col < len(column_keys):
            key = column_keys[col]
            return str(position.get(key, ''))
        
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """Retourne les en-têtes de colonnes."""
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if section < len(self.headers):
                return self.headers[section]
        return None
    
    def update_data(self, positions: List[Dict[str, Any]]):
        """
        Met à jour les données du modèle avec une nouvelle liste de positions.
        
        :param positions: Liste de dictionnaires représentant les positions
        """
        self.beginResetModel()
        self.positions = positions if positions else []
        self.endResetModel()
        self.model_updated.emit()
    
    def clear(self):
        """Vide toutes les positions."""
        self.beginResetModel()
        self.positions = []
        self.endResetModel()
        self.model_updated.emit()
