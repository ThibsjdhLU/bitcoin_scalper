"""
PositionDelegate : QStyledItemDelegate custom pour enrichir l'affichage de la table des positions du bot Bitcoin Scalper.
Affiche une icône achat/vente, colore la ligne selon le PnL, et affiche un tooltip détaillé.
"""

from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QWidget, QToolTip
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QBrush, QColor
from PyQt6.QtCore import QModelIndex, Qt, QRect
from pathlib import Path

class PositionDelegate(QStyledItemDelegate):
    """
    Delegate custom pour la table des positions : icône achat/vente, coloration PnL, tooltips détaillés.
    """
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.icon_buy = QIcon(str(Path(__file__).resolve().parent.parent.parent.parent / "resources" / "icons" / "status_running.svg"))  # À remplacer par une icône dédiée achat
        self.icon_sell = QIcon(str(Path(__file__).resolve().parent.parent.parent.parent / "resources" / "icons" / "status_stopped.svg"))  # À remplacer par une icône dédiée vente

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        # Affichage icône dans la colonne Sens
        if index.column() == 4:
            value = index.data(Qt.ItemDataRole.DisplayRole)
            icon = self.icon_buy if value == "Achat" else self.icon_sell if value == "Vente" else None
            # Badge d'alerte si position critique
            model = index.model()
            pos = model.positions[index.row()] if hasattr(model, 'positions') and len(model.positions) > index.row() else None
            is_critical = False
            if pos:
                # Critère : perte > 5% du capital ou drawdown > 5%
                pnl_pct = pos.get('pnl_pct', None)
                drawdown = pos.get('drawdown', None)
                if (pnl_pct is not None and pnl_pct < -5) or (drawdown is not None and drawdown < -0.05):
                    is_critical = True
            if icon:
                icon.paint(painter, option.rect, Qt.AlignmentFlag.AlignCenter)
                if is_critical:
                    # Dessiner un triangle rouge en haut à droite (badge alerte)
                    painter.save()
                    triangle = [
                        option.rect.topRight() + Qt.QPoint(-12, 4),
                        option.rect.topRight() + Qt.QPoint(-4, 4),
                        option.rect.topRight() + Qt.QPoint(-8, 16)
                    ]
                    painter.setBrush(QBrush(QColor("#f44336")))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPolygon(*triangle)
                    painter.restore()
                return
        # Surlignage de la ligne si position critique (perte > 5% ou drawdown > 5%)
        model = index.model()
        if hasattr(model, 'positions') and len(model.positions) > index.row():
            pos = model.positions[index.row()]
            is_critical = False
            pnl_pct = pos.get('pnl_pct', None)
            drawdown = pos.get('drawdown', None)
            if (pnl_pct is not None and pnl_pct < -5) or (drawdown is not None and drawdown < -0.05):
                is_critical = True
            if is_critical:
                painter.save()
                painter.fillRect(option.rect, QColor("#ffcccc"))  # Rouge clair
                painter.restore()
            else:
                # Coloration de la ligne selon le PnL (si la clé 'pnl' est présente)
                pnl = pos.get('pnl', None)
                if pnl is not None:
                    if pnl > 0:
                        painter.save()
                        painter.fillRect(option.rect, Qt.GlobalColor.darkGreen)
                        painter.restore()
                    elif pnl < 0:
                        painter.save()
                        painter.fillRect(option.rect, Qt.GlobalColor.darkRed)
                        painter.restore()
        super().paint(painter, option, index)

    def helpEvent(self, event, view, option, index):
        # Tooltip détaillé sur chaque cellule
        if event.type() == event.Type.ToolTip:
            model = index.model()
            if hasattr(model, 'positions') and len(model.positions) > index.row():
                pos = model.positions[index.row()]
                tooltip = '\n'.join(f"{k}: {v}" for k, v in pos.items())
                QToolTip.showText(event.globalPos(), tooltip, view)
                return True
        return super().helpEvent(event, view, option, index) 