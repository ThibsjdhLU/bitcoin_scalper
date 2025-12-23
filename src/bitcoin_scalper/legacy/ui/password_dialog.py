from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal

class PasswordDialog(QDialog):
    password_entered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Déverrouillage de la configuration sécurisée")
        self.setModal(True)
        layout = QVBoxLayout()
        label = QLabel("Veuillez entrer votre mot de passe pour déverrouiller la configuration :")
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Annuler")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addWidget(label)
        layout.addWidget(self.edit)
        layout.addLayout(btns)
        self.setLayout(layout)

    def accept(self):
        password = self.edit.text()
        if password:
            self.password_entered.emit(password)
            super().accept()

    def reject(self):
        super().reject() 