from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog


class ExecuteButtonsDialog(QDialog):
    def __init__(self):
        super().__init__()

        # Main dialog setup
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setObjectName("ArmControlDialog")
        self.setWindowTitle("SPOT")

        self.resize(350, 100)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")

        self.font = QtGui.QFont()
        self.font.setFamily("현대하모니 M")
        self.font.setPointSize(10)

        self.btn_source = QtWidgets.QPushButton(self)
        self.btn_source.setText("Capture (원본)")
        self.btn_source.setFont(self.font)
        self.btn_source.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_source.setObjectName("btn_source")
        self.verticalLayout.addWidget(self.btn_source)

        self.btn_target = QtWidgets.QPushButton(self)
        self.btn_target.setText("Capture (취득)")
        self.btn_target.setFont(self.font)
        self.btn_target.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_target.setObjectName("btn_target")
        self.verticalLayout.addWidget(self.btn_target)

        self.btn_corrected = QtWidgets.QPushButton(self)
        self.btn_corrected.setText("보정 위치로 이동 후 취득")
        self.btn_corrected.setFont(self.font)
        self.btn_corrected.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_corrected.setObjectName("btn_corrected")
        self.verticalLayout.addWidget(self.btn_corrected)

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = ExecuteButtonsDialog()
    ui.show()
    sys.exit(app.exec_())
