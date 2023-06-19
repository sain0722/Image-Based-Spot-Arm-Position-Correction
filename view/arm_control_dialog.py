# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'arm_control_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 448)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl_arm_manual_move_position = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("현대하모니 L")
        font.setPointSize(10)
        self.lbl_arm_manual_move_position.setFont(font)
        self.lbl_arm_manual_move_position.setObjectName("lbl_arm_manual_move_position")
        self.verticalLayout.addWidget(self.lbl_arm_manual_move_position)
        self.verticalFrame = QtWidgets.QFrame(Dialog)
        self.verticalFrame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.verticalFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame.setLineWidth(1)
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_73 = QtWidgets.QVBoxLayout()
        self.verticalLayout_73.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_73.setSpacing(4)
        self.verticalLayout_73.setObjectName("verticalLayout_73")
        self.horizontalLayout_136 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_136.setObjectName("horizontalLayout_136")
        self.label_164 = QtWidgets.QLabel(self.verticalFrame)
        self.label_164.setObjectName("label_164")
        self.horizontalLayout_136.addWidget(self.label_164)
        self.cmb_move_arm_axis = QtWidgets.QComboBox(self.verticalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cmb_move_arm_axis.sizePolicy().hasHeightForWidth())
        self.cmb_move_arm_axis.setSizePolicy(sizePolicy)
        self.cmb_move_arm_axis.setMinimumSize(QtCore.QSize(87, 0))
        self.cmb_move_arm_axis.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("현대하모니 B")
        font.setPointSize(10)
        self.cmb_move_arm_axis.setFont(font)
        self.cmb_move_arm_axis.setStyleSheet("")
        self.cmb_move_arm_axis.setEditable(False)
        self.cmb_move_arm_axis.setObjectName("cmb_move_arm_axis")
        self.cmb_move_arm_axis.addItem("")
        self.cmb_move_arm_axis.addItem("")
        self.cmb_move_arm_axis.addItem("")
        self.horizontalLayout_136.addWidget(self.cmb_move_arm_axis)
        self.verticalLayout_73.addLayout(self.horizontalLayout_136)
        self.horizontalLayout_137 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_137.setObjectName("horizontalLayout_137")
        self.label_165 = QtWidgets.QLabel(self.verticalFrame)
        self.label_165.setObjectName("label_165")
        self.horizontalLayout_137.addWidget(self.label_165)
        self.sbx_move_arm_rate = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.sbx_move_arm_rate.setDecimals(3)
        self.sbx_move_arm_rate.setMinimum(-0.5)
        self.sbx_move_arm_rate.setMaximum(0.5)
        self.sbx_move_arm_rate.setSingleStep(0.01)
        self.sbx_move_arm_rate.setProperty("value", 0.1)
        self.sbx_move_arm_rate.setObjectName("sbx_move_arm_rate")
        self.horizontalLayout_137.addWidget(self.sbx_move_arm_rate)
        self.label_166 = QtWidgets.QLabel(self.verticalFrame)
        self.label_166.setObjectName("label_166")
        self.horizontalLayout_137.addWidget(self.label_166)
        self.sbx_move_arm_end_time = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.sbx_move_arm_end_time.setDecimals(1)
        self.sbx_move_arm_end_time.setMinimum(0.0)
        self.sbx_move_arm_end_time.setMaximum(10.0)
        self.sbx_move_arm_end_time.setSingleStep(0.1)
        self.sbx_move_arm_end_time.setProperty("value", 2.0)
        self.sbx_move_arm_end_time.setObjectName("sbx_move_arm_end_time")
        self.horizontalLayout_137.addWidget(self.sbx_move_arm_end_time)
        self.verticalLayout_73.addLayout(self.horizontalLayout_137)
        self.btn_move_arm_manual = QtWidgets.QPushButton(self.verticalFrame)
        self.btn_move_arm_manual.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_move_arm_manual.setObjectName("btn_move_arm_manual")
        self.verticalLayout_73.addWidget(self.btn_move_arm_manual)
        self.verticalLayout_2.addLayout(self.verticalLayout_73)
        self.verticalLayout.addWidget(self.verticalFrame)
        self.lbl_arm_manual_move_rotation = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("현대하모니 L")
        font.setPointSize(10)
        self.lbl_arm_manual_move_rotation.setFont(font)
        self.lbl_arm_manual_move_rotation.setObjectName("lbl_arm_manual_move_rotation")
        self.verticalLayout.addWidget(self.lbl_arm_manual_move_rotation)
        self.verticalFrame_2 = QtWidgets.QFrame(Dialog)
        self.verticalFrame_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.verticalFrame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame_2.setLineWidth(1)
        self.verticalFrame_2.setObjectName("verticalFrame_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalFrame_2)
        self.verticalLayout_3.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_75 = QtWidgets.QVBoxLayout()
        self.verticalLayout_75.setObjectName("verticalLayout_75")
        self.horizontalLayout_142 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_142.setObjectName("horizontalLayout_142")
        self.label_170 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_170.setObjectName("label_170")
        self.horizontalLayout_142.addWidget(self.label_170)
        self.cmb_move_arm_axis_rot = QtWidgets.QComboBox(self.verticalFrame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cmb_move_arm_axis_rot.sizePolicy().hasHeightForWidth())
        self.cmb_move_arm_axis_rot.setSizePolicy(sizePolicy)
        self.cmb_move_arm_axis_rot.setMinimumSize(QtCore.QSize(87, 0))
        self.cmb_move_arm_axis_rot.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("현대하모니 B")
        font.setPointSize(10)
        self.cmb_move_arm_axis_rot.setFont(font)
        self.cmb_move_arm_axis_rot.setStyleSheet("")
        self.cmb_move_arm_axis_rot.setEditable(False)
        self.cmb_move_arm_axis_rot.setObjectName("cmb_move_arm_axis_rot")
        self.cmb_move_arm_axis_rot.addItem("")
        self.cmb_move_arm_axis_rot.addItem("")
        self.cmb_move_arm_axis_rot.addItem("")
        self.horizontalLayout_142.addWidget(self.cmb_move_arm_axis_rot)
        self.verticalLayout_75.addLayout(self.horizontalLayout_142)
        self.horizontalLayout_143 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_143.setObjectName("horizontalLayout_143")
        self.label_171 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_171.setObjectName("label_171")
        self.horizontalLayout_143.addWidget(self.label_171)
        self.spb_move_arm_angle_rot = QtWidgets.QSpinBox(self.verticalFrame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spb_move_arm_angle_rot.sizePolicy().hasHeightForWidth())
        self.spb_move_arm_angle_rot.setSizePolicy(sizePolicy)
        self.spb_move_arm_angle_rot.setMaximumSize(QtCore.QSize(100, 16777215))
        self.spb_move_arm_angle_rot.setMinimum(-90)
        self.spb_move_arm_angle_rot.setMaximum(90)
        self.spb_move_arm_angle_rot.setProperty("value", 5)
        self.spb_move_arm_angle_rot.setObjectName("spb_move_arm_angle_rot")
        self.horizontalLayout_143.addWidget(self.spb_move_arm_angle_rot)

        self.verticalLayout_75.addLayout(self.horizontalLayout_143)
        self.btn_move_arm_rotation = QtWidgets.QPushButton(self.verticalFrame_2)
        self.btn_move_arm_rotation.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_move_arm_rotation.setObjectName("btn_move_arm_rotation")
        self.verticalLayout_75.addWidget(self.btn_move_arm_rotation)

        self.verticalLayout_3.addLayout(self.verticalLayout_75)
        self.verticalLayout.addWidget(self.verticalFrame_2)

        self.horizontalLayout_141 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_141.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_141.setObjectName("horizontalLayout_141")
        self.btn_stow = QtWidgets.QPushButton(Dialog)
        self.btn_stow.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_stow.setObjectName("btn_stow")
        self.horizontalLayout_141.addWidget(self.btn_stow)
        self.btn_unstow = QtWidgets.QPushButton(Dialog)
        self.btn_unstow.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_unstow.setObjectName("btn_unstow")
        self.horizontalLayout_141.addWidget(self.btn_unstow)
        self.verticalLayout.addLayout(self.horizontalLayout_141)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.cmb_move_arm_axis.setCurrentIndex(1)
        self.cmb_move_arm_axis_rot.setCurrentIndex(0)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.lbl_arm_manual_move_position.setText(_translate("Dialog", "■ Arm Manual 이동 (Position)"))
        self.label_164.setText(_translate("Dialog", "Axis"))
        self.cmb_move_arm_axis.setCurrentText(_translate("Dialog", "y"))
        self.cmb_move_arm_axis.setItemText(0, _translate("Dialog", "x"))
        self.cmb_move_arm_axis.setItemText(1, _translate("Dialog", "y"))
        self.cmb_move_arm_axis.setItemText(2, _translate("Dialog", "z"))
        self.label_165.setText(_translate("Dialog", "Rate"))
        self.label_166.setText(_translate("Dialog", "End_Time"))
        self.btn_move_arm_manual.setText(_translate("Dialog", "move_arm_manual"))
        self.btn_move_arm_manual.setShortcut(_translate("Dialog", "Ctrl+Z"))
        self.lbl_arm_manual_move_rotation.setText(_translate("Dialog", "■ Arm Manual 이동 (Rotation)"))
        self.label_170.setText(_translate("Dialog", "Axis"))
        self.cmb_move_arm_axis_rot.setCurrentText(_translate("Dialog", "x"))
        self.cmb_move_arm_axis_rot.setItemText(0, _translate("Dialog", "x"))
        self.cmb_move_arm_axis_rot.setItemText(1, _translate("Dialog", "y"))
        self.cmb_move_arm_axis_rot.setItemText(2, _translate("Dialog", "z"))
        self.label_171.setText(_translate("Dialog", "Angle (radian)"))
        self.btn_move_arm_rotation.setText(_translate("Dialog", "rotation"))
        self.btn_move_arm_rotation.setShortcut(_translate("Dialog", "Ctrl+Z"))
        self.btn_stow.setText(_translate("Dialog", "stow"))
        self.btn_stow.setShortcut(_translate("Dialog", "Ctrl+Z"))
        self.btn_unstow.setText(_translate("Dialog", "unstow"))
        self.btn_unstow.setShortcut(_translate("Dialog", "Ctrl+Z"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())