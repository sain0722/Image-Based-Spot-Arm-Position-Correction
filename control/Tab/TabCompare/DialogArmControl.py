from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog


class ArmControlDialog(QDialog):
    def __init__(self):
        super().__init__()

        # Main dialog setup
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setObjectName("ArmControlDialog")
        self.resize(380, 330)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")

        self.font = QtGui.QFont()
        self.font.setFamily("현대하모니 M")
        self.font.setPointSize(10)

        self.setupPositionSection()
        self.setupRotationSection()
        self.setupStowAndUnstow()
        # self.setupButtonBox()
        self.setupCommandButtons()
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi()

    def setupPositionSection(self):
        self.lbl_arm_manual_move_position = QtWidgets.QLabel(self)
        self.lbl_arm_manual_move_position.setFont(self.font)
        self.lbl_arm_manual_move_position.setObjectName("lbl_arm_manual_move_position")
        self.verticalLayout.addWidget(self.lbl_arm_manual_move_position)
        self.verticalFrame = QtWidgets.QFrame(self)
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
        self.label_164.setFont(self.font)
        self.horizontalLayout_136.addWidget(self.label_164)
        self.cmb_move_arm_axis = QtWidgets.QComboBox(self.verticalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cmb_move_arm_axis.sizePolicy().hasHeightForWidth())
        self.cmb_move_arm_axis.setSizePolicy(sizePolicy)
        self.cmb_move_arm_axis.setMinimumSize(QtCore.QSize(87, 0))
        self.cmb_move_arm_axis.setMaximumSize(QtCore.QSize(100, 16777215))
        self.cmb_move_arm_axis.setFont(self.font)
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
        self.label_165.setFont(self.font)

        self.horizontalLayout_137.addWidget(self.label_165)
        self.sbx_move_arm_rate = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.sbx_move_arm_rate.setFont(self.font)
        self.sbx_move_arm_rate.setDecimals(3)
        self.sbx_move_arm_rate.setMinimum(-0.5)
        self.sbx_move_arm_rate.setMaximum(0.5)
        self.sbx_move_arm_rate.setSingleStep(0.01)
        self.sbx_move_arm_rate.setProperty("value", 0.1)
        self.sbx_move_arm_rate.setObjectName("sbx_move_arm_rate")
        self.horizontalLayout_137.addWidget(self.sbx_move_arm_rate)
        self.label_166 = QtWidgets.QLabel(self.verticalFrame)
        self.label_166.setFont(self.font)
        self.label_166.setObjectName("label_166")
        self.horizontalLayout_137.addWidget(self.label_166)

        self.sbx_move_arm_end_time = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.sbx_move_arm_end_time.setFont(self.font)
        self.sbx_move_arm_end_time.setDecimals(1)
        self.sbx_move_arm_end_time.setMinimum(0.0)
        self.sbx_move_arm_end_time.setMaximum(10.0)
        self.sbx_move_arm_end_time.setSingleStep(0.1)
        self.sbx_move_arm_end_time.setProperty("value", 1.5)
        self.sbx_move_arm_end_time.setObjectName("sbx_move_arm_end_time")
        self.horizontalLayout_137.addWidget(self.sbx_move_arm_end_time)
        self.verticalLayout_73.addLayout(self.horizontalLayout_137)
        self.btn_move_arm_manual = QtWidgets.QPushButton(self.verticalFrame)
        self.btn_move_arm_manual.setFont(self.font)
        self.btn_move_arm_manual.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_move_arm_manual.setObjectName("btn_move_arm_manual")
        self.verticalLayout_73.addWidget(self.btn_move_arm_manual)
        self.verticalLayout_2.addLayout(self.verticalLayout_73)
        self.verticalLayout.addWidget(self.verticalFrame)

    def setupRotationSection(self):
        self.lbl_arm_manual_move_rotation = QtWidgets.QLabel(self)
        self.lbl_arm_manual_move_rotation.setFont(self.font)
        self.lbl_arm_manual_move_rotation.setObjectName("lbl_arm_manual_move_rotation")
        self.verticalLayout.addWidget(self.lbl_arm_manual_move_rotation)
        self.verticalFrame_2 = QtWidgets.QFrame(self)
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
        self.label_170.setFont(self.font)
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
        self.cmb_move_arm_axis_rot.setFont(self.font)
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
        self.label_171.setFont(self.font)

        self.horizontalLayout_143.addWidget(self.label_171)
        self.spb_move_arm_angle_rot = QtWidgets.QSpinBox(self.verticalFrame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spb_move_arm_angle_rot.sizePolicy().hasHeightForWidth())
        self.spb_move_arm_angle_rot.setFont(self.font)
        self.spb_move_arm_angle_rot.setSizePolicy(sizePolicy)
        self.spb_move_arm_angle_rot.setMaximumSize(QtCore.QSize(100, 16777215))
        self.spb_move_arm_angle_rot.setMinimum(-90)
        self.spb_move_arm_angle_rot.setMaximum(90)
        self.spb_move_arm_angle_rot.setProperty("value", 5)
        self.spb_move_arm_angle_rot.setObjectName("spb_move_arm_angle_rot")
        self.horizontalLayout_143.addWidget(self.spb_move_arm_angle_rot)
        self.verticalLayout_75.addLayout(self.horizontalLayout_143)
        self.btn_move_arm_rotation = QtWidgets.QPushButton(self.verticalFrame_2)
        self.btn_move_arm_rotation.setFont(self.font)
        self.btn_move_arm_rotation.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_move_arm_rotation.setObjectName("btn_move_arm_rotation")

        self.verticalLayout_75.addWidget(self.btn_move_arm_rotation)
        self.verticalLayout_3.addLayout(self.verticalLayout_75)
        self.verticalLayout.addWidget(self.verticalFrame_2)

    def setupStowAndUnstow(self):
        self.lbl_arm_actions = QtWidgets.QLabel(self)
        self.lbl_arm_actions.setFont(self.font)
        self.lbl_arm_actions.setObjectName("lbl_arm_actions")
        self.lbl_arm_actions.setText("■ Arm Controls")
        self.verticalLayout.addWidget(self.lbl_arm_actions)

        self.horizontalLayout_141 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_141.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_141.setObjectName("horizontalLayout_141")
        self.btn_stow = QtWidgets.QPushButton(self)
        self.btn_stow.setFont(self.font)
        self.btn_stow.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_stow.setObjectName("btn_stow")
        self.horizontalLayout_141.addWidget(self.btn_stow)
        self.btn_unstow = QtWidgets.QPushButton(self)
        self.btn_unstow.setFont(self.font)
        self.btn_unstow.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_unstow.setObjectName("btn_unstow")
        self.horizontalLayout_141.addWidget(self.btn_unstow)
        self.verticalLayout.addLayout(self.horizontalLayout_141)

        self.horizontalLayout_arm_pose = QtWidgets.QHBoxLayout()
        self.horizontalLayout_arm_pose.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_arm_pose.setObjectName("horizontalLayout_arm_pose")

        self.btn_load_arm_pose = QtWidgets.QPushButton(self)
        self.btn_load_arm_pose.setFont(self.font)
        self.btn_load_arm_pose.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_load_arm_pose.setObjectName("btn_load_arm_pose")
        self.btn_load_arm_pose.setText("Load Arm Pose")
        self.horizontalLayout_arm_pose.addWidget(self.btn_load_arm_pose)
        self.btn_save_arm_pose = QtWidgets.QPushButton(self)
        self.btn_save_arm_pose.setFont(self.font)
        self.btn_save_arm_pose.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_save_arm_pose.setObjectName("btn_save_arm_pose")
        self.btn_save_arm_pose.setText("Save Arm Pose")

        self.horizontalLayout_arm_pose.addWidget(self.btn_save_arm_pose)
        self.verticalLayout.addLayout(self.horizontalLayout_arm_pose)

    def setupCommandButtons(self):
        self.lbl_command_actions = QtWidgets.QLabel(self)
        self.lbl_command_actions.setFont(self.font)
        self.lbl_command_actions.setObjectName("lbl_command_actions")
        self.lbl_command_actions.setText("■ Actions")
        self.verticalLayout.addWidget(self.lbl_command_actions)

        self.btn_load_source = QtWidgets.QPushButton(self)
        self.btn_load_source.setText("Load (원본)")
        self.btn_load_source.setFont(self.font)
        self.btn_load_source.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_load_source.setObjectName("btn_load_source")
        self.verticalLayout.addWidget(self.btn_load_source)

        self.horizontalLayout_source = QtWidgets.QHBoxLayout()
        self.horizontalLayout_source.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_source.setObjectName("horizontalLayout_source")

        self.btn_source = QtWidgets.QPushButton(self)
        self.btn_source.setText("Capture (원본)")
        self.btn_source.setFont(self.font)
        self.btn_source.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_source.setObjectName("btn_source")
        self.horizontalLayout_source.addWidget(self.btn_source)

        self.btn_source_save = QtWidgets.QPushButton(self)
        self.btn_source_save.setText("Save (원본)")
        self.btn_source_save.setFont(self.font)
        self.btn_source_save.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_source_save.setObjectName("btn_source_save")
        self.horizontalLayout_source.addWidget(self.btn_source_save)
        self.verticalLayout.addLayout(self.horizontalLayout_source)

        self.horizontalLayout_target = QtWidgets.QHBoxLayout()
        self.horizontalLayout_target.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_target.setObjectName("horizontalLayout_target")

        self.btn_target = QtWidgets.QPushButton(self)
        self.btn_target.setText("Capture (취득)")
        self.btn_target.setFont(self.font)
        self.btn_target.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_target.setObjectName("btn_target")
        self.horizontalLayout_target.addWidget(self.btn_target)

        self.btn_target_save = QtWidgets.QPushButton(self)
        self.btn_target_save.setText("Save (취득)")
        self.btn_target_save.setFont(self.font)
        self.btn_target_save.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_target_save.setObjectName("btn_target_save")
        self.horizontalLayout_target.addWidget(self.btn_target_save)
        self.verticalLayout.addLayout(self.horizontalLayout_target)

        self.btn_corrected = QtWidgets.QPushButton(self)
        self.btn_corrected.setText("보정 위치로 이동 후 취득")
        self.btn_corrected.setFont(self.font)
        self.btn_corrected.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_corrected.setObjectName("btn_corrected")
        self.verticalLayout.addWidget(self.btn_corrected)

        self.btn_oneshot = QtWidgets.QPushButton(self)
        self.btn_oneshot.setText("취득 + 보정 한번에")
        self.btn_oneshot.setFont(self.font)
        self.btn_oneshot.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_oneshot.setObjectName("btn_oneshot")
        self.verticalLayout.addWidget(self.btn_oneshot)

    def setupButtonBox(self):
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Arm Controler"))
        self.lbl_arm_manual_move_position.setText(_translate("Dialog", "■ Arm Manual 이동 (Position)"))
        self.label_164.setText(_translate("Dialog", "Axis"))
        self.cmb_move_arm_axis.setCurrentText(_translate("Dialog", "y"))
        self.cmb_move_arm_axis.setItemText(0, _translate("Dialog", "x"))
        self.cmb_move_arm_axis.setItemText(1, _translate("Dialog", "y"))
        self.cmb_move_arm_axis.setItemText(2, _translate("Dialog", "z"))
        self.label_165.setText(_translate("Dialog", "Rate"))
        self.label_166.setText(_translate("Dialog", "End_Time"))
        self.btn_move_arm_manual.setText(_translate("Dialog", "move_arm_manual"))
        self.lbl_arm_manual_move_rotation.setText(_translate("Dialog", "■ Arm Manual 이동 (Rotation)"))
        self.label_170.setText(_translate("Dialog", "Axis"))
        self.cmb_move_arm_axis_rot.setCurrentText(_translate("Dialog", "x"))
        self.cmb_move_arm_axis_rot.setItemText(0, _translate("Dialog", "x"))
        self.cmb_move_arm_axis_rot.setItemText(1, _translate("Dialog", "y"))
        self.cmb_move_arm_axis_rot.setItemText(2, _translate("Dialog", "z"))
        self.label_171.setText(_translate("Dialog", "Angle (radian)"))
        self.btn_move_arm_rotation.setText(_translate("Dialog", "rotation"))
        self.btn_stow.setText(_translate("Dialog", "Stow"))
        self.btn_unstow.setText(_translate("Dialog", "Unstow"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = ArmControlDialog()
    ui.show()
    sys.exit(app.exec_())
