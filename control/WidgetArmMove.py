from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from control.Control import MainFunctions
from control.utils.arm_calculate_utils import calculate_new_rotation


class WidgetArmMove:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.init_signals()

    def init_signals(self):
        self.main_widget.btn_main_move_arm_manual.clicked.connect(self.move_arm_manual)
        self.main_widget.btn_main_move_arm_rotation.clicked.connect(self.move_arm_rotation_manual)
        self.main_widget.btn_main_stow.clicked.connect(self.stow)
        self.main_widget.btn_main_unstow.clicked.connect(self.unstow)

    def move_arm_manual(self):
        axis = self.main_widget.cmb_move_arm_axis.currentText()
        joint_move_rate = self.main_widget.sbx_move_arm_rate.value()
        end_time_sec = self.main_widget.sbx_move_arm_end_time.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')

        self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, axis, joint_move_rate, end_time_sec)

    def move_arm_rotation_manual(self):
        axis = self.main_widget.cmb_move_arm_axis_rot.currentText()
        angle = self.main_widget.spb_move_arm_angle_rot.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        new_rotation = calculate_new_rotation(axis, angle, body_tform_hand.rotation)

        self.main_window.robot.robot_arm_manager.trajectory_rotation_manual(body_tform_hand, new_rotation)

    def stow(self):
        if self.main_window.robot.robot is None:
            message = "로봇 연결이 필요합니다."
            msg_box = QMessageBox()
            msg_box.setWindowFlags(Qt.WindowStaysOnTopHint)
            msg_box.information(None, "알림", message, QMessageBox.Ok)
            return

        self.main_window.robot.robot_arm_manager.stow()

    def unstow(self):
        if self.main_window.robot.robot is None:
            message = "로봇 연결이 필요합니다."
            msg_box = QMessageBox()
            msg_box.setWindowFlags(Qt.WindowStaysOnTopHint)
            msg_box.information(None, "알림", message, QMessageBox.Ok)
            return

        self.main_window.robot.robot_arm_manager.unstow()
