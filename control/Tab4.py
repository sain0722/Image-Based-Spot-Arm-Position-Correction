import json
import queue

from PyQt5.QtWidgets import QFileDialog
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b

from control.Control import MainFunctions, ThreadWorker
from control.utils.arm_calculate_utils import calculate_new_rotation
from control.utils.utils import get_position_and_rotation_from_label


class Tab4:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.buffer_list = [queue.Queue() for _ in range(8)]

        self.main_widget.btn_save_arm_status.clicked.connect(self.show_save_dialog)
        self.main_widget.btn_arm_status_thread_start.clicked.connect(self.arm_status_thread_start)
        self.status_thread = ThreadWorker()
        self.status_thread.progress.connect(self.update_status)

        self.main_widget.btn_move_arm_manual_2.clicked.connect(self.move_arm_manual)
        self.main_widget.btn_move_arm_rotation.clicked.connect(self.move_arm_rotation_manual)
        self.main_widget.btn_stow.clicked.connect(self.stow)
        self.main_widget.btn_unstow.clicked.connect(self.unstow)

        self.main_widget.btn_arm_status_load_2.clicked.connect(self.arm_status_load)
        self.main_widget.btn_manual_move_position_2.clicked.connect(self.move_body_position)
        self.main_widget.btn_manual_odom_position_2.clicked.connect(self.move_odom_position)
        self.main_widget.btn_manual_joint_move_2.clicked.connect(self.move_joint_position)

    def arm_status_thread_start(self):
        self.status_thread.start()

    def update_status(self):
        # self.main_window.robot.get_hand_position_dict()
        widget = self.main_widget
        if not self.main_window.robot.robot:
            return

        if not self.main_window.robot.get_odom_tform_hand_dict():
            return

        odom_position, odom_rotation = self.main_window.robot.get_odom_tform_hand_dict()
        self.update_odom_labels(widget, odom_position, odom_rotation)

        body_position, body_rotation = self.main_window.robot.get_hand_position_dict()
        self.update_body_labels(widget, body_position, body_rotation)

        joint_params = self.main_window.robot.get_current_joint_state()
        self.update_joint_labels(widget, joint_params)

    @staticmethod
    def update_joint_labels(widget, joint_params):
        widget.lbl_sh0_CurrentValue_2.setText(str(joint_params.get('sh0', '')))
        widget.lbl_sh1_CurrentValue_2.setText(str(joint_params.get('sh1', '')))
        widget.lbl_el0_CurrentValue_2.setText(str(joint_params.get('el0', '')))
        widget.lbl_el1_CurrentValue_2.setText(str(joint_params.get('el1', '')))
        widget.lbl_wr0_CurrentValue_2.setText(str(joint_params.get('wr0', '')))
        widget.lbl_wr1_CurrentValue_2.setText(str(joint_params.get('wr1', '')))

    @staticmethod
    def update_odom_labels(widget, odom_position, odom_rotation):
        widget.lbl_manual_odom_pos_x_2.setText(str(odom_position.get('x', '')))
        widget.lbl_manual_odom_pos_y_2.setText(str(odom_position.get('y', '')))
        widget.lbl_manual_odom_pos_z_2.setText(str(odom_position.get('z', '')))

        widget.lbl_manual_odom_rot_x_2.setText(str(odom_rotation.get('x', '')))
        widget.lbl_manual_odom_rot_y_2.setText(str(odom_rotation.get('y', '')))
        widget.lbl_manual_odom_rot_z_2.setText(str(odom_rotation.get('z', '')))
        widget.lbl_manual_odom_rot_w_2.setText(str(odom_rotation.get('w', '')))

    @staticmethod
    def update_body_labels(widget, body_position, body_rotation):
        widget.lbl_manual_pos_x_2.setText(str(body_position.get('x', '')))
        widget.lbl_manual_pos_y_2.setText(str(body_position.get('y', '')))
        widget.lbl_manual_pos_z_2.setText(str(body_position.get('z', '')))

        widget.lbl_manual_rot_x_2.setText(str(body_rotation.get('x', '')))
        widget.lbl_manual_rot_y_2.setText(str(body_rotation.get('y', '')))
        widget.lbl_manual_rot_z_2.setText(str(body_rotation.get('z', '')))
        widget.lbl_manual_rot_w_2.setText(str(body_rotation.get('w', '')))

    def show_save_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Arm Status", "", "JSON Files (*.json)", options=options)

        if file_path:
            self.save_arm_status(file_path)

    def save_arm_status(self, file_path):
        joint_params = {
            'sh0': float(self.main_widget.lbl_sh0_CurrentValue_2.text()),
            'sh1': float(self.main_widget.lbl_sh1_CurrentValue_2.text()),
            'el0': float(self.main_widget.lbl_el0_CurrentValue_2.text()),
            'el1': float(self.main_widget.lbl_el1_CurrentValue_2.text()),
            'wr0': float(self.main_widget.lbl_wr0_CurrentValue_2.text()),
            'wr1': float(self.main_widget.lbl_wr1_CurrentValue_2.text())
        }

        odom_tform_hand_position, odom_tform_hand_rotation = self.main_window.robot.get_odom_tform_hand_dict()
        body_tform_hand_position, body_tform_hand_rotation = self.main_window.robot.get_hand_position_dict()

        data = {
            'joint_params': joint_params,
            'odom_position': odom_tform_hand_position,
            'odom_rotation': odom_tform_hand_rotation,
            'body_position': body_tform_hand_position,
            'body_rotation': body_tform_hand_rotation
        }

        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        frame_name_fiducial = fiducial.apriltag_properties.frame_name_fiducial
        fiducial_tform_odom = get_a_tform_b(fiducial.transforms_snapshot, frame_name_fiducial, ODOM_FRAME_NAME)
        odom_tform_hand = self.main_window.robot.get_odom_tform_hand()
        # odom_tform_hand = math_helpers.SE3Pose.from_proto(odom_tform_hand)
        fiducial_tform_hand = fiducial_tform_odom * odom_tform_hand

        file_path += '.json'
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _joint(self):
        joint_params = self.main_window.robot.get_current_joint_state()

        if joint_params is None:
            return

        self.main_widget.lbl_sh0.setText(str(joint_params['sh0']))
        self.main_widget.lbl_sh1.setText(str(joint_params['sh1']))
        self.main_widget.lbl_el0.setText(str(joint_params['el0']))
        self.main_widget.lbl_el1.setText(str(joint_params['el1']))
        self.main_widget.lbl_wr0.setText(str(joint_params['wr0']))
        self.main_widget.lbl_wr1.setText(str(joint_params['wr1']))

    def move_arm_manual(self):
        axis = self.main_widget.cmb_move_arm_axis_2.currentText()
        joint_move_rate = self.main_widget.sbx_move_arm_rate_2.value()
        end_time_sec = self.main_widget.sbx_move_arm_end_time_2.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')

        self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, axis, joint_move_rate, end_time_sec)

    def move_arm_rotation_manual(self):
        axis = self.main_widget.cmb_move_arm_axis_rot_2.currentText()
        angle = self.main_widget.spb_move_arm_angle_rot_2.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        new_rotation = calculate_new_rotation(axis, angle, body_tform_hand.rotation)

        self.main_window.robot.robot_arm_manager.trajectory_rotation_manual(body_tform_hand, new_rotation)

    def stow(self):
        self.main_window.robot.robot_arm_manager.stow()

    def unstow(self):
        self.main_window.robot.robot_arm_manager.unstow()

    def arm_status_load(self):
        data = self.main_func.arm_json_load()
        if data:
            self.input_arm_status(data)

    def input_arm_status(self, data):
        if 'joint_params' not in data.keys():
            self.main_func.show_message_box('올바른 형식의 파일이 아닙니다.')
            return

        # 데이터 추출 및 레이블에 입력
        joint_params = data['joint_params']
        body_position = data['body_position']
        body_rotation = data['body_rotation']
        odom_position = data['odom_position']
        odom_rotation = data['odom_rotation']

        widget = self.main_widget

        # Joint Params
        widget.lbl_manual_sh0_2.setText(str(joint_params['sh0']))
        widget.lbl_manual_sh1_2.setText(str(joint_params['sh1']))
        widget.lbl_manual_el0_2.setText(str(joint_params['el0']))
        widget.lbl_manual_el1_2.setText(str(joint_params['el1']))
        widget.lbl_manual_wr0_2.setText(str(joint_params['wr0']))
        widget.lbl_manual_wr1_2.setText(str(joint_params['wr1']))

        # body tform hand
        widget.lbl_manual_2_pos_x.setText(str(body_position['x']))
        widget.lbl_manual_2_pos_y.setText(str(body_position['y']))
        widget.lbl_manual_2_pos_z.setText(str(body_position['z']))
        widget.lbl_manual_2_rot_x.setText(str(body_rotation['x']))
        widget.lbl_manual_2_rot_y.setText(str(body_rotation['y']))
        widget.lbl_manual_2_rot_z.setText(str(body_rotation['z']))
        widget.lbl_manual_2_rot_w.setText(str(body_rotation['w']))

        # odom tform hand
        widget.lbl_manual_odom_2_pos_x.setText(str(odom_position['x']))
        widget.lbl_manual_odom_2_pos_y.setText(str(odom_position['y']))
        widget.lbl_manual_odom_2_pos_z.setText(str(odom_position['z']))
        widget.lbl_manual_odom_2_rot_x.setText(str(odom_rotation['x']))
        widget.lbl_manual_odom_2_rot_y.setText(str(odom_rotation['y']))
        widget.lbl_manual_odom_2_rot_z.setText(str(odom_rotation['z']))
        widget.lbl_manual_odom_2_rot_w.setText(str(odom_rotation['w']))

    def move_joint_position(self):
        sh0 = float(self.main_widget.lbl_manual_sh0_2.text())
        sh1 = float(self.main_widget.lbl_manual_sh1_2.text())
        el0 = float(self.main_widget.lbl_manual_el0_2.text())
        el1 = float(self.main_widget.lbl_manual_el1_2.text())
        wr0 = float(self.main_widget.lbl_manual_wr0_2.text())
        wr1 = float(self.main_widget.lbl_manual_wr1_2.text())

        params = [sh0, sh1, el0, el1, wr0, wr1]
        return self.main_window.robot.robot_arm_manager.joint_move_manual(params)

    def move_body_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget, label_name="manual_2")
        frame_name = BODY_FRAME_NAME
        self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)

    def move_odom_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget, label_name="manual_odom_2")
        frame_name = ODOM_FRAME_NAME
        self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)
