import threading
from collections import deque
import queue
from functools import partial

from PyQt5 import sip
from PyQt5.QtCore import QThread, QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QDialog, QPushButton

import open3d as o3d
from bosdyn.client import frame_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_a_tform_b

from control.Control import *
from control.PointCloud import PointCloud, ICP
from control.utils.arm_calculate_utils import apply_spot_coordinate_matrix, apply_transformation_to_target
from control.utils.utils import get_qimage, get_qpixmap_grayscale, convert_to_target_pose
from control.Calculator import DepthAccumulator
np.random.seed(0)


class TabTest:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.init_signals()
        self.page5 = Page5(self.main_widget, self.main_window, self)
        self.fpfh_page = FPFHPage(self.main_widget, self.main_window)

        self.hand_status = ThreadWorker()
        self.hand_status.progress.connect(self._hand_status)

        self.arm_data = None
        self.saved_root = None

        # self.joint_value_update_thread = JointReadThread(self.main_window)
        # self.joint_value_update_thread.start()

    def init_signals(self):
        self.main_widget.cmbAlgoChange.currentIndexChanged.connect(self.onChangeIndex)

        self.main_widget.btnSavePath.clicked.connect(lambda: self.setting_save_path(self.main_widget.lblSavePath))
        self.main_widget.btn_hand_status.toggled.connect(self.hand_status_start)

        self.main_widget.btnPage1.clicked.connect(lambda: self.main_widget.Tab2_stackedWidget.setCurrentIndex(0))
        self.main_widget.btnPage2.clicked.connect(lambda: self.main_widget.Tab2_stackedWidget.setCurrentIndex(1))
        self.main_widget.btnPage3.clicked.connect(lambda: self.main_widget.Tab2_stackedWidget.setCurrentIndex(2))
        self.main_widget.btnPage4.clicked.connect(lambda: self.main_widget.Tab2_stackedWidget.setCurrentIndex(3))

        self.main_widget.btn_manual_move_position.clicked.connect(self.move_input_position)
        self.main_widget.btn_manual_odom_position.clicked.connect(self.move_odom_position)
        self.main_widget.btn_manual_joint_move.clicked.connect(self.move_joint_position)
        self.main_widget.btn_arm_status_load.clicked.connect(self.arm_status_load)

        self.main_widget.btn_manual_move_corrected_position.clicked.connect(self.move_corrected_body_position)
        self.main_widget.btn_manual_corrected_odom_position.clicked.connect(self.move_corrected_odom_position)

        # Main Function (데이터 수집 자동화)
        # self.main_widget.btn_start_data_correct.clicked.connect(self.start_data_correct)

        # Fiducial Test
        self.main_widget.btn_find_nearest_fid.clicked.connect(self.get_odom_tform_gripper)
        self.main_widget.btn_centering_nearest_fid.clicked.connect(self.centering_nearest_fiducial)
        self.main_widget.btn_arm_json_save.clicked.connect(self.arm_json_save)
        self.main_widget.btn_arm_json_load.clicked.connect(self.arm_json_load)
        self.main_widget.btn_move_arm.clicked.connect(self.move_arm)
        self.main_widget.btn_save_path.clicked.connect(lambda: self.setting_save_path(self.main_widget.lbl_save_path))
        self.main_widget.btn_capture.clicked.connect(self.capture)
        self.main_widget.btn_capture_and_save.clicked.connect(self.capture_and_save)
        self.main_widget.btn_move_arm_manual.clicked.connect(self.move_arm_manual)
        self.main_widget.btn_arm_rotation_manual.clicked.connect(self.arm_rotation_manual)

    # region event
    def onChangeIndex(self, index):
        self.main_widget.stackedWidget.setCurrentIndex(index)

    # endregion

    def hand_status_start(self, checked):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        if checked and not self.hand_status.isRunning():
            self.hand_status.set_stop_flag(False)
            self.hand_status.start()
            self.main_widget.btn_hand_status.setText("Arm Hand Position/Rotation Status \n Stop")
        else:
            self.hand_status.set_stop_flag(True)
            self.main_widget.btn_hand_status.setText("Arm Hand Position/Rotation Status \n Start")

    def setting_save_path(self, label):
        folder = self.main_window.file_dialog.getExistingDirectory(self.main_window, "Select Directory")
        label.setText(folder)
        self.saved_root = folder

    @staticmethod
    def save_depth_image(depth_image, filename, is_depth=True):
        # 파일명과 확장자 분리
        name, ext = os.path.splitext(filename)

        # 새로운 파일명 생성
        new_filename = name
        i = 1
        while os.path.exists(new_filename + ext):
            new_filename = f"{name}_{i}"
            i += 1

        new_filename += ext

        if is_depth:
            # 배열 저장
            np.save(new_filename, depth_image)
        else:
            # 이미지 저장
            cv2.imwrite(new_filename, depth_image)

    def _hand_status(self):
        position, rotation = self.main_window.robot.get_hand_position_dict()
        # in the loop
        # self.x_buffer.append(hand_position.x)
        #
        # if len(self.x_buffer) < 100:
        #     self.x_filtered = sum(self.x_buffer)/len(self.x_buffer)
        #
        # elif len(self.x_buffer) > 100:
        #     self.x_filtered = self.x_filtered + (self.x_buffer[-1] - self.x_buffer[0])/len(self.x_buffer)
        #     self.x_buffer.pop(0)

        set_position_and_rotation(self.main_widget, "status", position, rotation)

    # 파일 탐색기 대화 상자 열기
    @staticmethod
    def get_describe(depth_data):
        mean = np.mean(depth_data)
        std = np.std(depth_data)
        median = np.median(depth_data)
        p25, p50, p75 = np.percentile(depth_data, [25, 50, 75])
        _max = np.max(depth_data)
        size = np.shape(depth_data)

        # print('Mean:', mean)
        # print('Std:', std)
        # print('Median:', median)
        # print('25th Percentile:', p25)
        # print('50th Percentile:', p50)
        # print('75th Percentile:', p75)

        return mean, std, median, p25, p50, p75, _max, size

    def move_input_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget, label_name="manual")
        trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_pos_rot

        move_position(position, rotation, trajectory_function=trajectory_function)

    def move_odom_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget, label_name="manual_odom")
        trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_odometry

        move_position(position, rotation, trajectory_function=trajectory_function)

    def move_corrected_body_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget,
                                                                  label_name="manual_corrected")
        trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_pos_rot

        move_position(position, rotation, trajectory_function=trajectory_function)

    def move_corrected_odom_position(self):
        if self.main_window.robot.robot is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        position, rotation = get_position_and_rotation_from_label(widget=self.main_widget,
                                                                  label_name="manual_corrected_odom")
        trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_odometry

        move_position(position, rotation, trajectory_function=trajectory_function)

    def move_joint_position(self):
        sh0 = float(self.main_widget.lbl_manual_sh0.text())
        sh1 = float(self.main_widget.lbl_manual_sh1.text())
        el0 = float(self.main_widget.lbl_manual_el0.text())
        el1 = float(self.main_widget.lbl_manual_el1.text())
        wr0 = float(self.main_widget.lbl_manual_wr0.text())
        wr1 = float(self.main_widget.lbl_manual_wr1.text())

        params = [sh0, sh1, el0, el1, wr0, wr1]
        return self.main_window.robot.robot_arm_manager.joint_move_manual(params)

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
        odom_position = data['odom_position']
        odom_rotation = data['odom_rotation']

        widget = self.main_widget

        widget.lbl_manual_odom_pos_x.setText(str(odom_position['x']))
        widget.lbl_manual_odom_pos_y.setText(str(odom_position['y']))
        widget.lbl_manual_odom_pos_z.setText(str(odom_position['z']))

        widget.lbl_manual_odom_rot_x.setText(str(odom_rotation['x']))
        widget.lbl_manual_odom_rot_y.setText(str(odom_rotation['y']))
        widget.lbl_manual_odom_rot_z.setText(str(odom_rotation['z']))
        widget.lbl_manual_odom_rot_w.setText(str(odom_rotation['w']))

        widget.lbl_manual_sh0.setText(str(joint_params['sh0']))
        widget.lbl_manual_sh1.setText(str(joint_params['sh1']))
        widget.lbl_manual_el0.setText(str(joint_params['el0']))
        widget.lbl_manual_el1.setText(str(joint_params['el1']))
        widget.lbl_manual_wr0.setText(str(joint_params['wr0']))
        widget.lbl_manual_wr1.setText(str(joint_params['wr1']))

        # Source Odometry Input
        widget.lbl_odom_pos_x_src.setText(str(odom_position['x']))
        widget.lbl_odom_pos_y_src.setText(str(odom_position['y']))
        widget.lbl_odom_pos_z_src.setText(str(odom_position['z']))

        widget.lbl_odom_rot_x_src.setText(str(odom_rotation['x']))
        widget.lbl_odom_rot_y_src.setText(str(odom_rotation['y']))
        widget.lbl_odom_rot_z_src.setText(str(odom_rotation['z']))
        widget.lbl_odom_rot_w_src.setText(str(odom_rotation['w']))

        if 'body_position' in data.keys():
            body_position = data['body_position']
            body_rotation = data['body_rotation']

            # Source Body Input
            widget.lbl_pos_x_src.setText(str(body_position['x']))
            widget.lbl_pos_y_src.setText(str(body_position['y']))
            widget.lbl_pos_z_src.setText(str(body_position['z']))

            widget.lbl_rot_x_src.setText(str(body_rotation['x']))
            widget.lbl_rot_y_src.setText(str(body_rotation['y']))
            widget.lbl_rot_z_src.setText(str(body_rotation['z']))
            widget.lbl_rot_w_src.setText(str(body_rotation['w']))

    # Fiducial & Centering
    def find_nearest_fiducial(self):
        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        self.main_func.show_message_box(str(fiducial))

    def centering_nearest_fiducial(self):
        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        dist_margin = self.main_widget.sbx_dist_margin.value()
        self.main_window.robot.robot_fiducial_manager.centering_on_fiducial(dist_margin=dist_margin)

    def get_odom_tform_gripper(self):
        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        frame_name_fiducial = fiducial.apriltag_properties.frame_name_fiducial
        frame_name_fiducial_filtered = fiducial.apriltag_properties.frame_name_fiducial_filtered

        odom_tform_fiducial = get_a_tform_b(fiducial.transforms_snapshot, ODOM_FRAME_NAME, frame_name_fiducial)
        fiducial_tform_odom = get_a_tform_b(fiducial.transforms_snapshot, frame_name_fiducial, ODOM_FRAME_NAME)

        odom_tform_fiducial_filtered = get_a_tform_b(fiducial.transforms_snapshot, ODOM_FRAME_NAME, frame_name_fiducial_filtered)
        fiducial_filtered_tform_odom = get_a_tform_b(fiducial.transforms_snapshot, frame_name_fiducial_filtered, ODOM_FRAME_NAME)

        odom_tform_hand = self.main_window.robot.get_odom_tform_hand()

        fiducial_tform_gripper = fiducial_tform_odom * odom_tform_hand
        fiducial_filtered_tform_gripper = fiducial_filtered_tform_odom * odom_tform_hand

        position, rotation = se3pose_to_dict(fiducial_tform_gripper)
        fiducial_tform_hand = get_arm_position_dict(frame_name_fiducial, {"position": position,
                                                                          "rotation": rotation})

        position, rotation = se3pose_to_dict(fiducial_filtered_tform_gripper)
        fiducial_filtered_tform_hand = get_arm_position_dict(frame_name_fiducial_filtered, {"position": position,
                                                                                            "rotation": rotation})

        position, rotation = self.main_window.robot.get_hand_position_dict()
        body_tform_hand = get_arm_position_dict('body', {'position': position,
                                                         'rotation': rotation})

        data = [fiducial_tform_hand, fiducial_filtered_tform_hand, body_tform_hand]
        fiducial_tag_id = fiducial.apriltag_properties.tag_id
        dist_margin = round(self.main_widget.sbx_dist_margin.value(), 3)

        json_format = create_json_format(fid_id=fiducial_tag_id, dist_margin=dist_margin, data=data)
        self.arm_data = json_format
        return json_format

    def arm_json_save(self):
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Arm Status", "", "JSON Files (*.json)")
        if file_path:
            json_data = self.get_odom_tform_gripper()
            create_json_file(file_path, json_data)

    def arm_json_load(self):
        data = self.main_func.arm_json_load()
        if data:
            if 'frame_tform_gripper' not in data.keys():
                self.main_func.show_message_box('올바른 형식의 파일이 아닙니다.')
                return
            self.arm_data = data

    def move_arm(self):
        if self.arm_data is None:
            self.main_func.show_message_box('Arm 위치 설정이 되어있지 않습니다.')
            return

        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        odom_tform_fiducial_filtered = frame_helpers.get_a_tform_b(fiducial.transforms_snapshot,
                                                                   frame_helpers.ODOM_FRAME_NAME,
                                                                   fiducial.apriltag_properties.frame_name_fiducial_filtered)
        fiducial_tform_gripper = self.arm_data['frame_tform_gripper'][1]['transform']
        fiducial_tform_gripper = dict_to_se3pose(fiducial_tform_gripper)
        odom_tform_gripper_goal = odom_tform_fiducial_filtered * fiducial_tform_gripper

        end_seconds = self.main_widget.sbx_joint_move_end_time.value()

        self.main_window.robot.robot_arm_manager.move_to_frame_hand(odom_tform_gripper_goal,
                                                                    frame_helpers.ODOM_FRAME_NAME,
                                                                    end_seconds=end_seconds)

    def move_arm_manual(self):
        axis = self.main_widget.cmb_manual_arm_axis.currentText()
        # direction = self.main_widget.cmb_manual_arm_direction.currentText()
        joint_move_rate = self.main_widget.sbx_manual_move_rate.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        end_time_sec = self.main_widget.sbx_manual_move_end_time.value()
        self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, axis, joint_move_rate, end_time_sec)

    def arm_rotation_manual(self):
        axis = self.main_widget.cmb_rot_axis_manual.currentText()
        angle = self.main_widget.spb_rot_angle_manual.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        new_rotation = calculate_new_rotation(axis, angle, body_tform_hand.rotation)

        self.main_window.robot.robot_arm_manager.trajectory_rotation_manual(body_tform_hand, new_rotation)

    def capture(self):
        image, _ = self.main_window.robot.robot_camera_manager.take_image()
        color_qimage = get_qimage(image)
        self.main_widget.lbl_color_image.setPixmap(QPixmap.fromImage(color_qimage))

        return image

    def capture_and_save(self):
        image = self.capture()
        # 이미지 저장
        file_path = self.main_widget.lbl_save_path.text()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        file_name = f"captured_image_{current_time}.jpg"
        cv2.imwrite(os.path.join(file_path, file_name), image)


class HandStatusThread(QThread):
    function = pyqtSignal()

    def run(self):
        duration = 2.0
        start_time = time.time()
        end_time = start_time + duration + 1.0
        while time.time() < end_time:  # 스레드가 종료될 때까지 대기
            self.function.emit()


class Page5:
    def __init__(self, main_widget, main_window, tab2_instance):
        self.main_widget = main_widget
        self.main_window = main_window
        self.main_func = MainFunctions(self.main_window)
        self.tab2 = tab2_instance

        # 변수 설정
        self.odom_tform_hand_source = None
        self.odom_tform_hand_target = None

        self.source_data = dict()
        self.target_data = dict()

        self.hand_color_image_source = None
        self.hand_depth_image_source = None
        self.depth_image_source = None
        self.hand_color_in_depth_frame_source = None
        self.depth_data_uint8_source = None
        self.depth_data_source = None

        self.hand_color_image_target = None
        self.hand_depth_image_target = None
        self.depth_data_uint8_target = None
        self.depth_data_target = None
        self.depth_image_target = None
        self.hand_color_in_depth_frame_target = None

        self.hand_rotation_ref = None
        self.hand_rotation_new = None

        self.pos_x_ref = None
        self.pos_y_ref = None
        self.pos_z_ref = None
        self.translation = None

        self.source_pcd = None
        self.target_pcd = None

        self.pos_x_new = None
        self.pos_y_new = None
        self.pos_z_new = None

        self.pos_x_result = None
        self.pos_y_result = None
        self.pos_z_result = None

        self.arm_position_input_source = dict()
        self.arm_position_input_target = dict()

        self.source_se3pose = None
        self.target_se3pose = None

        self.icp_transformation = None

        self.traj_thread = None

        self.iqr = None

        # Thread Stop Flag
        self.thread_stop_flag = False

        # Target 데이터 Buffer
        self.target_data_buffer = deque()

        # ICP 클래스 변수
        self.icp = None

        # 선택된 Target
        self.selected_target_idx = 0

        # Source / Target 병합 이미치
        self.merged_image = None

        self.data_accumulator_source = DepthAccumulator(buffer_size=100)
        self.data_accumulator_target = DepthAccumulator(buffer_size=100)

        self.init_signals()

    def init_signals(self):
        # 이벤트 설정
        self.main_widget.btn_page5_capture_ref.clicked.connect(self.set_images_source)
        self.main_widget.btn_page5_capture_new.clicked.connect(self.set_images_target)

        self.main_widget.btn_capture_second_ref.clicked.connect(partial(self.capture_in_second, 'source'))
        self.main_widget.btn_capture_second_new.clicked.connect(partial(self.capture_in_second, 'target'))

        self.main_widget.btn_page5_view_pcd_ref.clicked.connect(self.view_pcd_source)
        self.main_widget.btn_page5_view_pcd_new.clicked.connect(self.view_pcd_target)

        self.main_widget.btn_page5_pcd_clear_ref.clicked.connect(self.clear_pcd_source)
        self.main_widget.btn_page5_pcd_clear_new.clicked.connect(self.clear_pcd_target)

        self.main_widget.btn_page5_load_depth_ref.clicked.connect(self.load_depth_acm_src)
        self.main_widget.btn_page5_load_depth_new.clicked.connect(self.load_depth_new)

        self.main_widget.btn_page5_save_data_ref.clicked.connect(self.save_data_ref)
        self.main_widget.btn_page5_save_data_new.clicked.connect(self.save_data_new)

        self.main_widget.btn_page5_init_transform.clicked.connect(self.init_transform)
        self.main_widget.btn_page5_surf.clicked.connect(self.surf)
        self.main_widget.btn_page5_icp.clicked.connect(self.execute_icp)
        self.main_widget.btn_page5_icp_result.clicked.connect(self.view_icp_matrix)
        self.main_widget.btn_page5_transform_icp.clicked.connect(self.transform_icp_result)

        # Page 3

        self.main_widget.cbx_odometry.stateChanged.connect(self.checkbox_state_changed)
        self.main_widget.rbn_translation.toggled.connect(self.radio_button_state_changed)
        self.main_widget.rbn_rotation.toggled.connect(self.radio_button_state_changed)
        self.radio_button_state_changed()

        self.main_widget.btn_start.toggled.connect(self.toggle_thread)
        self.main_widget.btn_tform_icp.clicked.connect(self.viz_tform_icp)
        self.main_widget.btn_tform_icp_feature.clicked.connect(self.viz_tform_icp_feature)
        self.main_widget.btn_tform_maxrix_dialog.clicked.connect(self.tform_matrix_dialog)

        self.main_widget.btn_move_corrected_pos.clicked.connect(self.move_corrected_pos)
        self.main_widget.btn_move_target_pos.clicked.connect(self.move_target_pos)
        self.main_widget.btn_move_source_pos.clicked.connect(self.move_source_pos)
        self.main_widget.btn_capture_corrected.clicked.connect(self.capture_corrected)

        self.main_widget.btn_save_merged_image.clicked.connect(self.save_merged_image)

    def capture_page5(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        iqr1 = int(self.main_widget.iqr1LineEdit.text())
        iqr3 = int(self.main_widget.iqr3LineEdit.text())
        self.iqr = [iqr1, iqr3]

        depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
            iqr=self.iqr,
            outlier_removal=self.main_widget.cbxOutlierRemove.isChecked()
        )

        depth_color = self.main_window.robot.robot_camera_manager.depth_to_color(depth_image)

        color = self.main_window.robot.robot_camera_manager.take_image_from_source('hand_color_in_hand_depth_frame')
        color_in_depth_frame = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)

        hand_color, data = self.main_window.robot.robot_camera_manager.take_image()

        return color_in_depth_frame, depth_color, depth_image, hand_color

    def capture_depth(self):
        # iqr1 = int(self.main_widget.iqr1LineEdit.text())
        # iqr3 = int(self.main_widget.iqr3LineEdit.text())
        # self.iqr = [iqr1, iqr3]
        self.iqr = [30, 70]

        depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
            iqr=self.iqr,
            outlier_removal=self.main_widget.cbxOutlierRemove.isChecked()
        )
        return depth_image

    def set_images_source(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()
        color_qimage = get_qimage(color_in_depth_frame)
        depth_color_qimage = get_qimage(depth_color)

        self.main_widget.lbl_page5_color_ref.setPixmap(QPixmap.fromImage(color_qimage))
        self.main_widget.lbl_page5_depth_ref.setPixmap(QPixmap.fromImage(depth_color_qimage))

        self.data_accumulator_source.add_data(depth_data)

        self.hand_color_image_source = hand_color
        self.hand_depth_image_source = depth_color
        self.depth_image_source = self.data_accumulator_source.get_filtered_data(is_remove_outlier=False)
        depth_data_uint8 = cv2.convertScaleAbs(self.depth_image_source, alpha=(255.0 / self.depth_image_source.max()))
        self.depth_data_source = depth_data
        self.depth_data_uint8_source = depth_data_uint8
        self.hand_color_in_depth_frame_source = color_in_depth_frame
        self.main_widget.lbl_ref_cum_cnt.setText(str(self.data_accumulator_source.n_accumulate))

        mean, std, median, p25, p50, p75, _max, size = self.tab2.get_describe(self.depth_image_source)
        self.main_widget.lbl_ref_median.setText(str(median))
        self.source_data['depth_median'] = median

    def set_images_target(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        # 촬영 시점의 로봇 좌표계 획득
        self.target_se3pose = self.main_window.robot.get_current_hand_position('hand')

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()
        color_qimage = get_qimage(color_in_depth_frame)
        depth_color_qimage = get_qimage(depth_color)

        self.data_accumulator_target.add_data(depth_data)

        self.main_widget.lbl_page5_color_new.setPixmap(QPixmap.fromImage(color_qimage))
        self.main_widget.lbl_page5_depth_new.setPixmap(QPixmap.fromImage(depth_color_qimage))

        self.hand_color_in_depth_frame_target = color_in_depth_frame
        self.depth_image_target = self.data_accumulator_target.get_filtered_data(is_remove_outlier=False)
        depth_data_uint8 = cv2.convertScaleAbs(self.depth_image_target, alpha=(255.0 / self.depth_image_target.max()))
        self.depth_data_uint8_target = depth_data_uint8
        self.depth_data_target = depth_data

        self.hand_depth_image_target = depth_color
        self.hand_color_image_target = hand_color

        self.main_widget.lbl_new_cum_cnt.setText(str(self.data_accumulator_target.n_accumulate))

        mean, std, median, p25, p50, p75, _max, size = self.tab2.get_describe(self.depth_image_target)
        self.main_widget.lbl_new_median.setText(str(median))

        # 임시 코드
        # save_file_path = self.main_widget.lblSavePath.text()
        # save_file_name = "depth_data.png"
        # save_file_npy = "depth_data.npy"
        #
        # unique_file_name = get_unique_filename(save_file_path, save_file_name)
        # unique_file_npy = get_unique_filename(save_file_path, save_file_npy)
        #
        # cv2.imwrite(os.path.join(save_file_path, unique_file_name), self.depth_data_target)
        # np.save(os.path.join(save_file_path, unique_file_npy), self.depth_data_target)

    def set_odom_position_data(self, odom_tform_hand_pose):
        position = {axis: getattr(odom_tform_hand_pose.position, axis) for axis in ['x', 'y', 'z']}
        rotation = {axis: getattr(odom_tform_hand_pose.rotation, axis) for axis in ['x', 'y', 'z', 'w']}

        set_position_and_rotation(self.main_widget, "manual_odom", position, rotation)

    def capture_in_second(self, mode):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        st = time.time()
        is_acm_save = False

        while time.time() <= st + 1:
            depth_data = self.capture_depth()
            position, rotation = self.main_window.robot.get_hand_position_dict()
            if mode == 'source':
                self.data_accumulator_source.add_data(depth_data)

                # 촬영 시점의 로봇 좌표계 획득
                self.source_se3pose = self.main_window.robot.get_current_hand_position('hand')
                self.arm_position_input_source = {
                    "position": {axis: position[axis] for axis in ['x', 'y', 'z']},
                    "rotation": {axis: rotation[axis] for axis in ['w', 'x', 'y', 'z']}
                }

                self.odom_tform_hand_source = self.main_window.robot.get_odom_tform_hand()
                self.set_odom_position_data(self.odom_tform_hand_source)

                if is_acm_save:
                    color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()
                    self.setting_capture_data(color_in_depth_frame, depth_color, hand_color, depth_data, mode=mode)

                    self.set_source_data()
                    self.save_source_data_acm()
            elif mode == 'target':
                self.data_accumulator_target.add_data(depth_data)

                # 촬영 시점의 로봇 좌표계 획득
                self.target_se3pose = self.main_window.robot.get_current_hand_position('hand')
                self.arm_position_input_target = {
                    "position": {axis: position[axis] for axis in ['x', 'y', 'z']},
                    "rotation": {axis: rotation[axis] for axis in ['w', 'x', 'y', 'z']}
                }
                self.odom_tform_hand_target = self.main_window.robot.get_odom_tform_hand()
                if is_acm_save:
                    color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()
                    self.setting_capture_data(color_in_depth_frame, depth_color, hand_color, depth_data, mode=mode)

                    self.set_target_data()
                    self.save_target_data_acm()

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()
        self.setting_capture_data(color_in_depth_frame, depth_color, hand_color, depth_data, mode=mode)

    def setting_capture_data(self, color_in_depth_frame, depth_color, hand_color, depth_data, mode):
        color_qimage = get_qimage(color_in_depth_frame)
        depth_color_qimage = get_qimage(depth_color)

        if mode == 'source':
            self.main_widget.lbl_page5_color_ref.setPixmap(QPixmap.fromImage(color_qimage))
            self.main_widget.lbl_page5_depth_ref.setPixmap(QPixmap.fromImage(depth_color_qimage))

            self.hand_color_image_source = hand_color
            self.hand_depth_image_source = depth_color
            self.depth_image_source = self.data_accumulator_source.get_filtered_data(is_remove_outlier=False)
            depth_data_uint8 = cv2.convertScaleAbs(self.depth_image_source,
                                                   alpha=(255.0 / self.depth_image_source.max()))
            self.depth_data_uint8_source = depth_data_uint8
            self.depth_data_source = depth_data
            self.hand_color_in_depth_frame_source = color_in_depth_frame

            self.main_widget.lbl_ref_cum_cnt.setText(str(self.data_accumulator_source.n_accumulate))

            mean, std, median, p25, p50, p75, _max, size = self.tab2.get_describe(self.depth_image_source)
            self.main_widget.lbl_ref_median.setText(str(median))
            self.source_data['depth_median'] = median

            # Arm position data
            self.arm_position_input_source = {
                "position": {
                    "x": float(self.main_widget.lbl_pos_x_src.text()),
                    "y": float(self.main_widget.lbl_pos_y_src.text()),
                    "z": float(self.main_widget.lbl_pos_z_src.text())
                },
                "rotation": {
                    "w": float(self.main_widget.lbl_rot_w_src.text()),
                    "x": float(self.main_widget.lbl_rot_x_src.text()),
                    "y": float(self.main_widget.lbl_rot_y_src.text()),
                    "z": float(self.main_widget.lbl_rot_z_src.text())
                }
            }
            self.source_data["arm_position_input"] = self.arm_position_input_source

        else:
            self.main_widget.lbl_page5_color_new.setPixmap(QPixmap.fromImage(color_qimage))
            self.main_widget.lbl_page5_depth_new.setPixmap(QPixmap.fromImage(depth_color_qimage))

            self.hand_color_image_target = hand_color
            self.hand_depth_image_target = depth_color
            self.depth_image_target = self.data_accumulator_target.get_filtered_data(is_remove_outlier=False)
            depth_data_uint8 = cv2.convertScaleAbs(self.depth_image_target,
                                                   alpha=(255.0 / self.depth_image_target.max()))
            self.depth_data_uint8_target = depth_data_uint8
            self.depth_data_target = depth_data
            self.hand_color_in_depth_frame_target = color_in_depth_frame

            self.main_widget.lbl_new_cum_cnt.setText(str(self.data_accumulator_target.n_accumulate))

            mean, std, median, p25, p50, p75, _max, size = self.tab2.get_describe(self.depth_image_target)
            self.main_widget.lbl_new_median.setText(str(median))
            self.target_data['depth_median'] = median

    def save_color_ref_page5(self):
        if self.hand_color_image_source is None:
            self.main_func.show_message_box('촬영된 이미지가 없습니다.')
            return

        cv2.imwrite('hand_color_ref.jpg', self.hand_color_image_source)
        cv2.imwrite('depth_color_ref.jpg', self.hand_depth_image_source)
        cv2.imwrite('depth_ref.jpg', self.depth_image_source)

    def save_hand_color_in_depth_frame_target(self):
        if self.hand_color_image_target is None:
            self.main_func.show_message_box('촬영된 이미지가 없습니다.')
            return

        cv2.imwrite('hand_color_new.jpg', self.hand_color_image_target)
        cv2.imwrite('depth_color_new.jpg', self.hand_color_in_depth_frame_target)
        cv2.imwrite('depth_new.jpg', self.depth_image_target)

    def set_source_pcd(self, depth):
        pcd = PointCloud(depth)
        filtered_pcd = pcd.apply_sor_filter()
        self.source_pcd = filtered_pcd
        self.source_pcd.estimate_normals()

    def set_target_pcd(self, depth):
        pcd = PointCloud(depth)
        filtered_pcd = pcd.apply_sor_filter()
        self.target_pcd = filtered_pcd
        self.target_pcd.estimate_normals()

    def view_pcd_source(self):
        threshold = float(self.main_widget.lbl_outlier_threshold.text())
        cum_data = self.data_accumulator_source.get_filtered_data(is_remove_outlier=True, threshold=threshold)
        ref_pcd = PointCloud(cum_data)
        # o3d.visualization.draw_geometries([ref_pcd.pcd],
        #                                   width=1440, height=968,
        #                                   left=50, top=50,
        #                                   front=[0, 0, -1],
        #                                   lookat=[0, 0, 0.8],
        #                                   up=[0.0, -1, 0.1],
        #                                   zoom=0.64)
        filtered_pcd = ref_pcd.apply_sor_filter()
        o3d.visualization.draw_geometries([filtered_pcd],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0.013, -0.081, -0.996],
                                          lookat=[0, 0, 0.8],
                                          up=[-0.01, -1, 0.08],
                                          zoom=0.3)

        # ref_cl, ref_ind = ref_pcd.pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
        # ref_inlier_cloud = ref_pcd.pcd.select_by_index(ref_ind)
        # self.source_pcd = ref_inlier_cloud
        self.source_pcd = filtered_pcd
        self.source_pcd.estimate_normals()

    def view_pcd_target(self):
        threshold = float(self.main_widget.lbl_outlier_threshold.text())
        cum_data = self.data_accumulator_target.get_filtered_data(is_remove_outlier=True, threshold=threshold)
        new_pcd = PointCloud(cum_data)

        # o3d.visualization.draw_geometries([new_pcd.pcd],
        #                                   width=1440, height=968,
        #                                   left=50, top=50,
        #                                   front=[0, 0, -1],
        #                                   lookat=[0, 0, 0.8],
        #                                   up=[0.0, -1, 0.1],
        #                                   zoom=0.64)
        filtered_pcd = new_pcd.apply_sor_filter()
        o3d.visualization.draw_geometries([filtered_pcd],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0.013, -0.081, -0.996],
                                          lookat=[0, 0, 0.8],
                                          up=[-0.01, -1, 0.08],
                                          zoom=0.3)

        # new_cl, new_ind = new_pcd.pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.2)
        # new_inlier_cloud = new_pcd.pcd.select_by_index(new_ind)
        # self.target_pcd = new_inlier_cloud

        self.target_pcd = filtered_pcd
        self.target_pcd.estimate_normals()

    def clear_pcd_source(self):
        self.data_accumulator_source.clear()
        self.data_accumulator_source = DepthAccumulator(buffer_size=10)
        self.main_widget.lbl_ref_cum_cnt.setText(str(self.data_accumulator_source.n_accumulate))

    def clear_pcd_target(self):
        self.data_accumulator_target.clear()
        self.data_accumulator_target = DepthAccumulator(buffer_size=10)
        self.main_widget.lbl_new_cum_cnt.setText(str(self.data_accumulator_target.n_accumulate))

    def load_depth(self):
        file, _ = QFileDialog.getOpenFileName(None, "열기", "", "Numpy 파일 (*.npy);;PNG 파일 (*.png)")

        return file

    def load_depth_acm_src(self):
        files, _ = QFileDialog.getOpenFileNames(None, "파일 선택", "", "PLY 파일 (*.ply);;Numpy 파일 (*.npy);;PNG 파일 (*.png)")

        if files:
            depth = None
            for file in files:
                if file.endswith('.ply'):
                    pointcloud = o3d.io.read_point_cloud(file)

                    pcd = PointCloud(pointcloud, is_pcd=True)
                    pcd_points = np.asarray(pcd.pcd.points)

                    depth_data = pcd.transformation_pcd_to_depth_vectorized(pcd_points)
                    self.data_accumulator_source.add_data(depth_data)

                    qpixmap = get_qpixmap_grayscale(depth_data)
                    self.main_widget.lbl_page5_depth_ref.setPixmap(qpixmap)

                if file.endswith('.png'):
                    image = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
                    # pcd = PointCloud(image, is_pcd=False)
                    self.data_accumulator_source.add_data(image)

            cum_data = self.data_accumulator_source.get_filtered_data(is_remove_outlier=False)
            source_pcd = PointCloud(cum_data)
            self.source_pcd = source_pcd.pcd

            o3d.visualization.draw_geometries([self.source_pcd],
                                              width=1440, height=968,
                                              left=50, top=50,
                                              front=[0, 0, -1],
                                              lookat=[0, 0, 0.8],
                                              up=[0.0, -1, 0.1],
                                              zoom=0.64)
            self.source_pcd.estimate_normals()

            self.main_widget.lbl_ref_cum_cnt.setText(str(self.data_accumulator_source.n_accumulate))
            self.main_widget.lbl_ref_median.setText(str(np.median(cum_data)))

    def load_depth_new(self):
        files, _ = QFileDialog.getOpenFileNames(None, "파일 선택", "", "PLY 파일 (*.ply);;Numpy 파일 (*.npy);;PNG 파일 (*.png)")

        if files:
            depth = None
            for file in files:
                if file.endswith('.ply'):
                    pointcloud = o3d.io.read_point_cloud(file)

                    pcd = PointCloud(pointcloud, is_pcd=True)
                    pcd_points = np.asarray(pcd.pcd.points)

                    depth_data = pcd.transformation_pcd_to_depth_vectorized(pcd_points)
                    self.data_accumulator_target.add_data(depth_data)

                    qpixmap = get_qpixmap_grayscale(depth_data)
                    self.main_widget.lbl_page5_depth_new.setPixmap(qpixmap)

            cum_data = self.data_accumulator_target.get_filtered_data(is_remove_outlier=False)
            target_pcd = PointCloud(cum_data)
            self.target_pcd = target_pcd.pcd

            o3d.visualization.draw_geometries([self.target_pcd],
                                              width=1440, height=968,
                                              left=50, top=50,
                                              front=[0, 0, -1],
                                              lookat=[0, 0, 0.8],
                                              up=[0.0, -1, 0.1],
                                              zoom=0.64)
            self.target_pcd.estimate_normals()

            self.main_widget.lbl_new_cum_cnt.setText(str(self.data_accumulator_target.n_accumulate))
            self.main_widget.lbl_new_median.setText(str(np.median(cum_data)))

    def set_source_data(self):
        threshold = float(self.main_widget.lbl_outlier_threshold.text())
        cum_data = self.data_accumulator_source.get_filtered_data(is_remove_outlier=True, threshold=threshold)

        self.set_source_pcd(cum_data)
        self.source_data['hand_color_image'] = self.hand_color_image_source
        self.source_data['hand_depth_image'] = self.hand_depth_image_source
        self.source_data['depth_image'] = self.depth_data_uint8_source
        self.source_data['depth_data'] = self.depth_data_source
        self.source_data['hand_color_in_depth_frame'] = self.hand_color_in_depth_frame_source
        self.source_data['accumulate_count'] = self.data_accumulator_source.n_accumulate
        self.source_data['depth_median'] = np.median(self.depth_image_source)
        self.source_data['arm_position_input'] = self.arm_position_input_source

        arm_position_real = self.source_se3pose
        odom_tform_hand = self.odom_tform_hand_source

        self.source_data['arm_position_real'] = {
            "position": {
                "x": arm_position_real.position.x,
                "y": arm_position_real.position.y,
                "z": arm_position_real.position.z,
            },
            "rotation": {
                "w": arm_position_real.rotation.w,
                "x": arm_position_real.rotation.x,
                "y": arm_position_real.rotation.y,
                "z": arm_position_real.rotation.z
            }
        }
        self.source_data['odom_tform_hand'] = {
            "position": {
                "x": odom_tform_hand.position.x,
                "y": odom_tform_hand.position.y,
                "z": odom_tform_hand.position.z,
            },
            "rotation": {
                "w": odom_tform_hand.rotation.w,
                "x": odom_tform_hand.rotation.x,
                "y": odom_tform_hand.rotation.y,
                "z": odom_tform_hand.rotation.z
            }
        }

    def set_target_data(self):
        threshold = float(self.main_widget.lbl_outlier_threshold.text())
        cum_data = self.data_accumulator_target.get_filtered_data(is_remove_outlier=True, threshold=threshold)

        self.set_target_pcd(cum_data)
        self.target_data['hand_color_image'] = self.hand_color_image_target
        self.target_data['hand_depth_image'] = self.hand_depth_image_target
        self.target_data['depth_image'] = self.depth_data_uint8_target
        self.target_data['depth_data'] = self.depth_data_target
        self.target_data['hand_color_in_depth_frame'] = self.hand_color_in_depth_frame_target
        self.target_data['accumulate_count'] = self.data_accumulator_target.n_accumulate
        self.target_data['depth_median'] = np.median(self.depth_image_target)
        self.target_data['arm_position_input'] = self.arm_position_input_target

        arm_position_real = self.target_se3pose
        odom_tform_hand = self.odom_tform_hand_target

        self.target_data['arm_position_real'] = {
            "position": {
                "x": arm_position_real.position.x,
                "y": arm_position_real.position.y,
                "z": arm_position_real.position.z,
            },
            "rotation": {
                "w": arm_position_real.rotation.w,
                "x": arm_position_real.rotation.x,
                "y": arm_position_real.rotation.y,
                "z": arm_position_real.rotation.z
            }
        }
        self.target_data['odom_tform_hand'] = {
            "position": {
                "x": odom_tform_hand.position.x,
                "y": odom_tform_hand.position.y,
                "z": odom_tform_hand.position.z,
            },
            "rotation": {
                "w": odom_tform_hand.rotation.w,
                "x": odom_tform_hand.rotation.x,
                "y": odom_tform_hand.rotation.y,
                "z": odom_tform_hand.rotation.z
            }
        }

    def save_data_ref(self):
        # 저장 데이터
        # 1. 컬러 이미지 (hand_color, jpg)
        # 2. Depth 이미지 (hand_depth, npy/png)
        # 3. 누적 포인트 클라우드 (npy, ply)
        # 4. metadata (누적 횟수, depth 중앙값, arm position 정보, outlier 정보 등)
        self.set_source_data()
        print("[arm_position_input]")
        print("[사용자가 내린 이동 명령 좌표]\n")
        for key in self.source_data['arm_position_input'].keys():
            print(key, "{")
            for item in self.source_data['arm_position_input'][key]:
                print(f"  {item}: {self.source_data['arm_position_input'][key][item]}")
            print("}")

        print("[arm_position_real]")
        print("[실제 arm이 이동한 좌표 (Body 기준)]\n")
        print(self.source_data['arm_position_real'])

        print("[odom_tform_hand]")
        print("[실제 arm이 이동한 좌표 (Odometry 기준)]\n")
        print(self.source_data['odom_tform_hand'])

        self.show_input_dialog_source()

        # data = self.data_accumulator_source.cumulative_data
        # np.save('reference.npy', data)
        # o3d.io.write_point_cloud("reference.ply", self.source_pcd)

    def save_data_new(self):
        self.target_data['hand_color_image'] = self.hand_color_image_target
        self.target_data['hand_depth_image'] = self.hand_depth_image_target
        self.target_data['depth_image'] = self.depth_data_uint8_target
        self.target_data['hand_color_in_depth_frame'] = self.hand_color_in_depth_frame_target
        self.target_data['accumulate_count'] = self.data_accumulator_target.n_accumulate
        self.target_data['depth_median'] = np.median(self.depth_image_target)
        self.target_data['arm_position_input'] = self.arm_position_input_target

        arm_position_real = self.target_se3pose
        odom_tform_hand = self.odom_tform_hand_target

        self.target_data['arm_position_real'] = {
            "position": {
                "x": arm_position_real.position.x,
                "y": arm_position_real.position.y,
                "z": arm_position_real.position.z,
            },
            "rotation": {
                "w": arm_position_real.rotation.w,
                "x": arm_position_real.rotation.x,
                "y": arm_position_real.rotation.y,
                "z": arm_position_real.rotation.z
            }
        }
        self.target_data['odom_tform_hand'] = {
            "position": {
                "x": odom_tform_hand.position.x,
                "y": odom_tform_hand.position.y,
                "z": odom_tform_hand.position.z,
            },
            "rotation": {
                "w": odom_tform_hand.rotation.w,
                "x": odom_tform_hand.rotation.x,
                "y": odom_tform_hand.rotation.y,
                "z": odom_tform_hand.rotation.z
            }
        }

        self.show_input_dialog_target()

    def show_input_dialog_source(self):
        input_dialog = CustomInputDialog(self.main_window)
        input_dialog.set_data(self.source_data)
        input_dialog.initUI()
        result = input_dialog.exec_()
        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.source_data[key] for key in keys}

        if result == QDialog.Accepted:
            if not input_dialog.filename_input.text():
                return
            saved_path = input_dialog.save_path_input.text()
            saved_fname = input_dialog.filename_input.text() + ".json"
            hand_color_image_name = input_dialog.filename_input.text() + "_hand_color.jpg"
            hand_depth_image_name = input_dialog.filename_input.text() + "_hand_depth.png"
            depth_image_name = input_dialog.filename_input.text() + "_depth_image.png"
            depth_data_png_name = input_dialog.filename_input.text() + "_depth.png"
            depth_data_npy_name = input_dialog.filename_input.text() + "_depth.npy"
            hand_color_in_depth_frame_name = input_dialog.filename_input.text() + "_hand_color_in_depth_frame.png"

            saved_json = os.path.join(saved_path, saved_fname)
            with open(saved_json, 'w') as f:
                json.dump(saved_dict, f, indent=4)

            cv2.imwrite(os.path.join(saved_path, hand_color_image_name), self.source_data['hand_color_image'])
            cv2.imwrite(os.path.join(saved_path, hand_depth_image_name), self.source_data['hand_depth_image'])
            cv2.imwrite(os.path.join(saved_path, depth_image_name), self.source_data['depth_image'])
            cv2.imwrite(os.path.join(saved_path, depth_data_png_name), self.source_data['depth_data'])
            cv2.imwrite(os.path.join(saved_path, hand_color_in_depth_frame_name),
                        self.source_data['hand_color_in_depth_frame'])
            np.save(os.path.join(saved_path, depth_data_npy_name), self.source_data['depth_data'])

            o3d.io.write_point_cloud(os.path.join(saved_path, "source.ply"), self.source_pcd)

            self.main_func.show_message_box("저장이 완료되었습니다.")

    def show_input_dialog_target(self):
        input_dialog = CustomInputDialog(self.main_window)
        input_dialog.set_data(self.target_data)
        input_dialog.initUI()
        result = input_dialog.exec_()
        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.target_data[key] for key in keys}

        if result == QDialog.Accepted:
            if not input_dialog.filename_input.text():
                return
            saved_path = input_dialog.save_path_input.text()
            saved_fname = input_dialog.filename_input.text() + ".json"
            hand_color_image_name = input_dialog.filename_input.text() + "_hand_color.jpg"
            hand_depth_image_name = input_dialog.filename_input.text() + "_hand_depth.png"
            depth_image_name = input_dialog.filename_input.text() + "_depth_image.png"
            depth_data_png_name = input_dialog.filename_input.text() + "_depth.png"
            depth_data_npy_name = input_dialog.filename_input.text() + "_depth.npy"
            hand_color_in_depth_frame_name = input_dialog.filename_input.text() + "_hand_color_in_depth_frame.png"

            saved_json = os.path.join(saved_path, saved_fname)
            with open(saved_json, 'w') as f:
                json.dump(saved_dict, f, indent=4)

            cv2.imwrite(os.path.join(saved_path, hand_color_image_name), self.target_data['hand_color_image'])
            cv2.imwrite(os.path.join(saved_path, hand_depth_image_name), self.target_data['hand_depth_image'])
            cv2.imwrite(os.path.join(saved_path, depth_image_name), self.target_data['depth_image'])
            try:
                cv2.imwrite(os.path.join(saved_path, depth_data_png_name), self.target_data['depth_data'])
            except:
                pass
            cv2.imwrite(os.path.join(saved_path, hand_color_in_depth_frame_name),
                        self.target_data['hand_color_in_depth_frame'])
            np.save(os.path.join(saved_path, depth_data_npy_name), self.target_data['depth_data'])

            self.main_func.show_message_box("저장이 완료되었습니다.")

    # def view_mesh_ref_page5(self):
    #     mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=self.source_pcd)
    #     num_vertices = len(mesh.vertices)
    #     # colors = np.random.rand(num_vertices, 3)
    #     # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #
    #     o3d.visualization.draw_geometries([mesh])
    #
    #     # Mesh의 각 정점의 depth 값을 계산합니다
    #     depths = np.linalg.norm(mesh.vertices, axis=1)
    #
    #     # Depth 값을 이용하여 색상 값을 설정합니다
    #     min_depth = np.min(depths)
    #     max_depth = np.max(depths)
    #     colors = np.zeros((len(mesh.vertices), 3))
    #     for i, depth in enumerate(depths):
    #         t = (depth - min_depth) / (max_depth - min_depth)
    #         colors[i] = [t, t, t]
    #
    #     # Mesh 데이터에 색상 값을 할당합니다
    #     mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #     o3d.visualization.draw_geometries([mesh])
    #
    #     pcd1 = mesh.sample_points_uniformly(number_of_points=num_vertices)
    #     o3d.visualization.draw_geometries([pcd1])

    def init_transform(self):
        if self.source_pcd is None:
            self.main_func.show_message_box('reference pointcloud가 생성되지 않았습니다.')
            return

        if self.target_pcd is None:
            self.main_func.show_message_box('target pointcloud가 생성되지 않았습니다.')
            return

        # numpy 배열 생성
        trans_init = np.zeros((4, 4))  # 4x4 크기의 0으로 초기화된 numpy 배열 생성

        for i in range(4):
            for j in range(4):
                cell_value = float(self.main_widget.page5_tableHMatrix_init.item(i, j).text())
                trans_init[i, j] = cell_value

        draw_registration_result(self.source_pcd, self.target_pcd, trans_init)

    def surf(self):
        if self.hand_color_image_source is None:
            self.main_func.show_message_box('reference 이미지가 촬영되지 않았습니다.')
            return

        if self.hand_color_image_target is None:
            self.main_func.show_message_box('target 이미지가 촬영되지 않았습니다.')
            return

        M, found = execute_surf(self.hand_color_image_source, self.hand_color_image_target)
        if M is None:
            self.main_func.show_message_box('SURF 정합에 실패했습니다.')
            return

        print(f"[SURF] x: {M[0][2]}, y: {M[1][2]}")

        # tx = M[0][2] * 0.0013458950201884
        tx = M[0][2] * 0.00142857
        # ty = M[1][2] * 0.0025720680142442
        ty = M[1][2] * 0.00142857
        trans_init = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        overlap = overlap_images(self.hand_color_image_source, found)
        cv2.imshow("overlap", overlap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 셀에 데이터 삽입
        for i in range(4):
            for j in range(4):
                item = QTableWidgetItem(str(trans_init[i][j]))  # 임의의 데이터 생성
                self.main_widget.page5_tableHMatrix_init.setItem(i, j, item)

        # 초기행렬의 tz값 추출 (중앙값 계산)
        ref_median = float(self.main_widget.lbl_ref_median.text())
        new_median = float(self.main_widget.lbl_new_median.text())
        tz = (new_median - ref_median) / 1000
        item = QTableWidgetItem(str(tz))
        self.main_widget.page5_tableHMatrix_init.setItem(2, 3, item)

    def execute_icp(self):
        if self.source_pcd is None:
            self.main_func.show_message_box('reference pointcloud가 생성되지 않았습니다.')
            return

        if self.target_pcd is None:
            self.main_func.show_message_box('target pointcloud가 생성되지 않았습니다.')
            return

        init_table = self.main_widget.page5_tableHMatrix_init
        sigma = float(self.main_widget.lbl_icp_sigma.text())
        threshold = float(self.main_widget.lbl_icp_threshold.text())

        # numpy 배열 생성
        trans_init = np.zeros((4, 4))  # 4x4 크기의 0으로 초기화된 numpy 배열 생성

        for i in range(4):
            for j in range(4):
                cell_value = float(init_table.item(i, j).text())
                trans_init[i, j] = cell_value
        # self.transformation_matrix[:, :3] += np.array(translation)
        # trans_init[:3, 3] *= 1.5

        # # Mean and standard deviation.
        # mu, sigma = 0, 0.05
        # source_noisy = apply_noise(self.source_pcd, mu, sigma)

        # print("Using the noisy source pointcloud to perform robust ICP.\n")
        # print("Robust point-to-plane ICP, threshold={}:".format(threshold))
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        # print("Using robust loss:", loss)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        self.source_pcd.estimate_normals()
        self.target_pcd.estimate_normals()

        n = int(self.main_widget.lbl_page5_icp_iter.text())
        reg_p2l = None
        for iteration in range(n):
            reg_p2l = o3d.pipelines.registration.registration_icp(
                self.source_pcd, self.target_pcd, threshold, trans_init, p2l)
            # print(reg_p2l)
            # print("Transformation is:")
            # print(reg_p2l.transformation)

            correspondences_array = np.asarray(reg_p2l.correspondence_set)
            correspondences_color = np.zeros((len(correspondences_array), 3))
            correspondences_color[:, 0] = 1  # 빨간색으로 설정

            source_points = np.asarray(self.source_pcd.points)
            target_points = np.asarray(self.target_pcd.points)

            correspondences_pcd = o3d.geometry.PointCloud()
            correspondences_pcd.points = o3d.utility.Vector3dVector(source_points[correspondences_array[:, 0], :])
            correspondences_pcd.colors = o3d.utility.Vector3dVector(correspondences_color)

            # lineset 생성
            lines = []
            for idx in range(len(correspondences_array)):
                pt1 = source_points[correspondences_array[idx, 0]]
                pt2 = target_points[correspondences_array[idx, 1]]
                lines.append([pt1, pt2])

            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(np.asarray(lines).reshape(-1, 3))
            lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i + 1] for i in range(0, len(lines), 2)]))

            trans_init = reg_p2l.transformation
            trans_init_inv = np.linalg.inv(trans_init)

            # draw_registration_result(self.source_pcd, self.target_pcd, reg_p2l.transformation)
            # draw_registration_result_with_cor(self.source_pcd,
            #                                   self.target_pcd,
            #                                   reg_p2l.transformation,
            #                                   correspondences_pcd,
            #                                   lineset)
            if self.main_widget.cbx_icp_show.isChecked():
                draw_registration_result_with_cor(self.source_pcd,
                                                  self.target_pcd,
                                                  trans_init_inv,
                                                  correspondences_pcd,
                                                  lineset)

            self.main_widget.lbl_icp_fitness.setText(str(reg_p2l.fitness))
            self.main_widget.lbl_icp_inlier_rmse.setText(str(reg_p2l.inlier_rmse))
            self.main_widget.lbl_icp_cor_points.setText(str(len(correspondences_array)))

        # self.icp_transformation = trans_init
        self.icp_transformation = trans_init

        # 셀에 데이터 삽입
        for i in range(4):
            for j in range(4):
                item = QTableWidgetItem(str(self.icp_transformation[i][j]))  # 임의의 데이터 생성
                self.main_widget.page5_tableHMatrix_result.setItem(i, j, item)

        transformation_matrix = apply_spot_coordinate_matrix(self.icp_transformation)

        # transformation_matrix = self.icp_transformation.copy()
        # transformation_matrix = np.linalg.inv(transformation_matrix)
        # transformation_matrix[:3, 3] = transformation_matrix[:3, 3][[2, 0, 1]]
        # transformation_matrix[0, 3] = -transformation_matrix[0, 3]

        if self.target_se3pose is None:
            return

        # 타겟 Arm 위치의 SE3Pose 데이터
        # target_pose = {
        #     'x': self.target_se3pose.position.x,
        #     'y': self.target_se3pose.position.y,
        #     'z': self.target_se3pose.position.z,
        #     'rotation': {'w': self.target_se3pose.rotation.w,
        #                  'x': self.target_se3pose.rotation.x,
        #                  'y': self.target_se3pose.rotation.y,
        #                  'z': self.target_se3pose.rotation.z}
        # }

        self.set_target_data()
        target_arm_position_real = self.target_data['arm_position_real']
        target_odom_tform_hand = self.target_data['odom_tform_hand']
        target_pose_body = convert_to_target_pose(target_arm_position_real)
        target_pose_odom = convert_to_target_pose(target_odom_tform_hand)
        # transformation_matrix = apply_spot_coordinate_matrix(icp_transformation)

        # transformation_matrix = reg_p2l.transformation[:3, 3][[2, 0, 1]]

        corrected_target_pose_body = apply_transformation_to_target(transformation_matrix, target_pose_body)
        corrected_target_pose_odom = apply_transformation_to_target(transformation_matrix, target_pose_odom)

        # 보정 좌표값 UI 적용

        var_list = ['x', 'y', 'z', 'rotation']
        rotation_list = ['w', 'x', 'y', 'z']

        # target_pose_body를 수정하고 라벨을 업데이트합니다.
        for var in var_list:
            if var == 'rotation':
                for rot_var in rotation_list:
                    value = corrected_target_pose_body[var][rot_var]
                    getattr(self.main_widget, f'lbl_manual_corrected_rot_{rot_var}').setText(f"{value:.6f}")
            else:
                value = corrected_target_pose_body[var]
                getattr(self.main_widget, f'lbl_manual_corrected_pos_{var}').setText(f"{value:.6f}")

        # target_pose_odom를 수정하고 라벨을 업데이트합니다.
        for var in var_list:
            if var == 'rotation':
                for rot_var in rotation_list:
                    value = corrected_target_pose_odom[var][rot_var]
                    getattr(self.main_widget, f'lbl_manual_corrected_odom_rot_{rot_var}').setText(f"{value:.6f}")
            else:
                value = corrected_target_pose_odom[var]
                getattr(self.main_widget, f'lbl_manual_corrected_odom_pos_{var}').setText(f"{value:.6f}")

    def icp_and_correct_arm(self):
        if self.source_pcd is None:
            self.main_func.show_message_box('reference pointcloud가 생성되지 않았습니다.')
            return

        if self.target_pcd is None:
            self.main_func.show_message_box('target pointcloud가 생성되지 않았습니다.')
            return

        init_table = self.main_widget.page5_tableHMatrix_init
        sigma = float(self.main_widget.lbl_icp_sigma.text())
        threshold = float(self.main_widget.lbl_icp_threshold.text())
        iteration = int(self.main_widget.lbl_page5_icp_iter.text())

        # numpy 배열 생성
        trans_init = np.zeros((4, 4))  # 4x4 크기의 0으로 초기화된 numpy 배열 생성

        for i in range(4):
            for j in range(4):
                cell_value = float(init_table.item(i, j).text())
                trans_init[i, j] = cell_value

        # ICP 실행
        icp = ICP(self.source_pcd, self.target_pcd)

        icp.set_init_transformation(trans_init)
        icp.robust_icp(iteration=iteration, sigma=sigma, threshold=threshold)

        # ICP 점수
        fitness = icp.reg_p2l.fitness

        # 보정좌표 산출
        transformation = icp.reg_p2l.transformation
        target_pose_body = convert_to_target_pose(self.target_data['arm_position_real'])
        transformation_matrix = apply_spot_coordinate_matrix(transformation)
        corrected_target_pose = apply_transformation_to_target(transformation_matrix, target_pose_body)
        position = {
            'x': corrected_target_pose['x'],
            'y': corrected_target_pose['y'],
            'z': corrected_target_pose['z']
        }

        rotation = corrected_target_pose['rotation']
        frame_name = BODY_FRAME_NAME

        # 보정 좌표로 이동
        self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)

        # 이동 후 이미지 캡쳐


    def set_robot_arm_position_value(self):
        result_x = float(self.main_widget.page5_tableHMatrix_result.item(3, 0).text())
        result_y = float(self.main_widget.page5_tableHMatrix_result.item(3, 1).text())
        result_z = float(self.main_widget.page5_tableHMatrix_result.item(3, 2).text())

        pos_x = float(self.main_widget.lbl_page5_pos_x_real_new.text()) - result_z
        pos_y = float(self.main_widget.lbl_page5_pos_y_real_new.text()) - result_x
        pos_z = float(self.main_widget.lbl_page5_pos_z_real_new.text()) - result_y

        self.main_widget.lbl_page5_pos_x_result.setText(str(pos_x))
        self.main_widget.lbl_page5_pos_y_result.setText(str(pos_y))
        self.main_widget.lbl_page5_pos_z_result.setText(str(pos_z))

    def move_arm_result_page5(self):
        self.pos_x_result = float(self.main_widget.lbl_page5_pos_x_result.text())
        self.pos_y_result = float(self.main_widget.lbl_page5_pos_y_result.text())
        self.pos_z_result = float(self.main_widget.lbl_page5_pos_z_result.text())

        self.traj_thread = threading.Thread(target=self.main_window.robot.robot_arm_manager.trajectory_pos_rot,
                                            args=[self.pos_x_result, self.pos_y_result, self.pos_z_result])
        self.traj_thread.start()
        duration = 2.0
        start_time = time.time()
        end_time = start_time + duration + 1.0
        while time.time() < end_time:  # 스레드가 종료될 때까지 대기
            pass

        self.update_result_info()

    def update_result_info(self):
        pos = self.main_window.robot.get_current_hand_position('hand').position

        self.main_widget.lbl_page5_pos_x_real_result.setText(str(pos.x))
        self.main_widget.lbl_page5_pos_y_real_result.setText(str(pos.y))
        self.main_widget.lbl_page5_pos_z_real_result.setText(str(pos.z))

    def view_icp_result(self):
        hand_color, data = self.main_window.robot.robot_camera_manager.take_image()

        masked = cv2.bitwise_and(self.hand_color_image_source, hand_color)

        cv2.imshow('reference', self.hand_color_image_source)
        cv2.imshow('transform', hand_color)
        cv2.imshow('difference', masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transform_icp_result(self):
        if self.source_pcd is None:
            self.main_func.show_message_box('reference pointcloud가 생성되지 않았습니다.')
            return

        if self.target_pcd is None:
            self.main_func.show_message_box('target pointcloud가 생성되지 않았습니다.')
            return

        result_table = self.main_widget.page5_tableHMatrix_result
        transformation = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                cell_value = float(result_table.item(i, j).text())
                transformation[i, j] = cell_value

        self.icp_transformation = transformation
        draw_registration_result(self.source_pcd, self.target_pcd, transformation)

    def view_icp_matrix(self):
        if self.icp_transformation is None:
            self.main_func.show_message_box("Matrix가 등록되지 않았습니다.")
            return

        matrix = self.icp_transformation
        dialog = MatrixDialog(matrix)
        dialog.exec_()

    def checkbox_state_changed(self, state):
        if state == Qt.Checked:  # 체크박스가 선택된 경우
            self.set_widgets_visibility(self.main_widget.layout_odometry_source, True)
        else:  # 체크박스가 선택 해제된 경우
            self.set_widgets_visibility(self.main_widget.layout_odometry_source, False)

    def radio_button_state_changed(self):
        if self.main_widget.rbn_translation.isChecked():  # translation 버튼이 선택된 경우
            self.set_widgets_visibility(self.main_widget.layout_position, True)
            self.set_widgets_visibility(self.main_widget.layout_rotation, False)
        elif self.main_widget.rbn_rotation.isChecked():  # rotation 버튼이 선택된 경우
            self.set_widgets_visibility(self.main_widget.layout_position, False)
            self.set_widgets_visibility(self.main_widget.layout_rotation, True)

    def set_widgets_visibility(self, layout, visible):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item, QWidgetItem):
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(visible)
            elif isinstance(item, QLayoutItem):
                sublayout = item.layout()
                if sublayout is not None:
                    self.set_widgets_visibility(sublayout, visible)

    def toggle_thread(self, checked):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        if checked:
            self.thread_stop_flag = False
            self.main_window.trajectory_worker.stop_flag = self.thread_stop_flag
            self.start_thread()
            self.main_widget.btn_start.setText("중지")
        else:
            self.thread_stop_flag = True
            self.main_window.trajectory_worker.stop_flag = self.thread_stop_flag
            self.main_widget.btn_start.setText("시작")
            if hasattr(self, 'thread') and self.thread:
                if not sip.isdeleted(self.thread):
                    if self.thread.isRunning():
                        self.thread.quit()
                        self.thread.wait()
                    self.thread.deleteLater()
                    self.thread = None

            # 새로운 worker 객체 생성
            self.main_window.trajectory_worker = TrajectoryWorker(self.main_window, self.main_widget, self.tab2)

    def start_thread(self):
        if hasattr(self, 'thread') and self.thread:
            if self.thread.isRunning():
                return

        worker = self.main_window.trajectory_worker
        self.thread = QThread()

        # worker.finished.connect(self.thread.quit)
        # worker.finished.connect(worker.deleteLater)  # worker 객체를 삭제합니다.
        worker.finished.connect(self.work_fisished_event)  # 스레드가 끝났을 때 "시작"으로 변경하기 위한 슬롯을 연결합니다.
        worker.moveToThread(self.thread)

        self.thread.started.connect(worker.run)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def work_fisished_event(self):
        self.main_widget.btn_start.setChecked(False)
        self.main_widget.btn_start.setText("시작")

        # 페이지 이동
        self.main_widget.Tab2_stackedWidget.setCurrentIndex(3)

    def start(self):
        # 1. Source 설정
        # - 설정된 값에 따라 Source 데이터를 취득 및 저장(메모리)
        # - 1) position, rotation 값 읽어오기
        # - 2) 해당 위치로 arm trajectory
        # - 3) 이미지 획득 및 저장
        if self.main_widget.cbx_unstow.isChecked():
            self.main_window.robot.robot_arm_manager.unstow()
            time.sleep(1.5)
        else:
            pos_x = float(self.main_widget.lbl_pos_x_src.text())
            pos_y = float(self.main_widget.lbl_pos_y_src.text())
            pos_z = float(self.main_widget.lbl_pos_z_src.text())

            rot_x = float(self.main_widget.lbl_rot_x_src.text())
            rot_y = float(self.main_widget.lbl_rot_y_src.text())
            rot_z = float(self.main_widget.lbl_rot_z_src.text())
            rot_w = float(self.main_widget.lbl_rot_w_src.text())

            source_trajectory = threading.Thread(target=self.main_window.robot.robot_arm_manager.trajectory_pos_rot,
                                                 args=[pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w])
            source_trajectory.start()
            source_trajectory.join()

        # time.sleep(2)
        self.clear_pcd_source()
        self.capture_in_second(mode="source")
        self.set_source_data()
        self.save_source_data()
        # capture
        # color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()

        color_qimage = get_qimage(self.source_data['hand_color_in_depth_frame'])
        depth_color_qimage = get_qimage(self.source_data['hand_depth_image'])

        self.main_widget.lbl_hand_color_src.setPixmap(QPixmap.fromImage(color_qimage))
        self.main_widget.lbl_hand_depth_src.setPixmap(QPixmap.fromImage(depth_color_qimage))
        print("이미지 설정 완료")

        # 2. Target 설정
        # - 설정된 값에 따라 Target 데이터를 취득 및 저장(메모리)
        # - 1) cmb_pos_axis: 데이터 경향성을 파악할 축을 선택 (x, y, z)
        # - 2) spb_interval: 반복으로 이동할 간격
        # - 3) spb_distance: arm이 움직일 최종 위치

        axis = self.main_widget.cmb_pos_axis.currentText()
        interval = self.main_widget.spb_interval.value()
        iteration = self.main_widget.spb_distance.value()

        # 3. 반복 진행
        # Source: x = 0.8, y = 0, z = 0.2
        # y축, interval = 0.01, distance = 0.1 인 경우
        # 10회 (distance / interval) 반복
        # x = 0.8, y = 0.01, z = 0.2
        # x = 0.8, y = 0.02, z = 0.2

        self.icp = ICP(self.source_pcd, self.target_pcd)
        self.target_data_buffer.clear()

        trajectory_position = {
            "x": pos_x,
            "y": pos_y,
            "z": pos_z
        }

        print("축: ", axis)
        for i in range(iteration):
            trajectory_position[axis] += interval
            print(f"{i + 1}\t pos {axis}: ", trajectory_position[axis])
            print("시작: ", datetime.now())
            # 로봇 Arm 이동
            target_trajectory = threading.Thread(target=self.main_window.robot.robot_arm_manager.trajectory_pos_rot,
                                                 args=[trajectory_position["x"],
                                                       trajectory_position["y"],
                                                       trajectory_position["z"],
                                                       rot_x, rot_y, rot_z, rot_w])
            target_trajectory.start()
            target_trajectory.join()

            # i번째 Target 데이터 획득 및 저장
            self.clear_pcd_target()
            self.capture_in_second(mode="target")

            self.arm_position_input_target['position'][axis] = trajectory_position[axis]
            self.set_target_data()
            self.save_target_data()

            # Target 데이터 버퍼에 저장
            self.target_data_buffer.append(copy.deepcopy(self.target_data))

            # 이미지 설정
            color_qimage = get_qimage(self.target_data['hand_color_in_depth_frame'])
            depth_color_qimage = get_qimage(self.target_data['hand_depth_image'])

            self.main_widget.lbl_hand_color_tgt.setPixmap(QPixmap.fromImage(color_qimage))
            self.main_widget.lbl_hand_depth_tgt.setPixmap(QPixmap.fromImage(depth_color_qimage))

            # Overlap 이미지 설정
            source_image = copy.deepcopy(self.source_data['hand_color_image'])
            target_image = copy.deepcopy(self.target_data['hand_color_image'])

            alpha = self.main_widget.sbx_alpha.value()
            beta = self.main_widget.sbx_beta.value()
            gamma = self.main_widget.sbx_gamma.value()
            overlapped = cv2.addWeighted(source_image, alpha, target_image, beta, gamma)
            # overlapped = cv2.addWeighted(source_image, 0.5, target_image, 0.5, 0)
            overlapped_qimage = get_qimage(overlapped)

            self.main_widget.lbl_src_tgt_bitwise.setPixmap(QPixmap.fromImage(overlapped_qimage))

            # Overlap 이미지 저장
            save_file_path = self.main_widget.lblSavePath.text()
            save_file_name = f"overlapped_{i + 1}.jpg"
            cv2.imwrite(os.path.join(save_file_path, save_file_name), overlapped)

            # ICP 변수 설정
            self.icp.set_target(self.target_pcd)
            trans_init = self.get_trans_init()
            self.icp.set_init_transformation(trans_init)

            # ICP 실행
            icp_st_time = datetime.now()
            print("ICP 시작: ", icp_st_time)

            self.icp.robust_icp()

            icp_end_time = datetime.now()
            print("ICP 경과시간: ", icp_end_time - icp_st_time)
            np.savetxt(f"{self.main_widget.lblSavePath.text()}/transformation_{i + 1}.txt",
                       self.icp.transformation_buffer[i], delimiter=",")
            print(f"{i + 1}번쨰 trajectory 완료.")
            print("완료: ", datetime.now())

        # 버튼 생성
        # button_generator = ThreadButtonGenerator(iteration=iteration)
        # button_generator.button_generation.connect(self.generate_buttons)
        # button_generator.start()

        # 작업이 끝난 후 버튼 생성
        # QTimer.singleShot(1, lambda: self.generate_buttons(iteration))
        # QMetaObject.invokeMethod(self.main_window, "generate_buttons", Qt.QueuedConnection, Q_ARG(int, iteration))
        # 기존에 있는 버튼 제거
        for i in reversed(range(self.main_widget.gridLayout_icp_btns.count())):
            self.main_widget.gridLayout_icp_btns.itemAt(i).widget().setParent(None)

        event = GenerateButtonsEvent(iteration)
        QCoreApplication.postEvent(self.main_window, event)

        # 로봇 보정 페이지 UI 적용 (Source)
        positions = ['x', 'y', 'z']
        rotations = ['x', 'y', 'z', 'w']
        for position in positions:
            value = str(round(self.source_data['arm_position_real']['position'][position], 4))
            getattr(self.main_widget, f"lbl_source_pos_{position}").setText(value)

        for rotation in rotations:
            value = str(round(self.source_data['arm_position_real']['rotation'][rotation], 6))
            getattr(self.main_widget, f"lbl_source_rot_{rotation}").setText(value)

        # self.generate_buttons(iteration)

    def get_trans_init(self):
        M, found = execute_surf(self.source_data['hand_color_image'], self.target_data['hand_color_image'])
        if M is None:
            self.main_func.show_message_box('SURF 정합에 실패했습니다.')
            return np.eye(4)

        trans_init = np.identity(4)
        print(f"[SURF] x: {M[0][2]}, y: {M[1][2]}")
        # tx = M[0][2] * 0.0013458950201884
        tx = M[0][2] * 0.00142857
        # ty = M[1][2] * 0.0025720680142442
        ty = M[1][2] * 0.00142857
        # 초기행렬의 tz값 추출 (중앙값 계산)
        tz = (self.target_data['depth_median'] - self.source_data['depth_median']) / 1000
        trans_init[:3, 3] = [tx, ty, tz]

        return trans_init

    def save_source_data(self):
        saved_path = self.main_widget.lblSavePath.text()
        saved_path = os.path.join(saved_path, "source")
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        now = datetime.now()
        now = datetime.strftime(now, "%Y%m%d_%H%M%S")

        saved_fname = f"data_{now}.json"
        hand_color_image_name = f"hand_color_{now}.jpg"
        hand_depth_image_name = f"hand_depth_{now}.png"
        depth_image_name = f"depth_{now}.png"
        depth_data_png_name = f"depth_data_{now}.png"
        depth_data_npy_name = f"depth_data_{now}.npy"
        hand_color_in_depth_frame_name = f"hand_color_in_depth_frame_{now}.png"
        point_cloud_name = f"point_cloud_{now}.ply"

        saved_json = os.path.join(saved_path, saved_fname)
        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.source_data[key] for key in keys}

        with open(saved_json, 'w') as f:
            json.dump(saved_dict, f, indent=4)

        cv2.imwrite(os.path.join(saved_path, hand_color_image_name), self.source_data['hand_color_image'])
        cv2.imwrite(os.path.join(saved_path, hand_depth_image_name), self.source_data['hand_depth_image'])
        cv2.imwrite(os.path.join(saved_path, depth_image_name), self.source_data['depth_image'])
        cv2.imwrite(os.path.join(saved_path, depth_data_png_name), self.source_data['depth_data'])
        cv2.imwrite(os.path.join(saved_path, hand_color_in_depth_frame_name),
                    self.source_data['hand_color_in_depth_frame'])
        o3d.io.write_point_cloud(os.path.join(saved_path, point_cloud_name), self.source_pcd)
        np.save(depth_data_npy_name, self.source_data['depth_data'])

    def save_target_data(self):
        saved_path = self.main_widget.lblSavePath.text()
        saved_path = os.path.join(saved_path, "target")
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        now = datetime.now()
        now = datetime.strftime(now, "%Y%m%d_%H%M%S")

        saved_fname = f"data_{now}.json"
        hand_color_image_name = f"hand_color_{now}.jpg"
        hand_depth_image_name = f"hand_depth_{now}.png"
        depth_image_name = f"depth_{now}.png"
        depth_data_png_name = f"depth_data_{now}.png"
        depth_data_npy_name = f"depth_data_{now}.npy"
        hand_color_in_depth_frame_name = f"hand_color_in_depth_frame_{now}.png"
        point_cloud_name = f"point_cloud_{now}.ply"

        saved_json = os.path.join(saved_path, saved_fname)
        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.target_data[key] for key in keys}

        with open(saved_json, 'w') as f:
            json.dump(saved_dict, f, indent=4)

        cv2.imwrite(os.path.join(saved_path, hand_color_image_name), self.target_data['hand_color_image'])
        cv2.imwrite(os.path.join(saved_path, hand_depth_image_name), self.target_data['hand_depth_image'])
        cv2.imwrite(os.path.join(saved_path, depth_image_name), self.target_data['depth_image'])
        cv2.imwrite(os.path.join(saved_path, hand_color_in_depth_frame_name),
                    self.target_data['hand_color_in_depth_frame'])
        o3d.io.write_point_cloud(os.path.join(saved_path, point_cloud_name), self.target_pcd)
        cv2.imwrite(os.path.join(saved_path, depth_image_name), self.target_data['depth_image'])
        cv2.imwrite(os.path.join(saved_path, depth_data_png_name), self.target_data['depth_data'])
        np.save(os.path.join(saved_path, depth_data_npy_name), self.target_data['depth_data'])

    def save_source_data_acm(self):
        saved_path = self.main_widget.lblSavePath.text()
        saved_path = os.path.join(saved_path, "source")
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        now = datetime.now()
        now = datetime.strftime(now, "%Y%m%d_%H%M%S")

        saved_fname = get_unique_filename(saved_path, f"data_{now}.json")
        hand_color_image_name = get_unique_filename(saved_path, f"hand_color_{now}.jpg")
        hand_depth_image_name = get_unique_filename(saved_path, f"hand_depth_{now}.png")
        depth_image_name = get_unique_filename(saved_path, f"depth_{now}.png")
        depth_data_png_name = get_unique_filename(saved_path, f"depth_data{now}.png")
        depth_data_npy_name = get_unique_filename(saved_path, f"depth_data{now}.npy")
        hand_color_in_depth_frame_name = get_unique_filename(saved_path, f"hand_color_in_depth_frame_{now}.png")
        point_cloud_name = get_unique_filename(saved_path, f"point_cloud_{now}.ply")

        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.source_data[key] for key in keys}

        with open(saved_fname, 'w') as f:
            json.dump(saved_dict, f, indent=4)

        cv2.imwrite(hand_color_image_name, self.source_data['hand_color_image'])
        cv2.imwrite(hand_depth_image_name, self.source_data['hand_depth_image'])
        cv2.imwrite(depth_image_name, self.source_data['depth_image'])
        cv2.imwrite(depth_data_png_name, self.source_data['depth_data'])
        cv2.imwrite(hand_color_in_depth_frame_name, self.source_data['hand_color_in_depth_frame'])
        o3d.io.write_point_cloud(point_cloud_name, self.source_pcd)
        np.save(depth_data_npy_name, self.source_data['depth_data'])

    def save_target_data_acm(self):
        saved_path = self.main_widget.lblSavePath.text()
        saved_path = os.path.join(saved_path, "target")
        folder_path = ""
        counter = 1
        while os.path.exists(saved_path):
            folder_path = f"{saved_path}_{counter}"
            counter += 1
            os.makedirs(folder_path)

            saved_path = folder_path

        now = datetime.now()
        now = datetime.strftime(now, "%Y%m%d_%H%M%S")

        saved_fname = get_unique_filename(saved_path, f"data_{now}.json")
        hand_color_image_name = get_unique_filename(saved_path, f"hand_color_{now}.jpg")
        hand_depth_image_name = get_unique_filename(saved_path, f"hand_depth_{now}.png")
        depth_image_name = get_unique_filename(saved_path, f"depth_{now}.png")
        depth_data_png_name = get_unique_filename(saved_path, f"depth_data{now}.png")
        depth_data_npy_name = get_unique_filename(saved_path, f"depth_data{now}.npy")
        hand_color_in_depth_frame_name = get_unique_filename(saved_path, f"hand_color_in_depth_frame_{now}.png")
        point_cloud_name = get_unique_filename(saved_path, f"point_cloud_{now}.ply")

        keys = ['depth_median', 'accumulate_count', 'arm_position_real', 'arm_position_input', 'odom_tform_hand']
        saved_dict = {key: self.target_data[key] for key in keys}

        with open(saved_fname, 'w') as f:
            json.dump(saved_dict, f, indent=4)

        cv2.imwrite(hand_color_image_name, self.target_data['hand_color_image'])
        cv2.imwrite(hand_depth_image_name, self.target_data['hand_depth_image'])
        cv2.imwrite(depth_image_name, self.target_data['depth_image'])
        cv2.imwrite(depth_data_png_name, self.target_data['depth_data'])
        cv2.imwrite(hand_color_in_depth_frame_name, self.target_data['hand_color_in_depth_frame'])
        o3d.io.write_point_cloud(point_cloud_name, self.target_pcd)
        np.save(depth_data_npy_name, self.target_data['depth_data'])

    def generate_buttons(self, n):
        # # 기존에 있는 버튼 제거
        # for i in reversed(range(self.main_widget.gridLayout_icp_btns.count())):
        #     self.main_widget.gridLayout_icp_btns.itemAt(i).widget().setParent(None)

        # n개의 새로운 버튼 생성
        max_cols = 5
        row, col = 0, 0
        for i in range(n):
            button = QPushButton(f'Target {i + 1}', parent=self.main_window)

            button.clicked.connect(lambda checked, i=i: self.target_button_clicked(i))
            self.main_widget.gridLayout_icp_btns.addWidget(button, row, col)

            col += 1
            if col == max_cols:
                col = 0
                row += 1

    def target_button_clicked(self, idx):
        self.selected_target_idx = idx

        # 변환행렬 데이터 삽입
        result_table = self.main_widget.page5_tableHMatrix_result
        icp_transformation = self.icp.transformation_buffer[idx]
        for i in range(4):
            for j in range(4):
                item = QTableWidgetItem(str(icp_transformation[i][j]))
                result_table.setItem(i, j, item)

        # Registration Result 데이터 삽입
        icp_result = self.icp.icp_result_buffer[idx]
        fitness = icp_result.fitness
        inlier_rmse = icp_result.inlier_rmse
        cor_points = len(icp_result.correspondence_set)

        self.main_widget.lbl_icp_reg_fitness.setText(str(fitness))
        self.main_widget.lbl_icp_reg_inlier_rmse.setText(str(inlier_rmse))
        self.main_widget.lbl_icp_reg_cor_points.setText(str(cor_points))

        # 로봇 보정 좌표 계산
        # if self.main_widget.cbx_odometry.isChecked():
        #     target_odom_tform_hand = self.target_data_buffer[idx]['odom_tform_hand']
        #     target_pose = convert_to_target_pose(target_odom_tform_hand)
        #     transformation_matrix = apply_spot_coordinate_matrix(icp_transformation)
        #
        # else:
        #     target_arm_position_real = self.target_data_buffer[idx]['arm_position_real']
        #     target_pose = convert_to_target_pose(target_arm_position_real)
        #     transformation_matrix = apply_spot_coordinate_matrix(icp_transformation)

        target_arm_position_real = self.target_data_buffer[idx]['arm_position_real']
        target_pose = convert_to_target_pose(target_arm_position_real)

        transformation_matrix = apply_spot_coordinate_matrix(icp_transformation)
        corrected_target_pose = apply_transformation_to_target(transformation_matrix, target_pose)

        # 로봇 보정 페이지 UI 적용
        positions = ['x', 'y', 'z']
        rotations = ['x', 'y', 'z', 'w']

        for position in positions:
            corrected_value = str(round(corrected_target_pose[position], 4))
            target_value = str(round(target_pose[position], 4))
            getattr(self.main_widget, f"lbl_corrected_pos_{position}").setText(corrected_value)
            getattr(self.main_widget, f"lbl_target_pos_{position}").setText(target_value)

        for rotation in rotations:
            corrected_value = str(round(corrected_target_pose['rotation'][rotation], 6))
            target_value = str(round(target_pose['rotation'][rotation], 6))
            getattr(self.main_widget, f"lbl_corrected_rot_{rotation}").setText(corrected_value)
            getattr(self.main_widget, f"lbl_target_rot_{rotation}").setText(target_value)

        self.main_widget.lbl_target_number.setText(f"Target - {idx + 1}")

        # Source / Target 병합 이미지 디스플레이
        overlapped = cv2.addWeighted(self.source_data['hand_color_image'], 0.5,
                                     self.target_data_buffer[idx]['hand_color_image'], 0.5, 0)
        overlapped_qimage = get_qimage(overlapped)
        self.main_widget.lbl_src_tgt_merge.setPixmap(QPixmap.fromImage(overlapped_qimage))

    def viz_tform_icp(self):
        # source = copy.deepcopy(self.source_pcd)
        # target = self.icp.target_buffer[self.selected_target_idx]
        # transformation = self.icp.transformation_buffer[self.selected_target_idx]
        # correspondences = self.icp.correspondences_pcd_buffer[self.selected_target_idx]
        # lineset = self.icp.lineset_buffer[self.selected_target_idx]
        if self.source_pcd is None:
            self.main_func.show_message_box("Source 포인트 클라우드가 등록되지 않았습니다.")
            return

        if not self.icp.target_buffer:
            self.main_func.show_message_box("Target 포인트 클라우드가 등록되지 않았습니다.")
            return

        source = copy.deepcopy(self.source_pcd)
        target = copy.deepcopy(self.icp.target_buffer[self.selected_target_idx])
        transformation = copy.deepcopy(self.icp.transformation_buffer[self.selected_target_idx])

        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])

        target.transform(np.linalg.inv(transformation))

        o3d.visualization.draw_geometries([source, target],
                                          width=1440, height=968, left=50, top=50,
                                          front=[-0.02, -0.02, -0.999],
                                          lookat=[0.02, -0.05, 0.9],
                                          up=[0.01, -0.99, 0.021],
                                          zoom=0.84)

    def viz_tform_icp_feature(self):
        if not self.icp.target_buffer:
            self.main_func.show_message_box("포인트 클라우드가 등록되지 않았습니다.")
            return

        self.icp.draw_registration_result(self.selected_target_idx)

    def tform_matrix_dialog(self):
        if not self.icp.transformation_buffer:
            self.main_func.show_message_box("Matrix가 등록되지 않았습니다.")
            return

        matrix = self.icp.transformation_buffer[self.selected_target_idx]
        dialog = MatrixDialog(matrix)
        dialog.exec_()

    def capture_corrected(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture_page5()

        self.merged_image = cv2.addWeighted(self.source_data['hand_color_image'], 0.5, hand_color, 0.5, 0)
        overlapped_qimage = get_qimage(self.merged_image)

        self.main_widget.lbl_src_corrected_merge.setPixmap(QPixmap.fromImage(overlapped_qimage))

    def move_position(self, widget, prefix):
        if self.main_window.robot.robot_arm_manager is None:
            self.main_func.show_message_box("로봇 연결이 필요합니다.")
            return

        labels = [
            (getattr(widget, f"lbl_{prefix}_pos_x"), "position_x"),
            (getattr(widget, f"lbl_{prefix}_pos_y"), "position_y"),
            (getattr(widget, f"lbl_{prefix}_pos_z"), "position_z"),
            (getattr(widget, f"lbl_{prefix}_rot_x"), "rotation_x"),
            (getattr(widget, f"lbl_{prefix}_rot_y"), "rotation_y"),
            (getattr(widget, f"lbl_{prefix}_rot_z"), "rotation_z"),
            (getattr(widget, f"lbl_{prefix}_rot_w"), "rotation_w")
        ]

        for label, label_name in labels:
            if label.text() == "":
                self.main_func.show_message_box(f"{label_name} 값이 입력되지 않았습니다.")
                return

        pos = [float(label.text()) for label, _ in labels[:3]]
        rot = [float(label.text()) for label, _ in labels[3:]]

        # if self.main_widget.cbx_odometry.isChecked():
        #     trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_odometry
        # else:
        #     trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_pos_rot

        trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_pos_rot
        self.traj_thread = threading.Thread(target=trajectory_function,
                                            args=pos + rot)
        self.traj_thread.start()

    def move_corrected_pos(self):
        widget = self.main_widget
        self.move_position(widget, "corrected")

    def move_target_pos(self):
        widget = self.main_widget
        self.move_position(widget, "target")

    def move_source_pos(self):
        widget = self.main_widget
        self.move_position(widget, "source")

    def save_merged_image(self):
        if self.merged_image is None:
            self.main_func.show_message_box("이미지가 없습니다.")
            return

        save_image_dialog = SaveImageDialog(self.main_window)
        result = save_image_dialog.exec_()

        if result == QDialog.Accepted:
            file_name = save_image_dialog.file_name_input.text()
            folder_path = save_image_dialog.folder_label.text()

            if file_name and folder_path != "선택된 폴더 없음":
                file_name = f"{file_name}.jpg"
                save_path = os.path.join(folder_path, file_name)

                # 이미지 저장
                cv2.imwrite(save_path, self.merged_image)
                print(f"이미지가 저장되었습니다: {save_path}")
            else:
                print("저장할 파일명이나 폴더가 선택되지 않았습니다.")
        else:
            print("저장이 취소되었습니다.")


class FPFHPage:
    def __init__(self, main_widget, main_window):
        self.main_widget = main_widget
        self.main_window = main_window
        self.main_func = MainFunctions(self.main_window)

        self.source_pcd = None
        self.target_pcd = None

        self.transformation = None

        self.ma_filter_src = DepthAccumulator(buffer_size=10)
        self.ma_filter_tgt = DepthAccumulator(buffer_size=10)

        self.init_signals()

    def init_signals(self):
        self.main_widget.btn_load_depth_fpfh_src.clicked.connect(self.load_depth_src)
        self.main_widget.btn_load_depth_fpfh_tgt.clicked.connect(self.load_depth_tgt)

        self.main_widget.btn_run_fpfh.clicked.connect(self.run_fpfh)

        self.main_widget.btn_view_pcd_fpfh_src.clicked.connect(self.view_pcd_src)
        self.main_widget.btn_view_pcd_fpfh_tgt.clicked.connect(self.view_pcd_tgt)

        self.main_widget.btn_view_pcd_src_tgt.clicked.connect(self.view_pcd_src_tgt)
        self.main_widget.btn_view_fpfh_result.clicked.connect(self.view_fpfh_result)
        self.main_widget.btn_evaluate_fpfh.clicked.connect(self.evaluate_fpfh)

    def load_depth_src(self):
        files, _ = QFileDialog.getOpenFileNames(None, "파일 선택", "", "PLY 파일 (*.ply);;Numpy 파일 (*.npy);;PNG 파일 (*.png)")

        if files:
            self.ma_filter_src.clear()

            for file in files:
                if file.endswith('.ply'):
                    pointcloud = o3d.io.read_point_cloud(file)

                    pcd = PointCloud(pointcloud, is_pcd=True)
                    pcd_points = np.asarray(pcd.pcd.points)

                    depth_data = pcd.transformation_pcd_to_depth_vectorized(pcd_points)
                    self.ma_filter_src.add_data(depth_data)

                    qpixmap = get_qpixmap_grayscale(depth_data)
                    # self.main_widget.lbl_depth_image_src.setPixmap(qpixmap)

            cum_data = self.ma_filter_src.get_filtered_data(is_remove_outlier=False)
            source_pcd = PointCloud(cum_data)
            self.source_pcd = source_pcd.pcd

            o3d.visualization.draw_geometries([self.source_pcd],
                                              width=1440, height=968,
                                              left=50, top=50,
                                              front=[0, 0, -1],
                                              lookat=[0, 0, 0.8],
                                              up=[0.0, -1, 0.1],
                                              zoom=0.64)
            # self.source_pcd.estimate_normals()

            # self.main_widget.lbl_cum_cnt_src.setText(str(self.ma_filter_src.n_accumulate))
            # self.main_widget.lbl_depth_median_src.setText(str(np.median(cum_data)))

    def load_depth_tgt(self):
        files, _ = QFileDialog.getOpenFileNames(None, "파일 선택", "", "PLY 파일 (*.ply);;Numpy 파일 (*.npy);;PNG 파일 (*.png)")

        if files:
            self.ma_filter_tgt.clear()
            depth = None
            for file in files:
                if file.endswith('.ply'):
                    pointcloud = o3d.io.read_point_cloud(file)

                    pcd = PointCloud(pointcloud, is_pcd=True)
                    pcd_points = np.asarray(pcd.pcd.points)

                    depth_data = pcd.transformation_pcd_to_depth_vectorized(pcd_points)
                    self.ma_filter_tgt.add_data(depth_data)

                    qpixmap = get_qpixmap_grayscale(depth_data)
                    # self.main_widget.lbl_depth_image_tgt.setPixmap(qpixmap)

            cum_data = self.ma_filter_tgt.get_filtered_data(is_remove_outlier=False)
            target_pcd = PointCloud(cum_data)
            self.target_pcd = target_pcd.pcd

            o3d.visualization.draw_geometries([self.target_pcd],
                                              width=1440, height=968,
                                              left=50, top=50,
                                              front=[0, 0, -1],
                                              lookat=[0, 0, 0.8],
                                              up=[0.0, -1, 0.1],
                                              zoom=0.64)
            # self.target_pcd.estimate_normals()

            # self.main_widget.lbl_cum_cnt_tgt.setText(str(self.ma_filter_tgt.n_accumulate))
            # self.main_widget.lbl_depth_median_tgt.setText(str(np.median(cum_data)))

    def view_pcd_src(self):
        o3d.visualization.draw_geometries([self.source_pcd],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 0.8],
                                          up=[0.0, -1, 0.1],
                                          zoom=0.64)

    def view_pcd_tgt(self):
        o3d.visualization.draw_geometries([self.target_pcd],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 0.8],
                                          up=[0.0, -1, 0.1],
                                          zoom=0.64)

    def view_pcd_src_tgt(self):
        o3d.visualization.draw_geometries([self.source_pcd, self.target_pcd],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 0.8],
                                          up=[0.0, -1, 0.1],
                                          zoom=0.64)

    def fpfh_feature_matching(self,
                              voxel_size=0.05,
                              is_downsample=False,
                              pcd_normal_max_nn=30,
                              fpfh_feature_max_nn=100,
                              mutual_filter=True,
                              ransac_n=3):
        # PLY 파일을 읽어들이기
        source = self.source_pcd
        target = self.target_pcd

        # 정규화 및 다운샘플링
        if is_downsample:
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)
        else:
            source_down = source
            target_down = target
        # o3d.visualization.draw_geometries([source_down, target_down])

        st_time = time.time()

        # 포인트 클라우드의 노멀 계산
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=pcd_normal_max_nn))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=pcd_normal_max_nn))

        esitmate_normal_time = time.time()

        # FPFH 특징 계산
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=fpfh_feature_max_nn))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=fpfh_feature_max_nn))

        compute_fpfh_feature_time = time.time()

        # RANSAC을 사용한 정합
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter,
            voxel_size * 1.5,  # max_correspondence_distance
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # estimation_method
            ransac_n,  # ransac_n
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
            ],  # checkers
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)  # criteria
        )

        reg_ransac_time = time.time()
        print(result)
        print(f"포인트클라우드 노멀 계산 경과 시간: {esitmate_normal_time - st_time}")
        print(f"fpfh feature 계산 경과 시간: {compute_fpfh_feature_time - esitmate_normal_time}")
        print(f"Registration 경과 시간: {reg_ransac_time - compute_fpfh_feature_time}")
        print(f"총 경과 시간: {reg_ransac_time - st_time}")

        # 결과 반환
        return result

    def run_fpfh(self):
        voxel_size = self.main_widget.sbx_voxel_size.value()
        pcd_normal_max_nn = self.main_widget.sbx_pcdn_maxnn.value()
        fpfh_feature_max_nn = self.main_widget.sbx_fpfh_maxnn.value()
        ransac_n = self.main_widget.sbx_ransac_n.value()
        mutual_filter = self.main_widget.cbx_mutual_filter.isChecked()
        is_downsampling = self.main_widget.cbx_downsampling.isChecked()

        result = self.fpfh_feature_matching(voxel_size, is_downsampling, pcd_normal_max_nn,
                                            fpfh_feature_max_nn, mutual_filter, ransac_n)

        self.transformation = result.transformation
        print("Fitness: ", result.fitness)
        print("Inlier RMSE: ", result.inlier_rmse)

        self.main_widget.lbl_fpfh_fitness.setText(str(result.fitness))
        self.main_widget.lbl_fpfh_inlier_rmse.setText(str(result.inlier_rmse))
        self.main_widget.lbl_fpfh_cor_points.setText(str(len(result.correspondence_set)))

        # 셀에 데이터 삽입
        for i in range(4):
            for j in range(4):
                item = QTableWidgetItem(str(self.transformation[i][j]))  # 임의의 데이터 생성
                self.main_widget.tableHMatrix_fpfh_result.setItem(i, j, item)

    def view_fpfh_result(self):
        source_transformed = copy.deepcopy(self.source_pcd)
        target = copy.deepcopy(self.target_pcd)
        source_transformed.transform(self.transformation)

        source_transformed.paint_uniform_color([1, 0, 0])  # 빨간색
        target.paint_uniform_color([0, 1, 0])  # 초록색

        o3d.visualization.draw_geometries([source_transformed, target],
                                          width=1440, height=968,
                                          left=50, top=50,
                                          front=[0, 0, -1],
                                          lookat=[0, 0, 0.8],
                                          up=[0.0, -1, 0.1],
                                          zoom=0.64)

    def evaluate_fpfh(self):
        threshold = self.main_widget.sbx_max_cor_distance.value()
        evaluation = o3d.pipelines.registration.evaluate_registration(
            self.source_pcd, self.target_pcd, threshold, self.transformation)

        print(evaluation)

        self.main_widget.lbl_fpfh_fitness_eval.setText(str(evaluation.fitness))
        self.main_widget.lbl_fpfh_inlier_rmse_eval.setText(str(evaluation.inlier_rmse))
        self.main_widget.lbl_fpfh_cor_points_eval.setText(str(len(evaluation.correspondence_set)))


def move_position(position, rotation, trajectory_function):
    pos_x = position['x']
    pos_y = position['y']
    pos_z = position['z']
    rot_x = rotation['x']
    rot_y = rotation['y']
    rot_z = rotation['z']
    rot_w = rotation['w']
    thread = threading.Thread(target=trajectory_function,
                              args=[pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w])
    thread.start()


def execute_surf(img1, img2, ratio_threshold=0.3):
    surf = cv2.xfeatures2d.SURF_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # (3) Create flann matcher
    # print("## (3) Create flann matcher")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

    # print("## (4) Detect keypoints and compute keypointer descriptors")
    kpts1, descs1 = surf.detectAndCompute(gray1, None)
    kpts2, descs2 = surf.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)

    # (6) Ratio test, to get good matches.
    # print("## (6) Ratio test, to get good matches.")
    good = [m1 for (m1, m2) in matches if m1.distance < ratio_threshold * m2.distance]
    found = None
    M = None
    if len(good) > 4:
        # (queryIndex for the small object, trainIndex for the scene )
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # outlier가 제거된 좋은 매칭 포인트들의 위치들을 추출합니다.
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        # 추출된 좋은 매칭 포인트들의 위치를 사용하여 homography matrix를 다시 추정합니다.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
        found = cv2.warpPerspective(img2, perspectiveM, (w, h))

        src_coord = []
        dst_coord = []
        for src_pt in src_pts[:]:
            x, y = src_pt[0]
            src_coord.append((x, y))

        for dst_pt in dst_pts[:]:
            x, y = dst_pt[0]
            dst_coord.append((x, y))

        diff = [np.array(s_coord) - np.array(d_coord)
                for s_coord, d_coord in zip(src_coord, dst_coord)]
        x_diff, y_diff = np.transpose(diff)

        x_diff_without_outliers = remove_outlier(x_diff, q1=40, q3=60)
        y_diff_without_outliers = remove_outlier(y_diff, q1=40, q3=60)
        # print(np.mean(x_diff_without_outliers))
        # print(np.mean(y_diff_without_outliers))

    return M, found


def remove_outlier(coord, q1, q3):
    # Calculate the first and third quartile (Q1 and Q3)
    Q1, Q3 = np.percentile(coord, [q1, q3])

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Remove the outliers.
    # data_without_outliers = [x for x in x_diff if lower_bound <= x <= upper_bound]
    data_without_outliers = [x for x in coord if lower_bound <= x <= upper_bound]

    return data_without_outliers


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # target_temp.transform(transformation)
    # o3d.visualization.draw([source_temp, target_temp])
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width=1440, height=968,
                                      left=50, top=50,
                                      front=[0.013, -0.081, -0.996],
                                      lookat=[0, 0, 0.8],
                                      up=[-0.01, -1, 0.08],
                                      zoom=0.3)

    # 시각화
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   window_name='source and target point cloud',
    #                                   width=800, height=600,
    #                                   left=50, top=50, point_show_normal=False,
    #                                   lookat=np.array([1.2397, 1.8220, 1.0438]),
    #                                   up=np.array([0.1204, -0.9851, 0.1244]))


def draw_registration_result_with_cor(source, target, transformation, correspondences, lineset):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    translation = np.array([
        [1, 0, 0, -1.3],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1]
    ])
    correspondences.transform(translation)
    lineset.transform(translation)
    source_temp_2 = copy.deepcopy(source_temp)
    target_temp_2 = copy.deepcopy(target_temp)
    source_temp_2.transform(translation)
    target_temp_2.transform(translation)

    # source_temp.transform(transformation)
    target_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp_2, target_temp_2, lineset, source_temp, target_temp],
                                      width=1440, height=968,
                                      left=50, top=50,
                                      front=[0.013, -0.081, -0.996],
                                      lookat=[0, 0, 0.8],
                                      up=[-0.01, -1, 0.08],
                                      zoom=0.3)

    # o3d.visualization.draw([source_temp, target_temp])
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   width=1440, height=968,
    #                                   left=50, top=50)


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


