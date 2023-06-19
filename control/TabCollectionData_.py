from datetime import datetime

import open3d as o3d
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from bosdyn.client import frame_helpers

from bosdyn.client.frame_helpers import *

from control.Calculator import DepthAccumulator
from control.Control import MainFunctions
from control.PointCloud import PointCloud
from control.utils.utils import *
from model.CorrectedMetadata import CorrectedMetadata
from model.ImagePath import ImagePath
from model.SourceMetadata import SourceMetadata, SourceImageData
from model.TargetMetadata import TargetMetadata


class TabCollectionData:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.joint_param_source  = None
        self.body_tform_hand_source = None
        self.odom_tform_hand_source = None

        self.data_accumulator_source = DepthAccumulator(buffer_size=100)
        self.data_accumulator_target = DepthAccumulator(buffer_size=100)
        self.data_accumulator_corrected = DepthAccumulator(buffer_size=100)

        self.source_metadata = SourceMetadata()
        self.source_image_data = SourceImageData()

        self.target_metadata = TargetMetadata()
        self.corrected_metadata = CorrectedMetadata()

        self.source_depth_data = None
        self.source_depth_median = None
        self.source_pcd = None

        self.target_depth_data = None
        self.target_depth_median = None
        self.target_pcd = None

        self.collection_thread = DataCollectionThread(self)

        # Fiducial
        self.arm_data_with_fiducial = None

        self.init_signal()

    def init_signal(self):
        self.main_widget.btn_col_save_path.clicked.connect(self.setting_save_path)
        self.main_widget.btn_col_navigate_to_src.clicked.connect(self.setting_navigate_to)
        self.main_widget.btn_load_arm_json_src.clicked.connect(self.load_arm_json_source)
        self.main_widget.btn_col_move_arm.clicked.connect(lambda: self.move_arm(self.main_widget.cmb_move_type.currentText()))
        self.main_widget.btn_col_capture_src.clicked.connect(self.capture_source)
        self.main_widget.btn_col_view_pcd_src.clicked.connect(self.view_pcd_source)
        self.main_widget.btn_col_clear_pcd_src.clicked.connect(self.clear_pcd_source)

        self.main_widget.rbn_setting_source.toggled.connect(self.handle_radio_button)
        self.main_widget.rbn_setting_target.toggled.connect(self.handle_radio_button)
        self.main_widget.rbn_setting_corrected.toggled.connect(self.handle_radio_button)

        self.main_widget.btn_col_start_target.toggled.connect(self.start_target_in_background)

        self.fiducial_page_init()

    def handle_radio_button(self):
        if self.main_widget.rbn_setting_source.isChecked():
            self.main_widget.stackedWidget_col_setting_page.setCurrentIndex(0)
            self.main_widget.lbl_setting_title.setText("Source Setting")
        elif self.main_widget.rbn_setting_target.isChecked():
            self.main_widget.stackedWidget_col_setting_page.setCurrentIndex(1)
            self.main_widget.lbl_setting_title.setText("Target Setting")
        elif self.main_widget.rbn_setting_corrected.isChecked():
            self.main_widget.stackedWidget_col_setting_page.setCurrentIndex(2)
            self.main_widget.lbl_setting_title.setText("Corrected Setting")
        elif self.main_widget.rbn_setting_fiducial.isChecked():
            self.main_widget.stackedWidget_col_setting_page.setCurrentIndex(3)
            self.main_widget.lbl_setting_title.setText("Fiducial Setting")

    def setting_save_path(self):
        folder = self.main_window.file_dialog.getExistingDirectory(self.main_window, "Select Directory")
        self.main_widget.lbl_save_path_src.setText(folder)

    def setting_navigate_to(self):
        destination = self.main_widget.lbl_col_waypoint_src.text()
        self.main_widget.lbl_col_waypoint2_target.setText(destination)

    def load_arm_json_source(self):
        data = self.main_func.arm_json_load()
        if data:
            if 'joint_params' not in data.keys():
                self.main_func.show_message_box('올바른 형식의 파일이 아닙니다.')
                return

            sh0 = data['joint_params']['sh0']
            sh1 = data['joint_params']['sh1']
            el0 = data['joint_params']['el0']
            el1 = data['joint_params']['el1']
            wr0 = data['joint_params']['wr0']
            wr1 = data['joint_params']['wr1']

            self.main_widget.lbl_col_sh0.setText(str(sh0))
            self.main_widget.lbl_col_sh1.setText(str(sh1))
            self.main_widget.lbl_col_el0.setText(str(el0))
            self.main_widget.lbl_col_el1.setText(str(el1))
            self.main_widget.lbl_col_wr0.setText(str(wr0))
            self.main_widget.lbl_col_wr1.setText(str(wr1))

            set_position_and_rotation(self.main_widget, "col_body", data['body_position'], data['body_rotation'])

    def move_arm(self, mode):
        move_type_mapping = {
            "body": {
                "label_name": "col_body",
                "frame_name": GRAV_ALIGNED_BODY_FRAME_NAME
            },
            "odom": {
                "label_name": "col_odom",
                "frame_name": ODOM_FRAME_NAME
            },
            "fiducial": {
                "label_name": "col_fid",
                "frame_name": ODOM_FRAME_NAME
            },
            "joint": {
                "label_name": "col_joint",
            }
        }
        move_type_info = move_type_mapping.get(mode)
        label_name = move_type_info.get("label_name")

        # 1. Arm 위치 정보 저장
        # 1) Joint Parameters
        sh0 = float(self.main_widget.lbl_col_sh0.text())
        sh1 = float(self.main_widget.lbl_col_sh1.text())
        el0 = float(self.main_widget.lbl_col_el0.text())
        el1 = float(self.main_widget.lbl_col_el1.text())
        wr0 = float(self.main_widget.lbl_col_wr0.text())
        wr1 = float(self.main_widget.lbl_col_wr1.text())

        params = [sh0, sh1, el0, el1, wr0, wr1]
        self.joint_param_source = params

        # 2) body_tform_hand
        position, rotation = get_position_and_rotation_from_label(self.main_widget, label_name)
        self.body_tform_hand_source = {
            "position": position,
            "rotation": rotation
        }
        # 2. Arm 위치 이동
        # if selected_move_type == "joint":
        #     self.main_window.robot.robot_arm_manager.joint_move_manual(params)
        # else:
        #     frame_name = move_type_info.get("frame_name")
        #     self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)

        frame_name = move_type_info.get("frame_name")
        if mode == "body":
            self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)
        elif mode == "odom":
            self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name)
        elif mode == "joint":
            self.main_window.robot.robot_arm_manager.joint_move_manual(params)

    def capture_source(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        # 1. 컬러 이미지
        hand_color, _ = self.main_window.robot.robot_camera_manager.take_image()
        hand_color_qimage = get_qimage(hand_color)
        self.main_widget.lbl_col_hand_color_src.setPixmap(QPixmap.fromImage(hand_color_qimage))

        # 1-1) 컬러 이미지 저장
        # hand_color_file_name = "image.jpg"
        # cv2.imwrite(hand_color_file_name, hand_color)

        # 2. Depth 이미지
        # 2-1) Depth 이미지 취득 시 Outlier 제거?
        iqr1 = int(self.main_widget.lbl_col_iqr1.text())
        iqr3 = int(self.main_widget.lbl_col_iqr3.text())
        iqr = [iqr1, iqr3]

        depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
            iqr=iqr,
            outlier_removal=self.main_widget.cbx_col_outlier_remove_iqr.isChecked()
        )

        # 2-2) Depth 이미지 누적
        self.data_accumulator_source.add_data(depth_image)
        self.main_widget.lbl_col_depth_acm_count.setText(str(self.data_accumulator_source.n_accumulate))
        # self.depth_image_source = self.data_accumulator_source.get_filtered_data(is_remove_outlier=False)
        threshold = float(self.main_widget.lbl_col_outlier_threshold.text())
        self.source_depth_data = self.data_accumulator_source.get_filtered_data(
            is_remove_outlier=self.main_widget.cbx_col_outlier_remove_threshold.isChecked(),
            threshold=threshold)
        self.source_depth_median = np.median(self.source_depth_data)
        if self.source_depth_median == 0:
            self.source_depth_median = np.mean(self.source_depth_data)

        self.main_widget.lbl_col_depth_median_src.setText(str(self.source_depth_median))

        # 2-3). Depth 데이터 저장
        odom_position, odom_rotation = self.main_window.robot.get_odom_tform_hand_dict()
        self.odom_tform_hand_source = {
            "position": odom_position,
            "rotation": odom_rotation
        }
        hand_coord_source = {
            "body": self.body_tform_hand_source,
            "odom": self.odom_tform_hand_source,
            "joint": self.joint_param_source
        }
        self.source_metadata.set_time(datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3])
        self.source_metadata.set_hand_coord(hand_coord_source)
        self.source_metadata.set_depth_median(self.source_depth_median)

        # 3. 이미지 Display
        depth_color = self.main_window.robot.robot_camera_manager.depth_to_color(depth_image)
        color_in_depth_frame = self.main_window.robot.robot_camera_manager.take_image_from_source('hand_color_in_hand_depth_frame')
        color_in_depth_frame = cv2.rotate(color_in_depth_frame, cv2.ROTATE_90_CLOCKWISE)

        color_in_depth_frame_qimage = get_qimage(color_in_depth_frame)
        depth_color_qimage = get_qimage(depth_color)

        self.main_widget.lbl_col_depth_color_src.setPixmap(QPixmap.fromImage(depth_color_qimage))
        self.main_widget.lbl_col_hcidf_src.setPixmap(QPixmap.fromImage(color_in_depth_frame_qimage))

        # 4. 이미지 데이터 세팅
        self.setting_source()
        self.view_pcd_source(is_show=False)
        self.source_image_data.set_data(hand_color=hand_color,
                                        depth_color=depth_color,
                                        hand_color_in_depth_frame=color_in_depth_frame,
                                        depth_data=self.source_depth_data,
                                        pointcloud=self.source_pcd)

        self.save_source_data()

    @staticmethod
    def _generate_unique_file_name(root, file_name):
        saved_path = os.path.join(root, file_name)
        counter = 1
        while os.path.exists(saved_path):
            base_name, extension = os.path.splitext(file_name)
            file_name = f"{base_name}_{counter}{extension}"
            saved_path = os.path.join(root, file_name)
            counter += 1
        return saved_path

    @staticmethod
    def _save_data(data, saved_path):
        _, extension = os.path.splitext(saved_path)
        if extension == '.jpg':
            cv2.imwrite(saved_path, data)
        elif extension == '.npy':
            np.save(saved_path, data)
        elif extension == '.ply':
            o3d.io.write_point_cloud(saved_path, data)

    def save_source_data(self):
        image_path = self.source_metadata.image_path
        root = image_path['root']

        # metadata 저장
        json_file_name = "source_metadata.json"
        json_save_path = self._generate_unique_file_name(root, json_file_name)
        self.source_metadata.save_to_json(json_save_path)

        # 이미지 저장
        for key, data in self.source_image_data:
            file_name = image_path[key + '_file_name']
            saved_path = self._generate_unique_file_name(root, file_name)
            self._save_data(data, saved_path)

    def capture_target(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        # 1. 컬러 이미지
        image, _ = self.main_window.robot.robot_camera_manager.take_image()
        color_qimage = get_qimage(image)
        self.main_widget.lbl_col_hand_color_tgt.setPixmap(QPixmap.fromImage(color_qimage))

        # 2. Depth 이미지
        # 2-1) Depth 이미지 취득 시 Outlier 제거?
        iqr1 = int(self.main_widget.lbl_col_iqr1.text())
        iqr3 = int(self.main_widget.lbl_col_iqr3.text())
        iqr = [iqr1, iqr3]

        depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
            iqr=iqr,
            outlier_removal=self.main_widget.cbx_col_outlier_remove_iqr.isChecked()
        )

        # 2-2) Depth 이미지 누적
        self.data_accumulator_target.add_data(depth_image)
        self.main_widget.lbl_col_depth_acm_count_tgt.setText(str(self.data_accumulator_target.n_accumulate))
        # self.depth_image_source = self.data_accumulator_source.get_filtered_data(is_remove_outlier=False)
        threshold = float(self.main_widget.lbl_col_outlier_threshold.text())
        self.target_depth_data = self.data_accumulator_target.get_filtered_data(
            is_remove_outlier=self.main_widget.cbx_col_outlier_remove_threshold_tgt.isChecked(),
            threshold=threshold)
        self.target_depth_median = np.median(self.target_depth_data)
        if self.target_depth_median == 0:
            self.target_depth_median = np.mean(self.target_depth_data)

        self.main_widget.lbl_col_depth_median_tgt.setText(str(self.target_depth_median))

        # 3. 이미지 Display
        depth_color = self.main_window.robot.robot_camera_manager.depth_to_color(depth_image)
        color_in_depth_frame = self.main_window.robot.robot_camera_manager.take_image_from_source('hand_color_in_hand_depth_frame')
        color_in_depth_frame = cv2.rotate(color_in_depth_frame, cv2.ROTATE_90_CLOCKWISE)

        color_in_depth_frame_qimage = get_qimage(color_in_depth_frame)
        depth_color_qimage = get_qimage(depth_color)

        self.main_widget.lbl_col_depth_color_tgt.setPixmap(QPixmap.fromImage(depth_color_qimage))
        self.main_widget.lbl_col_hcidf_tgt.setPixmap(QPixmap.fromImage(color_in_depth_frame_qimage))

    def view_pcd_source(self, is_show=True):
        threshold = float(self.main_widget.lbl_col_outlier_threshold.text())
        self.source_depth_data = self.data_accumulator_source.get_filtered_data(
                                    is_remove_outlier=self.main_widget.cbx_col_outlier_remove_threshold.isChecked(),
                                    threshold=threshold)
        source_pcd = PointCloud(self.source_depth_data)
        if self.main_widget.cbx_col_outlier_remove_sor.isChecked():
            nb_neighbors = int(self.main_widget.lbl_col_nb_neightbors.text())
            std_ratio = float(self.main_widget.lbl_col_std_ratio.text())
            source_pcd = source_pcd.apply_sor_filter(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            source_pcd = source_pcd.pcd

        if is_show:
            o3d.visualization.draw_geometries([source_pcd],
                                              width=1440, height=968,
                                              left=50, top=50,
                                              front=[-0.58, -0.1, -0.8],
                                              lookat=[0, 0, 0.8],
                                              up=[0.0, -1, 0.17],
                                              zoom=0.64)

        self.source_pcd = source_pcd
        self.source_pcd.estimate_normals()

    def clear_pcd_source(self):
        self.data_accumulator_source.clear()
        self.data_accumulator_source = DepthAccumulator(buffer_size=100)
        self.main_widget.lbl_col_depth_acm_count.setText(str(self.data_accumulator_source.n_accumulate))
        self.main_widget.lbl_col_depth_median_src.setText(str(0))

    def setting_source(self):
        # Source 데이터를 먼저 설정.
        # 1. Navigate To "sd"
        # 2. Arm 위치 설정
        #   1) JSON 파일 로드
        #   2) "해당 위치로 이동" 버튼 클릭
        #   3) 이동이 완료되면, "Arm 위치 저장" 버튼 클릭 후 JSON 파일 저장.
        # 3. 세팅된 데이터 확인
        #   1) 컬러 이미지, depth_color, hand_color_in_depth_frame
        #   2) hand_frame 좌표
        #   3) 포인트 클라우드
        # * 포인트 클라우드 취득 시 누적 및 outlier 조절 기능.

        # Source Metadata 설정
        image_path = ImagePath(root=self.main_widget.lbl_save_path_src.text(),
                               hand_color_file_name="hand_color_source.jpg",
                               depth_color_file_name="depth_color_source.jpg",
                               hand_color_in_depth_frame_file_name="hand_color_in_depth_frame_source.jpg",
                               depth_data_file_name="depth_source.npy",
                               pointcloud_file_name="pointcloud_source.ply")
        self.source_metadata.set_image_path(image_path.image_path)
        pass

    def setting_source_(self):
        pass

    def start_data_correct(self):
        # 1. 목적지로 이동
        #   1) Navigate To "pb"
        #   2) Navigate To "sd"
        #   다른 웨이포인트를 들러 이동함으로써 로봇 위치 오차를 생성

        pass

    def prepare_robot(self):
        # 로봇 연결 체크
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return False

        return True

    def navigate_waypoints(self):
        # 웨이포인트를 들러 이동하는 코드를 여기에 작성하세요
        # 1. 다른 웨이포인트를 들러 이동.
        waypoint1 = self.main_widget.lbl_col_waypoint1_target.text()
        waypoint2 = self.main_widget.lbl_col_waypoint2_target.text()
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        # Robot Initialize
        graph_nav_manager.list_graph_waypoint_and_edge_ids()

        graph_nav_manager.navigate_to(waypoint1)
        graph_nav_manager.navigate_to(waypoint2)

    def stow(self):
        # 취득 끝나면 stow 후 다시 반복하는 코드를 여기에 작성하세요
        self.main_window.robot.robot_arm_manager.stow()

    def start_target_in_background(self, toggled):
        if toggled:
            if hasattr(self, 'collection_thread') and self.collection_thread.isRunning():
                self.main_widget.btn_col_start_target.setText("Target 데이터 취득 시작")
                self.collection_thread.set_stop_flag(True)
            else:
                self.collection_thread = DataCollectionThread(self)

                n = self.main_widget.sbx_col_iteration.value()
                self.collection_thread.iteration = n

                self.collection_thread.set_stop_flag(False)
                self.collection_thread.stopped.connect(self.function_on_thread_stop)

                self.collection_thread.start()
                self.main_widget.btn_col_start_target.setText("데이터 취득 중지")

                # runnable = StartTargetRunnable(self)
                # self.thread_pool.start(runnable)
        else:
            if hasattr(self, 'collection_thread') and self.collection_thread.isRunning():
                self.collection_thread.set_stop_flag(True)

    def function_on_thread_stop(self):
        self.main_widget.btn_col_start_target.setChecked(False)
        self.main_widget.btn_col_start_target.setText("Target 데이터 취득 시작")

    # Fiducial Setting
    def fiducial_page_init(self):
        self.main_widget.btn_load_arm_json_fid.clicked.connect(self.arm_json_load)
        self.main_widget.btn_save_arm_json_fid.clicked.connect(self.arm_json_save)
        self.main_widget.btn_find_nearest_fid.clicked.connect(self.find_nearest_fiducial)
        self.main_widget.btn_col_save_path_fid.clicked.connect(self.setting_save_path_fiducial)
        self.main_widget.btn_col_centering_nearest_fid.clicked.connect(self.centering_nearest_fiducial)
        self.main_widget.btn_col_move_arm_fid.clicked.connect(self.move_arm_with_fiducial)
        self.main_widget.btn_col_move_arm_manual.clicked.connect(self.move_arm_manual)
        self.main_widget.btn_col_capture_fid.clicked.connect(self.capture_with_fiducial)
        self.main_widget.btn_col_capture_and_save_fid.clicked.connect(self.capture_and_save)

    def arm_json_load(self):
        data = self.main_func.arm_json_load()
        if data:
            if 'frame_tform_gripper' not in data.keys():
                self.main_func.show_message_box('올바른 형식의 파일이 아닙니다.')
                return
            self.arm_data_with_fiducial = data

    def arm_json_save(self):
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Arm Status", "", "JSON Files (*.json)")
        if file_path:
            json_data = self.get_odom_tform_gripper()
            create_json_file(file_path, json_data)

    def setting_save_path_fiducial(self):
        folder = self.main_window.file_dialog.getExistingDirectory(self.main_window, "Select Directory")
        self.main_widget.lbl_save_path_fid.setText(folder)

    def find_nearest_fiducial(self):
        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        self.main_func.show_message_box(str(fiducial))

    def centering_nearest_fiducial(self):
        dist_margin = self.main_widget.sbx_col_dist_margin.value()
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
        dist_margin = round(self.main_widget.sbx_col_dist_margin.value(), 3)

        json_format = create_json_format(fid_id=fiducial_tag_id, dist_margin=dist_margin, data=data)
        self.arm_data_with_fiducial = json_format
        return json_format

    def move_arm_with_fiducial(self):
        if self.arm_data_with_fiducial is None:
            self.main_func.show_message_box('Arm 위치 설정이 되어있지 않습니다.')
            return

        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        odom_tform_fiducial_filtered = frame_helpers.get_a_tform_b(fiducial.transforms_snapshot,
                                                                   frame_helpers.ODOM_FRAME_NAME,
                                                                   fiducial.apriltag_properties.frame_name_fiducial_filtered)
        fiducial_tform_gripper = self.arm_data_with_fiducial['frame_tform_gripper'][1]['transform']
        fiducial_tform_gripper = dict_to_se3pose(fiducial_tform_gripper)
        odom_tform_gripper_goal = odom_tform_fiducial_filtered * fiducial_tform_gripper

        end_seconds = self.main_widget.sbx_col_manual_move_end_time.value()

        self.main_window.robot.robot_arm_manager.move_to_frame_hand(odom_tform_gripper_goal,
                                                                    frame_helpers.ODOM_FRAME_NAME,
                                                                    end_seconds=end_seconds)

    def move_arm_manual(self):
        axis = self.main_widget.cmb_col_manual_arm_axis.currentText()
        direction = self.main_widget.cmb_col_manual_arm_direction.currentText()
        joint_move_rate = self.main_widget.sbx_col_manual_move_rate.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        end_time_sec = self.main_widget.sbx_col_manual_move_end_time.value()
        self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, axis, direction,
                                                                   joint_move_rate, end_time_sec)

    def capture_with_fiducial(self):
        image, _ = self.main_window.robot.robot_camera_manager.take_image()
        color_qimage = get_qimage(image)
        self.main_widget.lbl_col_hand_color_fid.setPixmap(QPixmap.fromImage(color_qimage))

        return image

    def capture_and_save(self):
        image = self.capture_with_fiducial()
        # 이미지 저장
        file_path = self.main_widget.lbl_save_path_fid.text()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        file_name = f"captured_image_{current_time}.jpg"
        cv2.imwrite(os.path.join(file_path, file_name), image)


class SourceWidget:
    def __init__(self, parent):
        super().__init__(parent)


class DataCollectionThread(QThread):
    # Signal for progress
    progress = pyqtSignal()

    # Signal for completion
    stopped = pyqtSignal()

    def __init__(self, window):
        super().__init__()
        self.parent = window
        self.stop_flag = False
        self.iteration = 0

    def run(self):
        if self.parent.prepare_robot():  # 로봇 준비
            for _ in range(self.iteration):
                if self.stop_flag:
                    break
                # 1. 다른 웨이포인트를 들러 이동.
                self.parent.navigate_waypoints()
                # 2. Arm 위치 이동
                self.parent.move_arm("body")
                # 3. 데이터 취득
                self.parent.capture_target()
                # 4. 취득 끝나면 stow
                self.parent.stow()

        self.stopped.emit()

    def set_stop_flag(self, flag):
        self.stop_flag = flag

    def is_stopped(self):
        return self.stop_flag
