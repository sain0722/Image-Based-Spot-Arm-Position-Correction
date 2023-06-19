import json
import os

import cv2
import numpy as np
import open3d as o3d
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from matplotlib import pyplot as plt

from control.Calculator import DepthAccumulator
from control.Control import MainFunctions
from control.PointCloud import PointCloud
from control.Tab.TabCompare.TabCompare_2 import IcpProcessor
from control.utils.arm_calculate_utils import *
from control.utils.utils import *
from spot.CarInspection.CarInspection import CarInspection


class TabCarInspection:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.car_inspection = CarInspection(self.main_window.robot)
        self.ai_keeper = AIKeeperWidget(self.main_window)
        self.init_signals()

    def init_signals(self):
        self.main_widget.btn_set_upload_filepath.clicked.connect(self.set_upload_filepath)
        self.main_widget.btn_go_to_inspection_waypoint.clicked.connect(self.go_to_inspection_waypoint)
        self.main_widget.btn_partial_inspection.clicked.connect(self.partial_inspection)
        self.main_widget.btn_full_inspection.clicked.connect(self.full_inspection)
        self.main_widget.btn_periodic_inspection.clicked.connect(self.periodic_inspection)

    def set_upload_filepath(self):
        folder = self.main_window.file_dialog.getExistingDirectory(self.main_window, "Select Directory")
        if folder:
            self.main_widget.lbl_upload_filepath.setText(folder)
            self.car_inspection.set_upload_filepath(folder)

    def go_to_inspection_waypoint(self):
        inspection_id = int(self.main_widget.lbl_inspection_id.text())
        self.car_inspection.go_to_inspection_waypoint(inspection_id)

    def partial_inspection(self):
        inspection_ids_str = self.main_widget.lbl_inspection_id_list.text()

        inspection_ids = inspection_ids_str.split()
        dock_at_the_end = self.main_widget.lbl_dock_at_the_end_partial.text()
        stow_in_between = self.main_widget.lbl_stow_in_between_partial.text()

        self.car_inspection.partial_inspection(inspection_ids, dock_at_the_end, stow_in_between)

    def full_inspection(self):
        dock_at_the_end = self.main_widget.lbl_dock_at_the_end_full.text()
        stow_in_between = self.main_widget.lbl_stow_in_between_full.text()

        self.car_inspection.full_inspection(dock_at_the_end, stow_in_between)

    def periodic_inspection(self):
        inspection_interval = float(self.main_widget.lbl_inspection_interval.text())
        number_of_cycles = int(self.main_widget.lbl_number_of_cycles.text())

        self.car_inspection.periodic_inspection(inspection_interval, number_of_cycles)


class AIKeeperWidget:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.graph_nav_manager = None

        # Images and Pointcloud
        keys = ["source", "target", "adjust"]
        self.hand_color = {key: None for key in keys}
        self.depth_image = {key: None for key in keys}
        self.depth_data = {key: None for key in keys}
        self.pointcloud = {key: None for key in keys}

        # Arm Position
        position_dict = {"position": None, "rotation": None}
        self.odom_tform_hand = {key: position_dict.copy() for key in keys}
        self.body_tform_hand = {key: position_dict.copy() for key in keys}

        self.source_pcd = None

        self.data_accumulator_target = DepthAccumulator(buffer_size=100)
        self.data_accumulator_adjust = DepthAccumulator(buffer_size=100)

        self.parameters = {'sor_used': True}

        self.init_signals()

    def init_signals(self):
        # self.main_widget.btn_initialize.clicked.connect(self.initialize)
        self.main_widget.btn_move_to_pid.clicked.connect(self.move_to_pid)
        self.main_widget.btn_centering_on_fid.clicked.connect(self.centering_on_target_fid)
        self.main_widget.btn_reach_to_pid.clicked.connect(self.reach_to_pid)
        self.main_widget.btn_take.clicked.connect(self.take)
        self.main_widget.btn_confirm_source.clicked.connect(self.confirm_source)
        self.main_widget.btn_adjust_and_take.clicked.connect(self.adjust_and_take)
        self.main_widget.btn_keeper_partial_inspection.clicked.connect(self.partial_inspection)
        self.main_widget.btn_keeper_full_inspection.clicked.connect(self.full_inspection)

    def initialize(self):
        self.graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        state, odom_tform_body = self.graph_nav_manager.get_localization_state()
        self.graph_nav_manager.list_graph_waypoint_and_edge_ids()

    def move_to_pid(self):
        if self.graph_nav_manager is None:
            self.initialize()

        picture_id = self.main_widget.lbl_move_picture_id.text()
        self.move_to_pid_with_parameter(picture_id)

    def move_to_pid_with_parameter(self, picture_id):
        if self.graph_nav_manager is None:
            self.initialize()

        file_path = "D:/TestSW/Source/230612/control/Tab/TabCarInspection/fiducial_to_pose.json"
        with open(file_path, "r") as f:
            fiducial_to_pose = json.load(f)

        inspection_pose = find_picture_id(fiducial_to_pose, picture_id)
        if inspection_pose is None:
            print("해당 Picture ID는 존재하지 않습니다.")
            return

        print(fiducial_to_pose)
        inspection_data = inspection_pose['inspection_data']
        navigation_data = inspection_pose['inspection_data']['navigation_data']

        waypoint = navigation_data['waypoint_name']
        self.graph_nav_manager.navigate_to(waypoint)

    def centering_on_nearest_fid(self, dist_margin=0.7):
        if self.graph_nav_manager is None:
            self.initialize()
        # fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        # dist_margin = self.main_widget.sbx_keeper_dist_margin.value()
        self.main_window.robot.robot_fiducial_manager.centering_on_nearest_fiducial(dist_margin=dist_margin)

    def centering_on_target_fid(self, picture_id=0, dist_margin=0.7):
        if self.graph_nav_manager is None:
            self.initialize()

        if picture_id == 0:
            picture_id = self.main_widget.lbl_move_picture_id.text()

        file_path = "D:/TestSW/Source/230612/control/Tab/TabCarInspection/fiducial_to_pose.json"
        with open(file_path, "r") as f:
            fiducial_to_pose = json.load(f)

        inspection_pose = find_picture_id(fiducial_to_pose, picture_id)
        fiducial_id = inspection_pose["fiducial_id"]
        self.main_window.robot.robot_fiducial_manager.centering_on_target_fiducial(fiducial_id, dist_margin)

    def reach_to_pid(self):
        if self.graph_nav_manager is None:
            self.initialize()
        picture_id = self.main_widget.lbl_reach_picture_id.text()
        self.reach_to_pid_with_parameter(picture_id)

    def reach_to_pid_with_parameter(self, picture_id):
        if self.graph_nav_manager is None:
            self.initialize()
        file_path = "D:/TestSW/Source/230612/control/Tab/TabCarInspection/fiducial_to_pose.json"
        with open(file_path, "r") as f:
            fiducial_to_pose = json.load(f)

        inspection_pose = find_picture_id(fiducial_to_pose, picture_id)
        if inspection_pose is None:
            print("해당 Picture ID는 존재하지 않습니다.")
            return

        print(fiducial_to_pose)
        inspection_data = inspection_pose['inspection_data']
        arm_pose_data = inspection_data['arm_pose']
        joint_params = arm_pose_data['joint_params']
        body_tform_hand_position = arm_pose_data['body_position']
        body_tform_hand_rotation = arm_pose_data['body_rotation']
        frame_name = BODY_FRAME_NAME
        self.main_window.robot.robot_arm_manager.trajectory(body_tform_hand_position, body_tform_hand_rotation, frame_name)

    def take(self):
        if self.graph_nav_manager is None:
            self.initialize()

        camera_manager = self.main_window.robot.robot_camera_manager
        image, _ = camera_manager.take_image()
        self.hand_color["target"] = image
        self.handle_depth_image("target")
        self.handle_arm_position("target")
        self.handle_pointcloud("target")

        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # self.stow()

    def stow(self):
        if self.graph_nav_manager is None:
            self.initialize()

        self.main_window.robot.robot_arm_manager.stow()

    def confirm_source(self):
        if self.graph_nav_manager is None:
            self.initialize()

        picture_id = self.main_widget.lbl_confirm_pid.text()
        self.confirm_source_with_parameter(picture_id)

    def confirm_source_with_parameter(self, picture_id):
        is_exist_pcd = False
        is_exist_img = False
        source_path = "D:/TestSW/Source/230612/keeper/arm_adjust/source/" + picture_id
        file_name = "source_pointcloud.ply"
        file_path = os.path.join(source_path, file_name)
        if os.path.exists(file_path):
            print("ply 파일 존재함.")
            source_points = o3d.io.read_point_cloud(file_path)
            self.pointcloud['source'] = source_points
            is_exist_pcd = True
        else:
            print("Source의 ply 파일 존재하지 않음.")

        hand_color_file_name = "source_hand_color.jpg"
        hand_color_file_path = os.path.join(source_path, hand_color_file_name)

        if os.path.exists(hand_color_file_path):
            print("hand_color 파일 존재함.")
            image = cv2.imread(hand_color_file_path)
            self.hand_color['source'] = image
            is_exist_img = True

        return is_exist_pcd, is_exist_img

    def adjust_and_take(self):
        if self.graph_nav_manager is None:
            self.initialize()

        self.spot_arm_correct()
        camera_manager = self.main_window.robot.robot_camera_manager
        image, _ = camera_manager.take_image()
        overlapped = overlap_images(self.hand_color["source"], image)
        # cv2.imshow("image", overlapped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # self.stow()

    def handle_depth_image(self, mode):
        accumulator = None

        if mode == "target":
            accumulator = self.data_accumulator_target
        elif mode in ["adjust", "adjust_2"]:
            accumulator = self.data_accumulator_adjust

        iteration = 16
        iqr_used = True
        iqr = [20, 80]
        self.depth_accumulate(accumulator, iteration, iqr, iqr_used)

        gaussian_used = True
        threshold = 3.0
        self.depth_data[mode] = get_depth_data(accumulator, threshold, gaussian_used)

    def depth_accumulate(self, accumulator, iteration, iqr, is_outlier_remove):
        # 2-1) Depth 이미지 취득 시 Outlier 제거?

        # 누적 n회
        accumulator.clear()
        for i in range(iteration):
            depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
                iqr=iqr,
                outlier_removal=is_outlier_remove
            )

            # 2-2) Depth 이미지 누적
            accumulator.add_data(depth_image)

    def handle_arm_position(self, mode):
        mode = mode if mode not in ["adjust", "adjust_2"] else "adjust"
        assert mode in ["source", "target", "adjust"], f"Invalid mode: {mode}"

        odom_position, odom_rotation = self.main_window.robot.get_odom_tform_hand_dict()
        body_position, body_rotation = self.main_window.robot.get_hand_position_dict()

        odom_tform_hand = {
            "position": odom_position,
            "rotation": odom_rotation
        }
        body_tform_hand = {
            "position": body_position,
            "rotation": body_rotation
        }

        self.odom_tform_hand[mode] = odom_tform_hand
        self.body_tform_hand[mode] = body_tform_hand

    def handle_pointcloud(self, mode):
        pcd = PointCloud(self.depth_data[mode])

        # 5. Pointcloud
        pcd = self.get_pointcloud(pcd)
        pcd.estimate_normals()

        self.pointcloud[mode] = pcd

    def get_pointcloud(self, pcd):
        if self.parameters['sor_used']:
            nb_neighbors = 20
            std_ratio = 2.0
            pcd = pcd.apply_sor_filter(nb_neighbors, std_ratio)
        else:
            pcd = pcd.pcd

        return pcd

    def spot_arm_correct(self):
        if self.run_icp() is None:
            return

        # 1. 보정값 산출
        transformation = self.icp_manager.get_transformation()
        corrected_target_pose = self.calculate_corrected_coord("target", transformation)

        # 2. 산출된 보정 위치로 이동
        self.move_arm_corrected(corrected_target_pose, end_time=1.0)

        # 3. 보정 위치에서 촬영
        self.handle_depth_image("adjust")
        self.handle_arm_position("adjust")
        self.handle_pointcloud("adjust")

        # 4. 보정 위치에서 촬영된 데이터로 2차 보정
        icp_iteration = 10
        icp_threshold = 0.02
        loss_sigma = 0.05
        corrected_icp_manager = IcpProcessor(self.pointcloud["source"], self.pointcloud["adjust"],
                                             icp_iteration, icp_threshold, loss_sigma)

        # self.icp_manager.set_threshold(self.parameters['icp_threshold'])
        corrected_icp_manager.run()
        fitness = corrected_icp_manager.icp.reg_p2l.fitness
        self.main_widget.lbl_icp_score_2.setText(str(fitness))

        transformation = corrected_icp_manager.get_transformation()
        corrected_target_pose = self.calculate_corrected_coord("adjust", transformation)

        # 5. 산출된 보정 위치로 이동
        self.move_arm_corrected(corrected_target_pose, end_time=0.3)

        # 6. 2차 보정 위치에서 촬영
        self.handle_depth_image("adjust_2")
        self.handle_arm_position("adjust_2")

    def run_icp(self):
        if self.pointcloud["source"] is None:
            self.main_func.show_message_box("원본 데이터가 설정되지 않았습니다.")
            return

        if self.pointcloud["target"] is None:
            self.main_func.show_message_box("타겟 데이터가 설정되지 않았습니다.")
            return

        self.run_spot_arm_correct_with_icp(self.pointcloud['source'], self.pointcloud['target'])
        return True

    def run_spot_arm_correct_with_icp(self, source, target):
        icp_iteration = 10
        icp_threshold = 0.02
        loss_sigma = 0.05
        self.icp_manager = IcpProcessor(source, target,
                                        icp_iteration, icp_threshold, loss_sigma)

        self.icp_manager.run()
        # fitness = self.icp_manager.icp.reg_p2l.fitness
        # self.main_widget.lbl_icp_score.setText(str(fitness))

    def calculate_corrected_coord(self, mode, transformation):
        target_pose = convert_to_target_pose(self.body_tform_hand[mode])

        transformation_matrix = apply_spot_coordinate_matrix(transformation)
        corrected_target_pose = apply_transformation_to_target(transformation_matrix, target_pose)
        return corrected_target_pose

    def move_arm_corrected(self, corrected_target_pose, end_time):
        position = {
            'x': corrected_target_pose['x'],
            'y': corrected_target_pose['y'],
            'z': corrected_target_pose['z']
        }

        rotation = corrected_target_pose['rotation']
        frame_name = BODY_FRAME_NAME
        self.main_window.robot.robot_arm_manager.trajectory(position, rotation, frame_name, end_time)

    def partial_inspection(self):
        # 1. Initialize
        self.initialize()

        picture_id = self.main_widget.lbl_move_picture_id.text()
        # 2. Move To Picture ID
        self.move_to_pid_with_parameter(picture_id)

        # 3. centering on fiducial
        if picture_id != "20603":
            dist_margin = 0.7
            self.centering_on_target_fid(dist_margin)

        # 4. Reach To Picture ID
        self.reach_to_pid_with_parameter(picture_id)

        # 5. Take Picture
        self.take()

        # 6. Confirm Source Pointcloud
        is_pcd, is_img = self.confirm_source_with_parameter(picture_id)
        if not is_pcd:
            return

        # 7. Run ICP
        # self.spot_arm_correct()
        self.adjust_and_take()
        camera_manager = self.main_window.robot.robot_camera_manager
        image, _ = camera_manager.take_image()
        overlapped = overlap_images(self.hand_color["source"], image)
        cv2.imwrite(picture_id + "_overlap.jpg", overlapped)

        overlapped_target = overlap_images(self.hand_color["source"], self.hand_color["target"])
        cv2.imwrite(picture_id + "_overlap_target.jpg", overlapped_target)

        # 8. Stow when completed.
        self.stow()

    def full_inspection(self):
        picture_ids = ["20201", "20102", "20103", "20104", "20401", "20501", "20603"]

        # 1. Initialize
        self.initialize()
        for picture_id in picture_ids:
            # 2. Move To Picture ID
            self.move_to_pid_with_parameter(picture_id)

            # 3. centering on fiducial
            if picture_id != "20603":
                dist_margin = 0.7
                self.centering_on_target_fid(picture_id, dist_margin)

            # 4. Reach To Picture ID
            self.reach_to_pid_with_parameter(picture_id)

            # 5. Take Picture
            self.take()

            # 6. Confirm Source Pointcloud
            is_pcd, is_img = self.confirm_source_with_parameter(picture_id)
            if not is_pcd:
                return

            # 7. Run ICP
            # self.spot_arm_correct()
            self.adjust_and_take()
            camera_manager = self.main_window.robot.robot_camera_manager
            image, _ = camera_manager.take_image()
            overlapped = overlap_images(self.hand_color["source"], image)
            cv2.imwrite(picture_id + "_overlap.jpg", overlapped)

            overlapped_target = overlap_images(self.hand_color["source"], self.hand_color["target"])
            cv2.imwrite(picture_id + "_overlap_target.jpg", overlapped_target)

            # 8. Stow when completed.
            self.stow()


def find_picture_id(data, user_input):
    for item in data:
        if item['inspection_data']['picture_id'] == user_input:
            return item
    return None  # 입력한 picture_id와 일치하는 값이 없는 경우


def get_depth_data(accumulator, threshold, is_outlier_remove):
    # 누적된 depth 데이터 획득
    # 이 과정에서 outlier를 제거할 지 안할지 선택.
    depth_data = accumulator.get_filtered_data(
        is_remove_outlier=is_outlier_remove,
        threshold=threshold)

    return depth_data
