import copy

from PyQt5.QtWidgets import QDialog
from bosdyn.client.frame_helpers import BODY_FRAME_NAME

from control.Calculator import DepthAccumulator
from control.Control import MainFunctions
from control.PointCloud import PointCloud, ICP, execute_surf
from control.Tab.TabCompare.DialogSetParameter import ParameterDialog
from control.WidgetArmMove import WidgetArmMove
from control.utils.arm_calculate_utils import apply_spot_coordinate_matrix, apply_transformation_to_target
from control.utils.utils import *

import open3d as o3d


class TabCompare:
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.data_accumulator_source = DepthAccumulator(buffer_size=100)
        self.data_accumulator_target = DepthAccumulator(buffer_size=100)
        self.data_accumulator_corrected = DepthAccumulator(buffer_size=100)

        # Images and Pointcloud
        keys = ["source", "target", "corrected"]
        self.hand_color = {key: None for key in keys}
        self.depth_image = {key: None for key in keys}
        self.depth_data = {key: None for key in keys}
        self.pointcloud = {key: None for key in keys}

        # Arm Position
        position_dict = {"position": None, "rotation": None}
        self.odom_tform_hand = {key: position_dict.copy() for key in keys}
        self.body_tform_hand = {key: position_dict.copy() for key in keys}

        self.icp_manager = None
        self.parameters = None
        self.init_signals()

        self.arm_widget = WidgetArmMove(main_window)

    def init_signals(self):
        self.main_widget.btn_main_capture.clicked.connect(lambda: self.capture("source"))
        self.main_widget.btn_main_capture_before.clicked.connect(self.capture_before)
        # self.main_widget.btn_main_capture_after.clicked.connect(lambda: self.capture("corrected"))

        self.main_widget.btn_main_setting_param.clicked.connect(self.setting_params)

        self.main_widget.btn_main_icp.clicked.connect(self.run_icp)

        self.main_widget.btn_main_move_arm_corrected.clicked.connect(self.spot_arm_correct)

    def setting_params(self):
        dialog = ParameterDialog()
        if self.parameters is not None:
            dialog.sbx_depth_acm_count.setValue(self.parameters['depth_acm_count'])

            dialog.cbx_iqr.setChecked(self.parameters['iqr_used'])
            dialog.sbx_iqr1.setValue(self.parameters['iqr1'])
            dialog.sbx_iqr3.setValue(self.parameters['iqr3'])

            dialog.cbx_gaussian.setChecked(self.parameters['gaussian_used'])
            dialog.sbx_gaussian_threshold.setValue(self.parameters['gaussian_threshold'])

            dialog.cbx_sor_filter.setChecked(self.parameters['sor_used'])
            dialog.sbx_nb_neighbors.setValue(self.parameters['nb_neighbors'])
            dialog.sbx_std_ratio.setValue(self.parameters['std_ratio'])

            dialog.sbx_icp_iteration.setValue(self.parameters['icp_iteration'])
            dialog.sbx_icp_threshold.setValue(self.parameters['icp_threshold'])
            dialog.sbx_loss_sigma.setValue(self.parameters['loss_sigma'])

        if dialog.exec():
            self.parameters = {
                'depth_acm_count': dialog.sbx_depth_acm_count.value(),
                'iqr_used': dialog.cbx_iqr.isChecked(),
                'iqr1': dialog.sbx_iqr1.value(),
                'iqr3': dialog.sbx_iqr3.value(),
                'gaussian_used': dialog.cbx_gaussian.isChecked(),
                'gaussian_threshold': dialog.sbx_gaussian_threshold.value(),
                'sor_used': dialog.cbx_sor_filter.isChecked(),
                'nb_neighbors': dialog.sbx_nb_neighbors.value(),
                'std_ratio': dialog.sbx_std_ratio.value(),
                'icp_iteration': dialog.sbx_icp_iteration.value(),
                'icp_threshold': dialog.sbx_icp_threshold.value(),
                'loss_sigma': dialog.sbx_loss_sigma.value()
            }

            print(self.parameters)  # you can use this parameter dictionary as per your need

    def handle_color_image(self, mode):
        # 1. 컬러 이미지
        hand_color, _ = self.main_window.robot.robot_camera_manager.take_image()
        self.hand_color[mode] = hand_color

        hand_color_qimage = get_qimage(hand_color)
        qpixmap = QPixmap.fromImage(hand_color_qimage)
        # mode = mode if mode not in ["corrected", "corrected_2"] else "corrected"
        label = getattr(self.main_widget, f"lbl_hand_color_{mode}")
        set_pixmap(label, qpixmap)

    def handle_depth_image(self, mode):
        accumulator = None

        if mode == "source":
            accumulator = self.data_accumulator_source
        elif mode == "target":
            accumulator = self.data_accumulator_target
        elif mode in ["corrected", "corrected_2"]:
            accumulator = self.data_accumulator_corrected

        iteration = self.parameters['depth_acm_count']
        iqr_used = self.parameters['iqr_used']
        iqr = [self.parameters['iqr1'], self.parameters['iqr3']]
        self.depth_accumulate(accumulator, iteration, iqr, iqr_used)

        gaussian_used = self.parameters['gaussian_used']
        threshold = self.parameters['gaussian_threshold']
        self.depth_data[mode] = self.get_depth_data(accumulator, threshold, gaussian_used)

    def handle_arm_position(self, mode):
        mode = mode if mode not in ["corrected", "corrected_2"] else "corrected"
        assert mode in ["source", "target", "corrected"], f"Invalid mode: {mode}"

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

    def capture(self, mode):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        if self.parameters is None:
            self.main_func.show_message_box("파라미터를 설정하고 시도하세요.")
            return

        # 1. 컬러 이미지
        self.handle_color_image(mode)

        # 2. Depth 이미지
        self.handle_depth_image(mode)

        # 3. Arm 포지션
        self.handle_arm_position(mode)

        # 4. Pointcloud로 변환
        self.handle_pointcloud(mode)

        if mode == "source":
            return

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

    @staticmethod
    def get_depth_data(accumulator, threshold, is_outlier_remove):
        # 누적된 depth 데이터 획득
        # 이 과정에서 outlier를 제거할 지 안할지 선택.
        depth_data = accumulator.get_filtered_data(
            is_remove_outlier=is_outlier_remove,
            threshold=threshold)

        return depth_data

    def get_pointcloud(self, pcd):
        if self.parameters['sor_used']:
            nb_neighbors = self.parameters['nb_neighbors']
            std_ratio = self.parameters['std_ratio']
            pcd = pcd.apply_sor_filter(nb_neighbors, std_ratio)
        else:
            pcd = pcd.pcd

        return pcd

    def handle_pointcloud(self, mode):
        pcd = PointCloud(self.depth_data[mode])

        # 5. Pointcloud
        pcd = self.get_pointcloud(pcd)
        pcd.estimate_normals()

        self.pointcloud[mode] = pcd

        pointcloud_show_and_save(pcd, filename=mode)

        pcd_image = cv2.imread(f"{mode}.png")
        pcd_qimage = get_qimage(pcd_image)

        # QPixmap을 리사이즈합니다.
        qpixmap = QPixmap.fromImage(pcd_qimage)
        label = getattr(self.main_widget, f"lbl_pcd_{mode}")
        set_pixmap(label, qpixmap)

        if mode == "target":
            self.handle_surf()

            # Source, Target overlap 이미지
            hand_color_overlapped = overlap_images(self.hand_color["source"], self.hand_color["target"])
            hand_color_overlapped_qimage = get_qimage(hand_color_overlapped)
            hand_color_overlapped_qpixmap = QPixmap.fromImage(hand_color_overlapped_qimage)
            set_pixmap(self.main_widget.lbl_color_overlap_target, hand_color_overlapped_qpixmap)

            # Source, Target Pointcloud overlap 이미지
            pointcloud_compare_show_and_save(self.pointcloud["source"], self.pointcloud["target"], filename="compare")
            compare_pcd_image = cv2.imread("compare.png")
            compare_pcd_qimage = get_qimage(compare_pcd_image)
            compare_pcd_qpixmap = QPixmap.fromImage(compare_pcd_qimage)
            set_pixmap(self.main_widget.lbl_pcd_overlap_target, compare_pcd_qpixmap)

        elif mode == "corrected":
            # 원본 + 보정 (RGB) overlap 이미지
            hand_color_overlapped = overlap_images(self.hand_color["source"], self.hand_color["corrected"])
            hand_color_overlapped_qimage = get_qimage(hand_color_overlapped)
            hand_color_overlapped_qpixmap = QPixmap.fromImage(hand_color_overlapped_qimage)
            set_pixmap(self.main_widget.lbl_color_overlap_corrected, hand_color_overlapped_qpixmap)

            # Source, Target Pointcloud overlap 이미지
            pointcloud_compare_show_and_save(self.pointcloud["source"], self.pointcloud["corrected"], "compare_corrected")
            compare_pcd_image = cv2.imread("compare_corrected.png")
            compare_pcd_qimage = get_qimage(compare_pcd_image)
            compare_pcd_qpixmap = QPixmap.fromImage(compare_pcd_qimage)
            set_pixmap(self.main_widget.lbl_pcd_overlap_corrected, compare_pcd_qpixmap)

        elif mode == "corrected_2":
            # 원본 + 보정 (RGB) overlap 이미지
            hand_color_overlapped = overlap_images(self.hand_color["source"], self.hand_color["corrected_2"])
            hand_color_overlapped_qimage = get_qimage(hand_color_overlapped)
            hand_color_overlapped_qpixmap = QPixmap.fromImage(hand_color_overlapped_qimage)
            set_pixmap(self.main_widget.lbl_color_overlap_corrected_2, hand_color_overlapped_qpixmap)

            # Source, Target Pointcloud overlap 이미지
            pointcloud_compare_show_and_save(self.pointcloud["source"], self.pointcloud["corrected_2"], "compare_corrected_2")
            compare_pcd_image = cv2.imread("compare_corrected_2.png")
            compare_pcd_qimage = get_qimage(compare_pcd_image)
            compare_pcd_qpixmap = QPixmap.fromImage(compare_pcd_qimage)
            set_pixmap(self.main_widget.lbl_pcd_overlap_corrected_2, compare_pcd_qpixmap)

    def handle_surf(self):
        # SURF 객체
        surf = SurfProcessor(self.hand_color["source"], self.hand_color["target"])

        # Source, Target SURF 결과 이미지
        matrix, surf_result_image = surf.run()
        if matrix is None:
            print("SURF 정합 실패.")
            self.main_widget.lbl_surf_target.setText("SURF Fail")
            self.main_widget.lbl_surf_overlap_target.setText("SURF Fail")
        else:
            surf_result_qimage = get_qimage(surf_result_image)
            surf_result_qpixmap = QPixmap.fromImage(surf_result_qimage)
            set_pixmap(self.main_widget.lbl_surf_target, surf_result_qpixmap)

            # SURF 결과 이미지 + Source overlap 이미지
            overlapped = overlap_images(surf_result_image, self.hand_color["source"])
            overlapped_qimage = get_qimage(overlapped)
            overlapped_qpixmap = QPixmap.fromImage(overlapped_qimage)
            set_pixmap(self.main_widget.lbl_surf_overlap_target, overlapped_qpixmap)

    def run_icp(self):
        if self.parameters is None:
            self.main_func.show_message_box("파라미터를 설정하고 시도하세요.")
            return

        if self.pointcloud["source"] is None:
            self.main_func.show_message_box("원본 데이터가 설정되지 않았습니다.")
            return

        if self.pointcloud["target"] is None:
            self.main_func.show_message_box("타겟 데이터가 설정되지 않았습니다.")
            return

        self.run_spot_arm_correct_with_icp()
        self.show_before_correct()

    def run_spot_arm_correct_with_icp(self):
        self.icp_manager = IcpProcessor(self.pointcloud["source"], self.pointcloud["target"],
                                        self.parameters['icp_iteration'], self.parameters['icp_threshold'],
                                        self.parameters['loss_sigma'])
        self.icp_manager.run()
        fitness = self.icp_manager.icp.reg_p2l.fitness
        self.main_widget.lbl_main_icp_fitness.setText(str(fitness))

    def show_before_correct(self):
        # ICP 결과 + Source Pointcloud overlap 이미지
        target = copy.deepcopy(self.pointcloud["target"])
        transformation = self.icp_manager.get_transformation()
        transformation = np.linalg.inv(transformation)
        target.transform(transformation)
        pointcloud_compare_show_and_save(self.pointcloud["source"], target, filename="compare_overlap")
        compare_overlap_image = cv2.imread("compare_overlap.png")
        compare_overlap_qimage = get_qimage(compare_overlap_image)
        compare_overlap_qpixmap = QPixmap.fromImage(compare_overlap_qimage)
        set_pixmap(self.main_widget.lbl_pcd_icp, compare_overlap_qpixmap)

        # (미구현) Source, ICP 결과 Color 이미지
        # ICP 결과를 통해 2D 이미지를 변환 (?)
        # self.main_widget.lbl_overlap_icp

    def capture_before(self):
        self.capture("target")
        self.run_icp()

    def spot_arm_correct(self):
        # 1. 보정값 산출
        corrected_target_pose = self.calculate_corrected_coord("target")

        # 2. 산출된 보정 위치로 이동
        self.move_arm_corrected(corrected_target_pose, end_time=1.0)

        # 3. 보정 위치에서 촬영
        self.capture("corrected")

        # 4. 보정 위치에서 촬영된 데이터로 2차 보정
        self.icp_manager.set_target(self.pointcloud["corrected"])
        self.icp_manager.set_transformation(np.eye(4))  # 2차 보정 시 초기행렬을 항등행렬로 재설정.
        self.icp_manager.run()
        fitness = self.icp_manager.icp.reg_p2l.fitness
        self.main_widget.lbl_main_icp_fitness_corrected.setText(str(fitness))
        corrected_target_pose = self.calculate_corrected_coord("corrected")

        # 5. 산출된 보정 위치로 이동
        self.move_arm_corrected(corrected_target_pose, end_time=0.3)

        # 6. 2차 보정 위치에서 촬영
        self.capture("corrected_2")

    def calculate_corrected_coord(self, mode):
        transformation = self.icp_manager.get_transformation()
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


def set_pixmap(label, qpixmap):
    qpixmap_resized = qpixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
    label.setPixmap(qpixmap_resized)


def pointcloud_show_and_save(pcd, filename):
    pcd = copy.deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # x축 주위로 180도 회전
    pcd.rotate(R, center=(0, 0, 0))

    # Visualizer를 생성합니다.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900, visible=False)  # `draw_geometries`에서 사용한 창 크기를 설정합니다.

    # Point Cloud를 추가합니다.
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_lookat([0.04, 0.048, -1.728])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.54)

    # 렌더링을 업데이트합니다.
    vis.poll_events()
    vis.update_renderer()

    # 화면을 캡쳐해서 파일로 저장합니다.
    vis.capture_screen_image(f"{filename}.png")
    # image = vis.capture_screen_float_buffer()

    vis.destroy_window()


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


def pointcloud_compare_show_and_save(pcd1, pcd2, filename):
    pcd1 = copy.deepcopy(pcd1)
    pcd2 = copy.deepcopy(pcd2)

    R1 = pcd1.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # x축 주위로 180도 회전
    pcd1.rotate(R1, center=(0, 0, 0))

    R2 = pcd2.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # x축 주위로 180도 회전
    pcd2.rotate(R2, center=(0, 0, 0))

    # Visualizer를 생성합니다.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900, visible=False)  # `draw_geometries`에서 사용한 창 크기를 설정합니다.
    # Point Cloud들의 색상을 변경합니다. RGB 값을 [0,1] 범위로 입력해야합니다.
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])

    # Point Cloud를 추가합니다.
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    # 렌더링을 업데이트합니다.
    vis.poll_events()
    vis.update_renderer()

    # 화면을 캡쳐해서 파일로 저장합니다.
    vis.capture_screen_image(f"{filename}.png")
    # image = vis.capture_screen_float_buffer()

    vis.destroy_window()


class IcpProcessor:
    def __init__(self, source, target, iteration, threshold, loss_sigma):
        self.source = source
        self.target = target
        self.iteration = iteration
        self.threshold = threshold
        self.loss_sigma = loss_sigma

        self.icp = ICP(self.source, self.target)
        self.running_count = 0

    def run(self):
        # ICP 알고리즘 실행
        self.icp.robust_icp(iteration=self.iteration, threshold=self.threshold, sigma=self.loss_sigma)
        self.running_count += 1

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_source(self, source):
        self.source = source
        self.icp.set_source(source)

    def set_target(self, target):
        self.target = target
        self.icp.set_target(target)

    def get_transformation(self):
        return self.icp.transformation_buffer[self.running_count - 1]

    def set_transformation(self, transformation):
        self.icp.set_init_transformation(transformation)

class SurfProcessor:
    def __init__(self, source, target, ratio_threshold=0.3):
        self.source = source
        self.target = target
        self.ratio_threshold = ratio_threshold

    def run(self):
        matrix, surf_result_image = execute_surf(self.source, self.target, self.ratio_threshold)
        return matrix, surf_result_image


