import time
from functools import partial

from control.Calculator import DepthAccumulator
from control.Control import MainFunctions
from control.utils.utils import *


class TabPage1:
    def __init__(self, main_widget, main_window, tab2_instance):
        self.main_widget = main_widget
        self.main_window = main_window
        self.main_func = MainFunctions(self.main_window)
        self.tab2 = tab2_instance

        self.iqr = None

        self.data_accumulator_source = DepthAccumulator(buffer_size=100)
        self.data_accumulator_target = DepthAccumulator(buffer_size=100)

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

        self.arm_position_input_source = dict()
        self.arm_position_input_target = dict()

        self.source_se3pose = None
        self.target_se3pose = None

    def init_signals(self):
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

    def capture_depth(self):
        iqr1 = int(self.main_widget.iqr1LineEdit.text())
        iqr3 = int(self.main_widget.iqr3LineEdit.text())
        self.iqr = [iqr1, iqr3]

        depth_image = self.main_window.robot.robot_camera_manager.get_depth_image(
            iqr=self.iqr,
            outlier_removal=self.main_widget.cbxOutlierRemove.isChecked()
        )
        return depth_image

    def capture(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        depth_image = self.capture_depth()
        depth_color = self.main_window.robot.robot_camera_manager.depth_to_color(depth_image)

        color = self.main_window.robot.robot_camera_manager.take_image_from_source('hand_color_in_hand_depth_frame')
        color_in_depth_frame = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)

        hand_color, data = self.main_window.robot.robot_camera_manager.take_image()

        return color_in_depth_frame, depth_color, depth_image, hand_color

    def set_images_source(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture()
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

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture()
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

    def capture_in_second(self, mode):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box('로봇 연결이 필요합니다.')
            return

        st = time.time()

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

            elif mode == 'target':
                self.data_accumulator_target.add_data(depth_data)

                # 촬영 시점의 로봇 좌표계 획득
                self.target_se3pose = self.main_window.robot.get_current_hand_position('hand')
                self.arm_position_input_target = {
                    "position": {axis: position[axis] for axis in ['x', 'y', 'z']},
                    "rotation": {axis: rotation[axis] for axis in ['w', 'x', 'y', 'z']}
                }
                self.odom_tform_hand_target = self.main_window.robot.get_odom_tform_hand()

        color_in_depth_frame, depth_color, depth_data, hand_color = self.capture()
        self.setting_capture_data(color_in_depth_frame, depth_color, hand_color, depth_data, mode=mode)

    def set_odom_position_data(self, odom_tform_hand_pose):
        position = {axis: getattr(odom_tform_hand_pose.position, axis) for axis in ['x', 'y', 'z']}
        rotation = {axis: getattr(odom_tform_hand_pose.rotation, axis) for axis in ['x', 'y', 'z', 'w']}

        set_position_and_rotation(self.main_widget, "manual_odom", position, rotation)

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
