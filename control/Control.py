import copy
import time
import traceback
from datetime import datetime

from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtWidgets import *
from bosdyn.client.frame_helpers import BODY_FRAME_NAME

from .PointCloud import ICP, execute_surf, get_trans_init
from control.utils.arm_calculate_utils import calculate_new_rotation
from control.utils.utils import *
import PyQt5.QtCore as QtCore


class MainFunctions:
    def __init__(self, main_window):
        self.main_window = main_window
        self.saved_folder = 'TEST_IMAGE/{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day)

    def dialog_get_image_path(self):
        _filter = "images (*.jpg *.png *.bmp)"
        fname = self.main_window.file_dialog.getOpenFileName(None, filter=_filter)[0]

        if not fname:
            return False

        return fname

    def arm_json_load(self):
        # 다이얼로그를 통해 JSON 파일 선택
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("JSON Files (*.json)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
        return

    def load_image(self, label: QLabel, is_draw_center_line: bool = False):
        fname: str = self.dialog_get_image_path()
        if not fname:
            return
        set_pixmap(fname, label, is_draw_center_line)

    def save_image(self, image):
        now = datetime.now()
        extension = "jpg"
        image_name = 'image'
        saved_name = '{}-{}-{}_{}-{}-{}-{}.{}'.format(now.year, now.month, now.day,
                                                      now.hour, now.minute, now.second, image_name, extension)

        if not os.path.exists(self.saved_folder):
            os.makedirs(self.saved_folder, exist_ok=True)

        saved_path = os.path.join(self.saved_folder, saved_name)
        cv2.imwrite(saved_path, image)

        self.show_message_box(f"{saved_name} 저장 완료")

    @staticmethod
    def clear_label(label: QLabel):
        label.setText(" ")

    def show_message_box(self, contents: str):
        self.msgbox = QMessageBox()
        self.msgbox.information(self.main_window, "알림", contents, QMessageBox.Ok)

    def hide_message_box_after_delay(self, delay):
        timer = QTimer(self.main_window)
        timer.timeout.connect(self.hide_message_box)
        timer.setSingleShot(True)
        timer.start(delay)

    def hide_message_box(self):
        if self.msgbox is not None:
            self.msgbox.hide()
            self.msgbox = None


class ThreadWorker(QThread):
    # Signal for progress
    progress = QtCore.pyqtSignal()

    # Signal for completion
    stopped = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_flag = False

    def run(self):
        while not self._stop_flag:
            self.progress.emit()
            time.sleep(0.15)

        self.stopped.emit()

    def set_stop_flag(self, flag):
        self._stop_flag = flag


class ThreadButtonGenerator(QThread):
    # Signal for progress
    button_generation = pyqtSignal(int)

    def __init__(self, iteration):
        super().__init__()
        self.iteration = iteration

    def run(self):
        self.button_generation.emit(self.iteration)


class TrajectoryRun(QThread):
    finished = pyqtSignal()

    def __init__(self, trajectory_pos_rot, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w):
        super().__init__()
        self.trajectory_pos_rot = trajectory_pos_rot
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.rot_w = rot_w

    def run(self):
        self.trajectory_pos_rot(self.pos_x, self.pos_y, self.pos_z, self.rot_x, self.rot_y, self.rot_z, self.rot_w)
        self.finished.emit()


class TrajectoryWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, main_window, main_widget, tab2_instance, parent=None):
        super().__init__(parent)
        self.new_rotation = None
        self.main_window = main_window
        self.main_widget = main_widget
        self.tab2 = tab2_instance
        self.stop_flag = False

        self.translation_axis = None
        self.interval = 0
        self.iteration = 0
        self.rotation_axis = None
        self.angle = 0

        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.rot_x = 0
        self.rot_y = 0
        self.rot_z = 0
        self.rot_w = 0

        self.trajectory_position = None

    def run(self):
        try:
            self.set_initial_position()
            self.capture_source()
            # 2. Target 설정
            # - 설정된 값에 따라 Target 데이터를 취득 및 저장(메모리)
            # - 1) cmb_pos_axis: 데이터 경향성을 파악할 축을 선택 (x, y, z)
            # - 2) spb_interval: 반복으로 이동할 간격
            # - 3) spb_distance: arm이 움직일 최종 위치
            self.set_parameters()

            # 3. 반복 진행
            # Source: x = 0.8, y = 0, z = 0.2
            # y축, interval = 0.01, distance = 0.1 인 경우
            # 10회 (distance / interval) 반복
            # x = 0.8, y = 0.01, z = 0.2
            # x = 0.8, y = 0.02, z = 0.2

            self.set_icp()

            print("축: ", self.translation_axis)
            for i in range(self.iteration):
                if self.stop_flag:  # 중단 기능
                    break

                self.move_arm()
                self.capture_and_save(i)
                self.run_icp(i)

            if self.stop_flag:
                self.finished.emit()
                return

            event = GenerateButtonsEvent(self.iteration)
            QCoreApplication.postEvent(self.main_window, event)

            # 로봇 보정 페이지 UI 적용 (Source)
            positions = ['x', 'y', 'z']
            rotations = ['x', 'y', 'z', 'w']
            if self.main_widget.cbx_odometry.isChecked():
                source_data_position = self.tab2.page5.source_data['odom_tform_hand']['position']
                source_data_rotation = self.tab2.page5.source_data['odom_tform_hand']['rotation']
            else:
                source_data_position = self.tab2.page5.source_data['arm_position_real']['position']
                source_data_rotation = self.tab2.page5.source_data['arm_position_real']['rotation']

            for position in positions:
                value = str(round(source_data_position[position], 4))
                getattr(self.main_widget, f"lbl_source_pos_{position}").setText(value)

            for rotation in rotations:
                value = str(round(source_data_rotation[rotation], 6))
                getattr(self.main_widget, f"lbl_source_rot_{rotation}").setText(value)

        except Exception as e:
            print("예외 발생:", e)
            traceback.print_exc()
        finally:
            self.finished.emit()

    def set_initial_position(self):
        # 기존에 있는 버튼 제거
        for i in reversed(range(self.main_widget.gridLayout_icp_btns.count())):
            self.main_widget.gridLayout_icp_btns.itemAt(i).widget().setParent(None)

        if self.main_widget.cbx_unstow.isChecked():
            self.main_window.robot.robot_arm_manager.unstow()
            position, rotation = self.main_window.robot.get_hand_position_dict()
            self.pos_x = position['x']
            self.pos_y = position['y']
            self.pos_z = position['z']
            self.rot_x = rotation['x']
            self.rot_y = rotation['y']
            self.rot_z = rotation['z']
            self.rot_w = rotation['w']
            self.frame_name = BODY_FRAME_NAME
            self.trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_manual

            time.sleep(1.5)
        else:
            if self.main_widget.cbx_odometry.isChecked():
                self.pos_x = float(self.main_widget.lbl_odom_pos_x_src.text())
                self.pos_y = float(self.main_widget.lbl_odom_pos_y_src.text())
                self.pos_z = float(self.main_widget.lbl_odom_pos_z_src.text())
                self.rot_x = float(self.main_widget.lbl_odom_rot_x_src.text())
                self.rot_y = float(self.main_widget.lbl_odom_rot_y_src.text())
                self.rot_z = float(self.main_widget.lbl_odom_rot_z_src.text())
                self.rot_w = float(self.main_widget.lbl_odom_rot_w_src.text())
                self.trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_odometry
            else:
                self.pos_x = float(self.main_widget.lbl_pos_x_src.text())
                self.pos_y = float(self.main_widget.lbl_pos_y_src.text())
                self.pos_z = float(self.main_widget.lbl_pos_z_src.text())
                self.rot_x = float(self.main_widget.lbl_rot_x_src.text())
                self.rot_y = float(self.main_widget.lbl_rot_y_src.text())
                self.rot_z = float(self.main_widget.lbl_rot_z_src.text())
                self.rot_w = float(self.main_widget.lbl_rot_w_src.text())
                self.trajectory_function = self.main_window.robot.robot_arm_manager.trajectory_pos_rot

            self.trajectory_source()

    def trajectory_source(self):
        source_trajectory = TrajectoryRun(self.trajectory_function,
                                          self.pos_x, self.pos_y, self.pos_z,
                                          self.rot_x, self.rot_y, self.rot_z, self.rot_w)
        source_trajectory.start()
        source_trajectory.wait()

    def capture_source(self):
        self.tab2.page5.clear_pcd_source()
        self.tab2.page5.capture_in_second(mode="source")
        self.tab2.page5.set_source_data()
        self.tab2.page5.save_source_data()

        color_qimage = get_qimage(self.tab2.page5.source_data['hand_color_in_depth_frame'])
        depth_color_qimage = get_qimage(self.tab2.page5.source_data['hand_depth_image'])

        self.main_widget.lbl_hand_color_src.setPixmap(QPixmap.fromImage(color_qimage))
        self.main_widget.lbl_hand_depth_src.setPixmap(QPixmap.fromImage(depth_color_qimage))
        print("이미지 설정 완료")

    def set_parameters(self):
        if self.is_translation():
            self.translation_axis = self.main_widget.cmb_pos_axis.currentText()
            self.interval = self.main_widget.spb_interval.value()
            self.iteration = self.main_widget.spb_distance.value()
        else:
            self.rotation_axis = self.main_widget.cmb_rot_axis.currentText()
            self.angle = self.main_widget.spb_rot_angle.value()
            self.iteration = self.main_widget.spb_rot_iteration.value()

    def is_translation(self):
        return self.main_widget.rbn_translation.isChecked()

    def is_rotation(self):
        return self.main_widget.rbn_rotation.isChecked()

    def set_icp(self):
        self.tab2.page5.icp = ICP(self.tab2.page5.source_pcd, self.tab2.page5.target_pcd)
        self.tab2.page5.target_data_buffer.clear()

    def move_arm(self):
        self.trajectory_position = {
            "x": self.pos_x,
            "y": self.pos_y,
            "z": self.pos_z
        }

        if self.is_translation():
            self.trajectory_position[self.translation_axis] += self.interval

        # print(f"{i + 1}\t pos {self.translation_axis}: ", self.trajectory_position[self.translation_axis])
        print("시작: ", datetime.now())

        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')

        if self.is_translation():
            # 로봇 Arm 이동
            if self.main_widget.cbx_unstow.isChecked():
                direction = "up"
                joint_move_rate = self.interval
                end_time_sec = 1.5
                self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, self.translation_axis,
                                                                           direction, joint_move_rate, end_time_sec)
            else:
                target_trajectory = TrajectoryRun(self.trajectory_function,
                                                  self.trajectory_position["x"],
                                                  self.trajectory_position["y"],
                                                  self.trajectory_position["z"],
                                                  self.rot_x, self.rot_y, self.rot_z, self.rot_w)

                target_trajectory.start()
                target_trajectory.wait()
        else:
            body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
            new_rotation = calculate_new_rotation(self.rotation_axis, self.angle, body_tform_hand.rotation)
            self.new_rotation = new_rotation
            self.main_window.robot.robot_arm_manager.trajectory_rotation_manual(body_tform_hand, new_rotation)

    def capture_and_save(self, idx):
        # i번째 Target 데이터 획득 및 저장
        self.tab2.page5.clear_pcd_target()
        self.tab2.page5.capture_in_second(mode="target")

        axis = self.rotation_axis if self.translation_axis is None else self.translation_axis

        self.tab2.page5.arm_position_input_target['position'][axis] = self.trajectory_position[axis]
        self.tab2.page5.set_target_data()
        self.tab2.page5.save_target_data()

        # Target 데이터 버퍼에 저장
        self.tab2.page5.target_data_buffer.append(copy.deepcopy(self.tab2.page5.target_data))

        # 이미지 설정
        color_qimage = get_qimage(self.tab2.page5.target_data['hand_color_in_depth_frame'])
        depth_color_qimage = get_qimage(self.tab2.page5.target_data['hand_depth_image'])

        self.main_widget.lbl_hand_color_tgt.setPixmap(QPixmap.fromImage(color_qimage))
        self.main_widget.lbl_hand_depth_tgt.setPixmap(QPixmap.fromImage(depth_color_qimage))

        # Overlap 이미지 설정
        source_image = copy.deepcopy(self.tab2.page5.source_data['hand_color_image'])
        target_image = copy.deepcopy(self.tab2.page5.target_data['hand_color_image'])

        alpha = self.main_widget.sbx_alpha.value()
        beta = self.main_widget.sbx_beta.value()
        gamma = self.main_widget.sbx_gamma.value()
        overlapped = cv2.addWeighted(source_image, alpha, target_image, beta, gamma)
        # overlapped = cv2.addWeighted(source_image, 0.5, target_image, 0.5, 0)
        overlapped_qimage = get_qimage(overlapped)

        self.main_widget.lbl_src_tgt_bitwise.setPixmap(QPixmap.fromImage(overlapped_qimage))

        # Overlap 이미지 저장
        save_file_path = self.main_widget.lblSavePath.text()
        save_file_name = f"overlapped_{idx + 1}.jpg"
        cv2.imwrite(os.path.join(save_file_path, save_file_name), overlapped)

    def run_icp(self, idx):
        # ICP 변수 설정
        self.tab2.page5.icp.set_target(self.tab2.page5.target_pcd)
        if self.is_translation():
            M, surf_image = execute_surf(self.tab2.page5.hand_color_image_source, self.tab2.page5.hand_color_image_target)
            trans_init = get_trans_init(M, self.tab2.page5.source_data['depth_median'], self.tab2.page5.target_data['depth_median'])
            cv2.imwrite(f"{self.main_widget.lblSavePath.text()}/surf_{idx + 1}.jpg", surf_image)
        else:
            trans_init = np.eye(4)

            # rotation = np.eye(3)
            # rotation[0, 1] = self.new_rotation.x
            # rotation[1, 0] = -self.new_rotation.x
            #
            # rotation[1, 2] = self.new_rotation.y
            # rotation[2, 1] = -self.new_rotation.y
            #
            # rotation[0, 2] = self.new_rotation.z
            # rotation[2, 0] = self.new_rotation.z
            #
            # trans_init[:3, :3] = rotation
            # trans_init[:3, 3] = [0, 0, 0]

        self.tab2.page5.icp.set_init_transformation(trans_init)

        # ICP 실행
        icp_st_time = datetime.now()
        print("ICP 시작: ", icp_st_time)

        self.tab2.page5.icp.robust_icp()

        icp_end_time = datetime.now()
        print("ICP 경과시간: ", icp_end_time - icp_st_time)
        np.savetxt(f"{self.main_widget.lblSavePath.text()}/transformation_{idx + 1}.txt",
                   self.tab2.page5.icp.transformation_buffer[idx], delimiter=",")
        print(f"{idx + 1}번쨰 trajectory 완료.")
        print("완료: ", datetime.now())


class GenerateButtonsEvent(QEvent):
    def __init__(self, iteration):
        super().__init__(QEvent.Type(QEvent.registerEventType()))
        self.iteration = iteration


class CustomInputDialog2(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('사용자 정의 입력 대화상자')

        # 데이터 표시
        data_display = QTextEdit(self)
        data_display.setHtml("""
        <h3>저장될 데이터 목록</h3>
        <ol>
            <li>color_image : 저장할 파일명_color_image.jpg</li>
            <li>depth_image : 저장할 파일명_depth_image.png</li>
            <li>Accumulate_count : 누적 횟수 (int)</li>
            <li>Depth_median : depth 데이터의 중앙값 (float, 소수점 첫째자리)</li>
            <li>arm_position_input : position: {x:0.8, y:0, z:0.2}, rotation: {w:1, x:0, y:0, z:0} 형태의 데이터</li>
            <li>arm_position_real : 5번과 같은 형태의 데이터</li>
            <li>odom_tform_hand : 5번과 같은 형태의 데이터</li>
            <li>outlier_remove : iqr1(int), iqr3(int), threshold(float), nb_neighbors(int), std_ratio(float)</li>
        </ol>
        """)
        data_display.setReadOnly(True)

        # 입력 필드 생성
        self.file_name_input = QLineEdit(self)
        self.save_location_input = QLineEdit(self)
        self.save_location_button = QPushButton("찾아보기", self)
        self.save_location_button.clicked.connect(self.show_directory_dialog)

        # 입력 필드 레이아웃 설정
        form_layout = QFormLayout()
        form_layout.addRow("저장할 파일명", self.file_name_input)
        form_layout.addRow("저장 위치", self.save_location_input)
        form_layout.addWidget(self.save_location_button)

        # 확인 및 취소 버튼 생성
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 전체 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(data_display)
        layout.addLayout(form_layout)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def show_directory_dialog(self):
        # 파일 저장 대화상자 생성
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        save_directory = QFileDialog.getExistingDirectory(self, "저장 위치 선택", "", options=options)

        if save_directory:
            self.save_location_input.setText(save_directory)


class CustomInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.hand_color_image = None
        self.hand_depth_image = None
        self.depth_image = None
        self.hand_color_in_depth_frame = None

        self.accumulate_count = None
        self.depth_median = None
        self.arm_position_input = None
        self.arm_position_real  = None
        self.odom_tform_hand    = None

    def initUI(self):
        self.setWindowTitle('데이터 저장')

        # 데이터 설정
        data = {
            "arm_position_input": self.arm_position_input,
            "arm_position_real": self.arm_position_real,
            "odom_tform_hand": self.odom_tform_hand
        }

        # 데이터 레이블 생성
        accumulate_count = 20
        depth_median = 768.0

        # QTreeWidget 생성 및 설정
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabels(["Key", "Value"])
        tree_widget.setColumnWidth(0, 150)
        tree_widget.setMinimumHeight(200)

        # 누적 횟수 항목 생성 및 값 설정
        accumulate_count_item = QTreeWidgetItem(tree_widget)
        accumulate_count_item.setText(0, "누적횟수")
        accumulate_count_item.setText(1, str(self.accumulate_count))

        # depth_median 항목 생성 및 값 설정
        depth_median_item = QTreeWidgetItem(tree_widget)
        depth_median_item.setText(0, "depth_median")
        depth_median_item.setText(1, str(self.depth_median))

        # 데이터를 QTreeWidget에 추가
        for key, value in data.items():
            parent_item = QTreeWidgetItem(tree_widget)
            parent_item.setText(0, key)
            for sub_key, sub_value in value.items():
                sub_item = QTreeWidgetItem(parent_item)
                sub_item.setText(0, sub_key)
                for sub_sub_key, sub_sub_value in sub_value.items():
                    sub_sub_item = QTreeWidgetItem(sub_item)
                    sub_sub_item.setText(0, sub_sub_key)
                    sub_sub_item.setText(1, str(sub_sub_value))

        # 레이아웃 설정
        grid_layout = QGridLayout()

        # 이미지 레이블 생성 및 이미지 설정
        color_image_label = QLabel(self)
        depth_image_label = QLabel(self)
        depth_color_image_label = QLabel(self)
        hand_depth_in_depth_frame_label = QLabel(self)
        path = "D:/TestSW/Source/data/ICP/20230426/Reference"
        q_hand_color_image = get_qimage(self.hand_color_image)
        q_hand_depth_image = get_qimage(self.hand_depth_image)
        q_depth_image = get_qimage(self.depth_image)
        q_hand_color_in_depth_frame = get_qimage(self.hand_color_in_depth_frame)

        color_image = QPixmap(q_hand_color_image).scaled(640, 480)
        depth_image = QPixmap(q_hand_depth_image).scaled(171, 224)
        depth_color_image = QPixmap(q_depth_image).scaled(171, 224)
        hand_depth_in_depth_frame = QPixmap(q_hand_color_in_depth_frame).scaled(171, 224)

        color_image_label.setPixmap(color_image)
        depth_image_label.setPixmap(depth_image)
        depth_color_image_label.setPixmap(depth_color_image)
        hand_depth_in_depth_frame_label.setPixmap(hand_depth_in_depth_frame)

        # 입력 필드 생성
        # 현재 시간 정보 가져오기
        now = datetime.now()

        # 파일명 만들기
        filename = '{}'.format(now.strftime('%Y%m%d_%H%M%S'))

        self.filename_input = QLineEdit(self)
        self.filename_input.setText(filename)
        self.save_path_input = QLineEdit(self)

        self.save_path_button = QPushButton("저장 위치 선택", self)
        self.save_path_button.clicked.connect(self.show_save_path_dialog)

        # 입력 필드 레이아웃 설정
        form_layout = QFormLayout()
        form_layout.addRow("저장할 파일명", self.filename_input)
        form_layout.addRow("저장 위치", self.save_path_input)
        form_layout.addWidget(self.save_path_button)

        # 확인 및 취소 버튼 생성
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 그리드 레이아웃에 위젯 추가
        grid_layout.addWidget(color_image_label, 0, 0)
        grid_layout.addWidget(depth_image_label, 0, 1)
        grid_layout.addWidget(depth_color_image_label, 0, 2)
        grid_layout.addWidget(hand_depth_in_depth_frame_label, 0, 3)

        # 전체 레이아웃 설정
        main_layout = QVBoxLayout()

        # 간격 설정
        main_layout.setSpacing(10)

        main_layout.addLayout(grid_layout)
        main_layout.addWidget(tree_widget)
        main_layout.addLayout(form_layout)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def show_save_path_dialog(self):
        # 파일 저장 위치 대화상자 생성
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        options |= QFileDialog.DontUseNativeDialog
        save_path = QFileDialog.getExistingDirectory(self, "저장 위치 선택", "", options=options)

        if save_path:
            self.save_path_input.setText(save_path)

    # 데이터 설정 메서드
    def set_data(self, data):
        self.hand_color_image = data.get('hand_color_image')
        self.hand_depth_image = data.get('hand_depth_image')
        self.depth_image = data.get('depth_image')
        self.hand_color_in_depth_frame = data.get('hand_color_in_depth_frame')

        self.accumulate_count = data.get('accumulate_count')
        self.depth_median = data.get('depth_median')
        self.arm_position_input = data.get('arm_position_input')
        self.arm_position_real = data.get('arm_position_real')
        self.odom_tform_hand = data.get('odom_tform_hand')

    @property
    def data(self):
        return {
            'hand_color_image': self.hand_color_image,
            'hand_depth_image': self.hand_depth_image,
            'depth_image': self.depth_image,
            'hand_color_in_depth_frame': self.hand_color_in_depth_frame,
            'accumulate_count': self.accumulate_count,
            'depth_median': self.depth_median,
            'arm_position_input': self.arm_position_input,
            'arm_position_real': self.arm_position_real,
            'odom_tform_hand': self.odom_tform_hand
        }

    @data.setter
    def data(self, new_data):
        self.hand_color_image = new_data['hand_color_image']
        self.hand_depth_image = new_data['hand_depth_image']
        self.depth_image = new_data['depth_image']
        self.hand_color_in_depth_frame = new_data['hand_color_in_depth_frame']
        self.accumulate_count = new_data['accumulate_count']
        self.depth_median = new_data['depth_median']
        self.arm_position_input = new_data['arm_position_input']
        self.arm_position_real = new_data['arm_position_real']
        self.odom_tform_hand = new_data['odom_tform_hand']

    # hand_color_image getter 및 setter
    def get_hand_color_image(self):
        return self.hand_color_image

    def set_hand_color_image(self, hand_color_image):
        self.hand_color_image = hand_color_image

    # hand_depth_image getter 및 setter
    def get_hand_depth_image(self):
        return self.hand_depth_image

    def set_hand_depth_image(self, hand_depth_image):
        self.hand_depth_image = hand_depth_image

    # depth_image getter 및 setter
    def get_depth_image(self):
        return self.depth_image

    def set_depth_image(self, depth_image):
        self.depth_image = depth_image

    # hand_color_in_depth_frame getter 및 setter
    def get_hand_color_in_depth_frame(self):
        return self.hand_color_in_depth_frame

    def set_hand_color_in_depth_frame(self, hand_color_in_depth_frame):
        self.hand_color_in_depth_frame = hand_color_in_depth_frame

    # accumulate_count getter 및 setter
    def get_accumulate_count(self):
        return self.accumulate_count

    def set_accumulate_count(self, accumulate_count):
        self.accumulate_count = accumulate_count

    # depth_median getter 및 setter
    def get_depth_median(self):
        return self.depth_median

    def set_depth_median(self, depth_median):
        self.depth_median = depth_median

    # arm position getter 및 setter
    def arm_position_input(self):
        return self.arm_position_input

    def set_arm_position_input(self, arm_position_input):
        self.arm_position_input = arm_position_input

    def arm_position_real(self):
        return self.arm_position_ireal

    def set_arm_position_real(self, arm_position_real):
        self.arm_position_real = arm_position_real

    def odom_tform_hand(self):
        return self.odom_tform_hand

    def set_odom_tform_hand(self, odom_tform_hand):
        self.odom_tform_hand = odom_tform_hand


class SaveImageDialog(QDialog):
    def __init__(self, parent=None):
        super(SaveImageDialog, self).__init__(parent)

        self.setWindowTitle("이미지 저장하기")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.layout = QVBoxLayout()

        self.file_name_label = QLabel("파일명 (.jpg 확장자 생략):")
        self.file_name_input = QLineEdit()

        self.folder_button = QPushButton("저장 폴더 선택")
        self.folder_label = QLabel("선택된 폴더 없음")

        self.save_button = QPushButton("저장하기")
        self.cancel_button = QPushButton("취소하기")

        self.layout.addWidget(self.file_name_label)
        self.layout.addWidget(self.file_name_input)
        self.layout.addWidget(self.folder_button)
        self.layout.addWidget(self.folder_label)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

        self.folder_button.clicked.connect(self.select_folder)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def select_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "저장할 폴더 선택", "", options=options)
        if folder_path:
            self.folder_label.setText(folder_path)


class MatrixDialog(QDialog):
    def __init__(self, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matrix Dialog")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # 윈도우 사이즈 지정
        self.resize(650, 220)

        layout = QVBoxLayout(self)
        self.table_widget = QTableWidget(4, 4)
        horizontal_header = self.table_widget.horizontalHeader()
        horizontal_header.setDefaultSectionSize(150)

        layout.addWidget(self.table_widget)
        self.set_matrix(matrix)

        save_button = QPushButton("Save")
        layout.addWidget(save_button)
        save_button.clicked.connect(self.save_matrix)

    def set_matrix(self, matrix):
        for row in range(4):
            for col in range(4):
                item = QTableWidgetItem(str(matrix[row][col]))
                self.table_widget.setItem(row, col, item)

    def save_matrix(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Matrix", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as file:
                for row in range(4):
                    line = ', '.join(str(self.table_widget.item(row, col).text()) for col in range(4))
                    file.write(line + '\n')

