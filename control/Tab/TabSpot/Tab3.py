import re
import time
from datetime import datetime
from functools import partial

from PyQt5.QtCore import QThread
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QFileDialog, QApplication
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME
from bosdyn.client.lease import NoSuchLease
from bosdyn.client.power import EstoppedError

from control.Control import MainFunctions
from control.utils.utils import *
from spot.CarInspection.CarInspection import CarInspection


class ThreadWorker(QThread):
    # Signal for progress
    progress = QtCore.pyqtSignal()

    # Signal for completion
    completed = QtCore.pyqtSignal()

    def run(self):
        while True:
            self.progress.emit()
            time.sleep(0.2)


def method_decorator(method):
    def wrapper(self, *args):
        try:
            if len(args) <= 1:
                result = method(self)
            else:
                result = method(self, args[0])

            result = str(result)
            succeed_msg = ""

        except AttributeError:
            if self.main_window.robot.robot_lease_keepalive is None:
                result = "[]No lease. Have to get a lease."
            else:
                result = "Must be connect the robot."
            succeed_msg = "Failed"

        except EstoppedError as e:
            # Estop 켜져 있지 않은 경우.
            result = e.error_message
            succeed_msg = "Failed"
            self.main_func.show_message_box(result)

        except NoSuchLease:
            result = "No lease. Have to get a lease."
            succeed_msg = "Failed"

        except Exception as e:
            result = str(e)
            succeed_msg = "Failed"
            self.main_func.show_message_box(result)

        method_name = method.__name__.lstrip("_")
        result = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {succeed_msg} {method_name} : {result}"

        self.main_widget.listWidget.addItem(result)
        self.main_widget.listWidget.scrollToBottom()

    return wrapper


def grpc_decorator(command):
    def wrapper(self, *args):
        try:
            if len(args) <= 1:
                result = command(self)
            else:
                result = command(self, args[0])

        except AttributeError:
            result = "Must be connect the robot."
        except Exception as e:
            result = e

        if type(result) == int:
            succeed_msg = "Success"
        else:
            succeed_msg = "Failed"

            # TypeError("'NoneType' object is not subscriptable",)
            if type(result) == TypeError:
                result = "lease is being initialized.."
            # Message: No lease for resource "body"
            elif type(result) == NoSuchLease:
                result = "No lease. Have to get a lease."

        result = str(result)
        method_name = command.__name__.lstrip("_")
        result = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {succeed_msg} {method_name} : {result}"

        self.main_widget.listWidget.addItem(result)
        self.main_widget.listWidget.scrollToBottom()

    return wrapper


class Tab3:
    saved_joint_param = {}

    def __init__(self, main_window):
        self.main_window = main_window
        self.main_widget = self.main_window.main_window
        self.main_func = MainFunctions(self.main_window)

        self.status_thread = None
        self.joint_thread  = None

        self.init_signals()

    def init_ui(self):
        self.main_widget.spbRobotSpeed.setValue(self.main_window.robot.robot_move_manager.VELOCITY_BASE_SPEED)
        self.main_widget.spbRobotAngular.setValue(self.main_window.robot.robot_move_manager.VELOCITY_BASE_ANGULAR)

        self.main_widget.spbJointSpeed.setValue(self.main_window.robot.robot_arm_manager.JOINT_MOVE_RATE)
        self.main_widget.spbJointTime.setValue(self.main_window.robot.robot_arm_manager.JOINT_TIME_SEC)

    def init_signals(self):
        self.main_widget.btnConnect.clicked.connect(self._connect)
        self.main_widget.btnLease.clicked.connect(self._lease)
        self.main_widget.btnPower.clicked.connect(self._power)
        # self.main_widget.btnEstop.clicked.connect(self._estop)
        # self.main_widget.btnStatus.clicked.connect(self._start)

        self.main_widget.btnCaptureColor.clicked.connect(self._capture_color)
        self.main_widget.btnCaptureSource.clicked.connect(self._capture_source)

        self.main_widget.btnMoveUp.clicked.connect(self._move_up)
        self.main_widget.btnMoveDown.clicked.connect(self._move_down)
        self.main_widget.btnMoveLeft.clicked.connect(self._strafe_left)
        self.main_widget.btnMoveRight.clicked.connect(self._strafe_right)
        self.main_widget.btnStrafeLeft.clicked.connect(self._turn_left)
        self.main_widget.btnStrafeRight.clicked.connect(self._turn_right)

        self.main_widget.btnDock.clicked.connect(self._dock)
        self.main_widget.btnUndock.clicked.connect(self._undock)

        self.main_widget.btnStow.clicked.connect(self._stow)
        self.main_widget.btnUnstow.clicked.connect(self._unstow)
        self.main_widget.btnGripperOpen.clicked.connect(self._gripper_open)
        self.main_widget.btnGripperClose.clicked.connect(self._gripper_close)
        self.main_widget.btnStandUp.clicked.connect(self._stand)
        self.main_widget.btnSitDown.clicked.connect(self._sit)

        self.main_widget.cmbChangeBodyPage.currentIndexChanged.connect(self.onChangeIndex)

        self.main_widget.btn_sh0_left .clicked.connect(partial(self._joint_move, "sh0_left"))
        self.main_widget.btn_sh0_right.clicked.connect(partial(self._joint_move, "sh0_right"))
        self.main_widget.btn_sh1_up   .clicked.connect(partial(self._joint_move, "sh1_up"))
        self.main_widget.btn_sh1_down .clicked.connect(partial(self._joint_move, "sh1_down"))
        self.main_widget.btn_el0_up   .clicked.connect(partial(self._joint_move, "el0_up"))
        self.main_widget.btn_el0_down .clicked.connect(partial(self._joint_move, "el0_down"))
        self.main_widget.btn_el1_left .clicked.connect(partial(self._joint_move, "el1_left"))
        self.main_widget.btn_el1_right.clicked.connect(partial(self._joint_move, "el1_right"))
        self.main_widget.btn_wr0_up   .clicked.connect(partial(self._joint_move, "wr0_up"))
        self.main_widget.btn_wr0_down .clicked.connect(partial(self._joint_move, "wr0_down"))
        self.main_widget.btn_wr1_left .clicked.connect(partial(self._joint_move, "wr1_left"))
        self.main_widget.btn_wr1_right.clicked.connect(partial(self._joint_move, "wr1_right"))

        # self.main_widget.btnJointMoveManual.clicked.connect(self._joint_move_manual)
        # self.main_widget.btnSetJointParam.clicked.connect(self._set_joint_param)
        # self.main_widget.btnSaveJointParam_1.clicked.connect(partial(self._save_joint_param, 1))
        # self.main_widget.btnSaveJointParam_2.clicked.connect(partial(self._save_joint_param, 2))
        # self.main_widget.btnSaveJointParam_3.clicked.connect(partial(self._save_joint_param, 3))
        # self.main_widget.btnLoadJointParam_1.clicked.connect(partial(self._load_joint_param, 1))
        # self.main_widget.btnLoadJointParam_2.clicked.connect(partial(self._load_joint_param, 2))
        # self.main_widget.btnLoadJointParam_3.clicked.connect(partial(self._load_joint_param, 3))

        self.main_widget.spbRobotSpeed.valueChanged.connect(self.robot_body_speed_change)
        self.main_widget.spbRobotAngular.valueChanged.connect(self.robot_angular_change)
        self.main_widget.spbJointSpeed.valueChanged.connect(self.arm_joint_speed_change)
        self.main_widget.spbJointTime.valueChanged.connect(self.arm_joint_time_change)

        self.status_thread = ThreadWorker()
        self.status_thread.progress.connect(self._status)

        self.joint_thread = ThreadWorker()
        self.joint_thread.progress.connect(self._joint)

        # GraphNav Page
        self.main_widget.btn_get_localization_state.clicked.connect(self.get_localization)
        self.main_widget.btn_get_list_graph.clicked.connect(lambda: self.get_list_graph(True))
        self.main_widget.btn_upload_graph.clicked.connect(self.upload_graph)
        self.main_widget.btn_navigate_to.clicked.connect(self.navigate_to)
        self.main_widget.btn_navigate_route.clicked.connect(self.navigate_route)

        self.main_widget.btn_clear_map.clicked.connect(self.clear_map)
        self.main_widget.btn_start_recording.clicked.connect(self.start_recording)
        self.main_widget.btn_stop_recording.clicked.connect(self.stop_recording)
        self.main_widget.btn_get_recording_status.clicked.connect(self.get_recording_status)
        self.main_widget.btn_create_waypoint.clicked.connect(self.create_waypoint)
        self.main_widget.btn_download_full_graph.clicked.connect(self.download_full_graph)
        self.main_widget.btn_create_new_edge.clicked.connect(self.create_new_edge)

        # Tab2 - stow/unstow button
        self.main_widget.btn_stow_manual.clicked.connect(self._stow)
        self.main_widget.btn_unstow_manual.clicked.connect(self._unstow)

    def onChangeIndex(self, index):
        self.main_widget.bodyWidgetSpot.setCurrentIndex(index)

    def _status_thread_start(self):
        self.status_thread.start()

    def _joint_thread_start(self):
        self.joint_thread.start()

    def _connect(self):
        self.main_widget.listWidget.addItem("Connecting...")
        robot_ip = self.main_widget.iPLineEdit.text()
        username = self.main_widget.usernameLineEdit.text()
        password = self.main_widget.passwordLineEdit.text()
        connect, content = self.main_window.robot.connect(robot_ip, username, password)
        if connect:
            self._status_thread_start()
            self._joint_thread_start()
            self.init_ui()

            # CarInspection 초기화
            self.main_window.tab_car_inspection.car_inspection = CarInspection(self.main_window.robot)

        else:
            # self.main_func.show_message_box("The robot may be offline or otherwise unreachable.")
            self.main_func.show_message_box(content)

        # self.main_func.show_message_box("연결 \n" + str(self.main_window.robot.robot_id))
        # self.main_widget.listWidget.addItem("Connect.")

    def _test_stereo_capture(self):
        self.main_func.load_image(self.main_widget.lblResultImage, False)

    @method_decorator
    def _lease(self):
        return self.main_window.robot.command_dictionary["lease"]()

    @method_decorator
    def _power(self):
        return self.main_window.robot.command_dictionary["power"]()

    # def _estop(self):
    #     func = self.main_window.robot.command_dictionary["estop"]
    #     func()

    def _status(self):
        lease   = self.main_window.robot.command_dictionary["get_lease"]()
        power   = self.main_window.robot.command_dictionary["get_power"]()
        status, bar_val, time_left = self.main_window.robot.command_dictionary["get_battery"]()

        style = get_power_style(power)

        self.main_widget.lblLeaseValue.setText(lease)
        self.main_widget.lblPowerValue.setText(power)
        set_status_time_value(self.main_widget.progressBar_battery, status, bar_val, time_left)
        set_label_color(self.main_widget.lblPowerValue, style["color"], style["background-color"])

        self.main_widget.lblLeaseValue_title.setText(lease)
        self.main_widget.lblPowerValue_title.setText(power)
        set_status_time_value(self.main_widget.progressBar_battery_title, status, bar_val, time_left)
        set_label_color(self.main_widget.lblPowerValue_title, style["color"], style["background-color"])

    def _joint(self):
        joint_params = self.main_window.robot.get_current_joint_state()

        if joint_params is None:
            return

        self.main_widget.lbl_sh0_CurrentValue.setText(str(joint_params['sh0']))
        self.main_widget.lbl_sh1_CurrentValue.setText(str(joint_params['sh1']))
        self.main_widget.lbl_el0_CurrentValue.setText(str(joint_params['el0']))
        self.main_widget.lbl_el1_CurrentValue.setText(str(joint_params['el1']))
        self.main_widget.lbl_wr0_CurrentValue.setText(str(joint_params['wr0']))
        self.main_widget.lbl_wr1_CurrentValue.setText(str(joint_params['wr1']))

    def _set_joint_param(self):
        self.main_widget.sh0LineEdit.setText(self.main_widget.lbl_sh0_CurrentValue.text())
        self.main_widget.sh1LineEdit.setText(self.main_widget.lbl_sh1_CurrentValue.text())
        self.main_widget.el0LineEdit.setText(self.main_widget.lbl_el0_CurrentValue.text())
        self.main_widget.el1LineEdit.setText(self.main_widget.lbl_el1_CurrentValue.text())
        self.main_widget.wr0LineEdit.setText(self.main_widget.lbl_wr0_CurrentValue.text())
        self.main_widget.wr1LineEdit.setText(self.main_widget.lbl_wr1_CurrentValue.text())

    def _save_joint_param(self, key):
        try:
            sh0 = float(self.main_widget.sh0LineEdit.text())
            sh1 = float(self.main_widget.sh1LineEdit.text())
            el0 = float(self.main_widget.el0LineEdit.text())
            el1 = float(self.main_widget.el1LineEdit.text())
            wr0 = float(self.main_widget.wr0LineEdit.text())
            wr1 = float(self.main_widget.wr1LineEdit.text())

            self.saved_joint_param[key] = [sh0, sh1, el0, el1, wr0, wr1]
        except ValueError as e:
            print("ValueError: ", e)

    def _load_joint_param(self, key):
        try:
            sh0, sh1, el0, el1, wr0, wr1 = self.saved_joint_param[key]
            self.main_widget.sh0LineEdit.setText(str(sh0))
            self.main_widget.sh1LineEdit.setText(str(sh1))
            self.main_widget.el0LineEdit.setText(str(el0))
            self.main_widget.el1LineEdit.setText(str(el1))
            self.main_widget.wr0LineEdit.setText(str(wr0))
            self.main_widget.wr1LineEdit.setText(str(wr1))
        except KeyError as e:
            print("KeyError: ", e)

    def _capture_color(self):
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return
        image, image_data = self.main_window.robot.robot_camera_manager.take_image()
        qimage = get_qimage(image)
        self.main_widget.lblImage.setPixmap(QPixmap.fromImage(qimage))

        self.main_widget.lblImageSize.setText(str(image_data.image_size))
        self.main_widget.lblArmPosition.setText("Position \n" + str(image_data.image_arm_wr1.position))
        self.main_widget.lblArmRotation.setText("Rotation \n" + str(image_data.image_arm_wr1.rotation))
        self.main_widget.lblSensorPosition.setText("Position \n" + str(image_data.image_hand_sensor.position))
        self.main_widget.lblSensorRotation.setText("Rotation \n" + str(image_data.image_hand_sensor.rotation))
        fl_x = image_data.image_pinhole_intrinsics.focal_length.x
        fl_y = image_data.image_pinhole_intrinsics.focal_length.y
        pp_x = image_data.image_pinhole_intrinsics.principal_point.x
        pp_y = image_data.image_pinhole_intrinsics.principal_point.y
        self.main_widget.lblFocalLength.setText("Focal Length \n" + str(image_data.image_pinhole_intrinsics.focal_length))
        self.main_widget.lblPrincipalPoint.setText("Principal Point \n" + str(image_data.image_pinhole_intrinsics.principal_point))

    def _capture_source(self):
        """
            frontright_depth
            frontleft_depth
            left_depth
            right_depth
            back_depth
            hand_depth
            hand_color_in_hand_depth_frame
            hand_depth_in_hand_color_frame
        """
        source = self.main_widget.cmbDepth.currentText()
        if self.main_window.robot.robot_camera_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        try:
            image, _, image_data = self.main_window.robot.robot_camera_manager.take_image_from_source(source)
        except ValueError:
            image = self.main_window.robot.robot_camera_manager.take_image_from_source(source)
            qimage = get_qimage(image)
            self.main_widget.lblImage.setPixmap(QPixmap.fromImage(qimage))
            return
        # image, _ = self.main_window.robot.robot_camera_manager.get_depth_image_from_source(source)
        qimage = get_qimage(image)
        self.main_widget.lblImage.setPixmap(QPixmap.fromImage(qimage))
        # return self.main_window.robot.robot_move_manager.selfright()

        self.main_widget.lblImageSize.setText(str(image_data.image_size))
        self.main_widget.lblArmPosition.setText("Position \n" + str(image_data.image_arm_wr1.position))
        self.main_widget.lblArmRotation.setText("Rotation \n" + str(image_data.image_arm_wr1.rotation))
        self.main_widget.lblSensorPosition.setText("Position \n" + str(image_data.image_hand_sensor.position))
        self.main_widget.lblSensorRotation.setText("Rotation \n" + str(image_data.image_hand_sensor.rotation))
        fl_x = image_data.image_pinhole_intrinsics.focal_length.x
        fl_y = image_data.image_pinhole_intrinsics.focal_length.y
        pp_x = image_data.image_pinhole_intrinsics.principal_point.x
        pp_y = image_data.image_pinhole_intrinsics.principal_point.y
        self.main_widget.lblFocalLength.setText("Focal Length \n" + str(image_data.image_pinhole_intrinsics.focal_length))
        self.main_widget.lblPrincipalPoint.setText("Principal Point \n" + str(image_data.image_pinhole_intrinsics.principal_point))

    @method_decorator
    def _move_up(self):
        return self.main_window.robot.robot_move_manager.move_forward()

    @method_decorator
    def _move_down(self):
        return self.main_window.robot.robot_move_manager.move_backward()

    @method_decorator
    def _strafe_left(self):
        return self.main_window.robot.robot_move_manager.strafe_left()

    @method_decorator
    def _strafe_right(self):
        return self.main_window.robot.robot_move_manager.strafe_right()

    @method_decorator
    def _turn_left(self):
        return self.main_window.robot.robot_move_manager.turn_left()

    @method_decorator
    def _turn_right(self):
        return self.main_window.robot.robot_move_manager.turn_right()

    @method_decorator
    def _sit(self):
        return self.main_window.robot.robot_move_manager.sit()

    @method_decorator
    def _stand(self):
        return self.main_window.robot.robot_move_manager.stand()

    # def _dock(self):
    #     try:
    #         self.main_window.robot.dock()
    #     except method_decorator as e:
    #         self.main_func.show_message_box(str(e))

    @method_decorator
    def _dock(self):
        return self.main_window.robot.dock()

    @method_decorator
    def _undock(self):
        return self.main_window.robot.undock()

    @grpc_decorator
    def _stow(self):
        return self.main_window.robot.robot_arm_manager.stow()

    @grpc_decorator
    def _unstow(self):
        return self.main_window.robot.robot_arm_manager.unstow()

    @grpc_decorator
    def _gripper_open(self):
        return self.main_window.robot.robot_arm_manager.gripper_open()

    @grpc_decorator
    def _gripper_close(self):
        return self.main_window.robot.robot_arm_manager.gripper_close()

    @grpc_decorator
    def _joint_move(self, target):
        return self.main_window.robot.robot_arm_manager.joint_move(target)

    @grpc_decorator
    def _joint_move_manual(self):
        sh0 = float(self.main_widget.sh0LineEdit.text())
        sh1 = float(self.main_widget.sh1LineEdit.text())
        el0 = float(self.main_widget.el0LineEdit.text())
        el1 = float(self.main_widget.el1LineEdit.text())
        wr0 = float(self.main_widget.wr0LineEdit.text())
        wr1 = float(self.main_widget.wr1LineEdit.text())

        params = [sh0, sh1, el0, el1, wr0, wr1]
        return self.main_window.robot.robot_arm_manager.joint_move_manual(params)

    def robot_body_speed_change(self, value):
        self.main_window.robot.robot_move_manager.VELOCITY_BASE_SPEED = value

    def robot_angular_change(self, value):
        self.main_window.robot.robot_move_manager.VELOCITY_BASE_ANGULAR = value

    def arm_joint_speed_change(self, value):
        self.main_window.robot.robot_arm_manager.JOINT_MOVE_RATE = value

    def arm_joint_time_change(self, value):
        self.main_window.robot.robot_arm_manager.JOINT_TIME_SEC = value

    # GraphNav Page
    def get_localization(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        state, odom_tform_body = graph_nav_manager.get_localization_state()

        self.main_widget.btn_copy_selected.clicked.connect(self.copyToClipboard)

        # ListView 초기화
        self.main_widget.log_navigation.clear()

        # state 데이터 처리:
        state_str = str(state)  # state를 문자열로 변환합니다.
        state_lines = state_str.split('\n')  # 줄바꿈 문자를 기준으로 문자열을 분리합니다.

        self.main_widget.log_navigation.addItem("State:")  # 분리된 각 줄을 listWidget에 추가합니다.
        for line in state_lines:
            self.main_widget.log_navigation.addItem(line)  # 분리된 각 줄을 listWidget에 추가합니다.

        self.main_widget.log_navigation.addItem("odom_tform_body:")  # 분리된 각 줄을 listWidget에 추가합니다.
        odom_tform_body_str = str(odom_tform_body)
        self.main_widget.log_navigation.addItem(odom_tform_body_str)  # 분리된 각 줄을 listWidget에 추가합니다.

    def copyToClipboard(self):
        selected_items = self.main_widget.log_navigation.selectedItems()
        clipboard = QApplication.clipboard()
        clipboard.setText('\n'.join(item.text() for item in selected_items))

    def get_list_graph(self, is_clear=True):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        waypoints_list, edges_list = graph_nav_manager.list_graph_waypoint_and_edge_ids()

        # ListView 초기화
        if is_clear:
            self.main_widget.log_navigation.clear()

        # waypoints_list와 edges_list를 log_navigation에 추가
        self.main_widget.log_navigation.addItem("Waypoints List:")
        for waypoint in waypoints_list:
            self.main_widget.log_navigation.addItem(waypoint)

        self.main_widget.log_navigation.addItem("Edges List:")
        for edge in edges_list:
            self.main_widget.log_navigation.addItem(edge)

        def copyToClipboard(item):
            clipboard = QApplication.clipboard()
            match = re.search(r'Waypoint name: (.*?) id:', item.text())

            if match:
                waypoint_name = match.group(1)
                print(waypoint_name)
                clipboard.setText(waypoint_name)
                self.main_widget.lineEditWaypoint.setText(waypoint_name)

        self.main_widget.log_navigation.itemClicked.connect(copyToClipboard)

    def localization(self):
        pass

    def upload_graph(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        upload_filepath = QFileDialog.getExistingDirectory(None, 'Select Directory')
        if upload_filepath:
            graph_nav_manager.upload_graph_and_snapshots(upload_filepath)

            self.get_list_graph()

    def navigate_to(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        waypoint = self.main_widget.lineEditWaypoint.text()
        graph_nav_manager.navigate_to(waypoint)

    def navigate_route(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        waypoint1 = self.main_widget.lbl_route_waypoint1.text()
        waypoint2 = self.main_widget.lbl_route_waypoint2.text()
        graph_nav_manager.navigate_route([waypoint1, waypoint2])

    def clear_map(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        graph_nav_manager.clear_map()

    def start_recording(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        start_message = graph_nav_manager.start_recording()

        self.main_widget.log_navigation.clear()
        self.main_widget.log_navigation.addItem(start_message)

    def stop_recording(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        graph_nav_manager.stop_recording()

    def get_recording_status(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        status_message = graph_nav_manager.get_recording_status()

        self.main_widget.log_navigation.clear()
        self.main_widget.log_navigation.addItem(status_message)

    def create_waypoint(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        waypoint_name = self.main_widget.lbl_waypoint_name.text()
        message = graph_nav_manager.create_waypoint(waypoint_name)

        self.main_widget.log_navigation.clear()
        self.main_widget.log_navigation.addItem(message)
        self.get_list_graph(is_clear=False)

    def download_full_graph(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        download_filepath = QFileDialog.getExistingDirectory(None, 'Select Directory')
        if download_filepath:
            graph_nav_manager.set_download_filepath(download_filepath)
            graph_nav_manager.download_full_graph()

    def create_new_edge(self):
        graph_nav_manager = self.main_window.robot.robot_graphnav_manager
        if graph_nav_manager is None:
            self.main_func.show_message_box("Robot 연결이 필요합니다.")
            return

        waypoint1 = self.main_widget.lblWaypoint1.text()
        waypoint2 = self.main_widget.lblWaypoint2.text()
        graph_nav_manager.create_new_edge([waypoint1, waypoint2])

    # Page 2

    # Page: Fiducial
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
        # 다이얼로그를 통해 JSON 파일 선택
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("JSON Files (*.json)")
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                with open(file_path, 'r') as file:
                    self.arm_data = json.load(file)

    def move_arm(self):
        if self.arm_data is None:
            self.main_func.show_message_box('Arm 위치 설정이 되어있지 않습니다.')
            return

        fiducial = self.main_window.robot.robot_fiducial_manager.get_fiducial()
        odom_tform_fiducial_filtered = get_a_tform_b(fiducial.transforms_snapshot,
                                                     ODOM_FRAME_NAME,
                                                     fiducial.apriltag_properties.frame_name_fiducial_filtered)
        fiducial_tform_gripper = self.arm_data['frame_tform_gripper'][1]['transform']
        fiducial_tform_gripper = dict_to_se3pose(fiducial_tform_gripper)
        odom_tform_gripper_goal = odom_tform_fiducial_filtered * fiducial_tform_gripper

        end_seconds = self.main_widget.sbx_joint_move_end_time.value()

        self.main_window.robot.robot_arm_manager.move_to_frame_hand(odom_tform_gripper_goal,
                                                                    ODOM_FRAME_NAME,
                                                                    end_seconds=end_seconds)

    def move_arm_manual(self):
        axis = self.main_widget.cmb_arm_axis.currentText()
        direction = self.main_widget.cmb_arm_direction.currentText()
        joint_move_rate = self.main_widget.sbx_joint_move_rate.value()
        body_tform_hand = self.main_window.robot.get_current_hand_position('hand')
        end_time_sec = self.main_widget.sbx_joint_move_end_time.value()
        self.main_window.robot.robot_arm_manager.trajectory_manual(body_tform_hand, axis, direction,
                                                                   joint_move_rate, end_time_sec)

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