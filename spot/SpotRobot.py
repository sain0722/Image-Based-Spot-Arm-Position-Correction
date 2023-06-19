import logging
import queue
import threading
import time

from bosdyn.api.docking import docking_pb2
from bosdyn.api.docking.docking_pb2 import DockingCommandResponse
from bosdyn.client import UnableToConnectToRobotError, RpcError, InvalidLoginError, ResponseError
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.common import maybe_raise, common_lease_errors
from bosdyn.client.docking import DockingClient, blocking_go_to_prep_pose
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.power import PowerClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, CommandFailedError
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.api.power_pb2 as PowerServiceProto

from bosdyn.util import secs_to_hms, now_sec, seconds_to_timestamp
import bosdyn.client
from bosdyn.mission.client import MissionClient

from control.utils.utils import se3pose_to_dict
from spot.CarInspection.CarInspection import SpotInspection
from spot.SpotGraphNav import SpotGraphNav, SpotGraphNavRecoding
from spot.SpotArm import SpotArm
from spot.SpotAuth import blocking_stand
from spot.SpotCamera import SpotCamera
from spot.SpotCommand import try_grpc, RobotCommandExecutor
from spot.SpotMove import SpotMove


class Robot:
    dock_id = 525

    def __init__(self):
        self.robot = None
        # self.logger = logging.getLogger(self._name or 'bosdyn.Robot')
        self.logger = logging.getLogger('bosdyn.Robot')
        self.robot_id                    = None
        self.power_client                = None
        # self.estop_client                = None
        self.lease_client                = None
        self.robot_state_client          = None
        self.robot_command_client        = None
        self.image_client                = None
        self.gripper_camera_param_client = None
        self.world_object_client         = None
        self.graph_nav_client            = None

        # Setup the recording service client.
        self.recording_client = None
        self.map_processing_client = None

        # Mission Client
        self.mission_client = None

        # lease
        self._lease_keepalive = None

        # estop
        # self._estop_keepalive     = None
        # self._estop_endpoint      = None

        # command
        self.robot_commander = None
        self.robot_move_manager = None
        self.robot_arm_manager = None
        self.robot_camera_manager = None
        self.robot_graphnav_manager = None
        self.robot_recoding_manager = None
        self.robot_fiducial_manager = None
        self.robot_inspection_manager = None

        self._robot_state_task = None
        self.async_tasks       = None

        self.get_state_thread  = None

        self.command_dictionary = {
            # "estop"      : self._toggle_estop,
            "lease"      : self._toggle_lease,
            "power"      : self._toggle_power,
            "get_lease"  : self._lease_str,
            # "get_estop"  : self._estop_str,
            "get_power"  : self._power_state_str,
            "get_battery": self._battery_str
        }

    def connect(self, hostname, username, password):
        # 연결 체크: robot 객체가 생성되었으면 이미 연결되어 있는 것.
        if self.robot:
            return False, 'Already Connected'

        connect, content = self.create_robot(hostname, username, password)
        if connect:
            self.initialize_robot()

        return connect, content

    def create_robot(self, hostname, username, password):
        try:
            sdk = bosdyn.client.create_standard_sdk('TWIM', [MissionClient])
            robot = sdk.create_robot(hostname)
            # bosdyn.client.util.authenticate(robot)
            robot.authenticate(username=username, password=password)
            self.robot = robot
            return True, 'succeed'

        except UnableToConnectToRobotError as exc:
            print(exc)
            return False, exc.error_message
            # quit()

        except RpcError as exc:
            print(exc)
            return False, exc.error_message

        except InvalidLoginError as exc:
            print(exc)
            return False, exc.error_message

        except Exception as exc:
            print(exc)
            return False, "Exception"

    def initialize_robot(self):
        # self.robot = create_robot(hostname)
        if self.robot is None:
            return

        self.robot_id             = self.robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self.power_client         = self.robot.ensure_client(PowerClient.default_service_name)
        # self.estop_client         = self.robot.ensure_client(EstopClient.default_service_name)
        self.lease_client         = self.robot.ensure_client(LeaseClient.default_service_name)
        # try:
        #     self._estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        #     self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        # except:
        #     # Not the estop.
        #     self._estop_client   = None
        #     self._estop_endpoint = None

        self.robot_state_client          = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.robot_command_client        = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client                = self.robot.ensure_client(ImageClient.default_service_name)

        self.gripper_camera_param_client = self.robot.ensure_client(GripperCameraParamClient.default_service_name)
        self.world_object_client         = self.robot.ensure_client(WorldObjectClient.default_service_name)

        self.graph_nav_client            = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.recording_client = self.robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self.map_processing_client = self.robot.ensure_client(MapProcessingServiceClient.default_service_name)

        # Create the client for Mission Service
        self.mission_client = self.robot.ensure_client(MissionClient.default_service_name)

        # client_metadata
        session_name = 'recoding_session_test'
        user_name = self.robot._current_user
        client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
            session_name=session_name, client_username=user_name, client_id='RecordingClient',
            client_type='Python SDK')

        # commander initialize
        self.robot_commander = RobotCommandExecutor(self.robot_command_client)
        self.robot_move_manager     = SpotMove(self.robot_commander)
        self.robot_arm_manager      = SpotArm(self)
        self.robot_camera_manager   = SpotCamera(self.image_client, self.gripper_camera_param_client)
        self.robot_graphnav_manager = SpotGraphNav(self.graph_nav_client)
        self.robot_recoding_manager = SpotGraphNavRecoding([self.recording_client,
                                                            self.graph_nav_client,
                                                            self.map_processing_client], client_metadata)
        self.robot_inspection_manager = SpotInspection(self)
        self.robot_fiducial_manager = self.robot_inspection_manager.move_with_fiducial

        self._robot_state_task = AsyncRobotState(self.robot_state_client)
        self.async_tasks       = AsyncTasks([self._robot_state_task])

        self.start_getting_state()

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    @property
    def robot_lease_keepalive(self):
        return self._lease_keepalive

    def start_getting_state(self):
        # @TODO: robot state async callback_done function
        callback_is_done = False

        self.get_state_thread = threading.Thread(target=self.thread_robot_state)
        self.get_state_thread.daemon = True
        self.get_state_thread.start()

    def thread_robot_state(self):
        while True:
            try:
                self.async_tasks.update()
            except UnableToConnectToRobotError:
                print("The robot may be offline or otherwise unreachable.")
                return

            time.sleep(0.1)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self.power_client.power_command_async(request)

    def _safe_power_off(self):
        # bosdyn.client.power.power_off(self.power_client)
        # self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())
        self.robot_commander.start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def dock(self):
        # make sure standing
        blocking_stand(self.robot_command_client)

        if self.robot_arm_manager.is_arm_unstow():
            self.robot_arm_manager.gripper_close()

        # Create a queue for the result
        q = queue.Queue()

        # Define a new function that calls the original function and puts the result in the queue
        def wrapper_func():
            result = blocking_dock_robot(self.robot, self.dock_id)
            q.put(result)

        # Dock the robot
        dock_thread = threading.Thread(target=wrapper_func, daemon=True)
        dock_thread.start()

        # Wait for the result
        return_value = q.get()
        return return_value

    def undock(self):
        # make sure standing
        # blocking_stand(self.robot_command_client)
        # blocking_go_to_prep_pose(self.robot, self.dock_id)

        # blocking_undock(self.robot)
        if self.robot is None:
            return

        # Create a queue for the result
        q = queue.Queue()

        # Define a new function that calls the original function and puts the result in the queue
        def wrapper_func():
            result = blocking_undock(self.robot)
            q.put(result)

        # Dock the robot
        dock_thread = threading.Thread(target=wrapper_func, daemon=True)
        dock_thread.start()

        # Wait for the result
        return_value = q.get()
        return return_value

    def get_current_joint_state(self):
        state = self.robot_state
        if not state:
            return None
        joint_states = state.kinematic_state.joint_states
        joint_names = ['arm0.sh0', 'arm0.sh1', 'arm0.el0', 'arm0.el1', 'arm0.wr0', 'arm0.wr1']
        joint_pos_list = [
            state.position.value
            for state in joint_states if state.name in joint_names
        ]
        joint_pos_dict = {
            name.split(".")[1]: round(value, 4)
            for name, value in zip(joint_names, joint_pos_list)
        }

        return joint_pos_dict

    def get_current_hand_position(self, key):
        """
        :param key: hand, body, flat_body, gpe, odom, vision, link_wr1
        :return: { position, rotation }
        """
        if not self.robot_state:
            return None

        kinematic_state = self.robot_state.kinematic_state
        return kinematic_state.transforms_snapshot.child_to_parent_edge_map[key].parent_tform_child

    def get_odom_tform_hand(self):
        if not self.robot_state:
            return None

        odom_tform_hand = get_a_tform_b(self.robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, HAND_FRAME_NAME)

        return odom_tform_hand

    def get_hand_position_dict(self):
        hand_snapshot = self.get_current_hand_position('hand')
        if hand_snapshot is None:
            return

        position, rotation = se3pose_to_dict(hand_snapshot)
        return position, rotation

    def get_odom_tform_hand_dict(self):
        odom_tform_hand = self.get_odom_tform_hand()
        if odom_tform_hand is None:
            return

        position, rotation = se3pose_to_dict(odom_tform_hand)
        return position, rotation

    # region estop, lease, power, battery
    # def _toggle_estop(self):
    #     """toggle estop on/off. Initial state is ON"""
    #     if self._estop_client is not None and self._estop_endpoint is not None:
    #         if not self._estop_keepalive:
    #             self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
    #         else:
    #             try_grpc("stopping estop", self._estop_keepalive.stop)
    #             self._estop_keepalive.shutdown()
    #             self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self.lease_client is not None:
            if self._lease_keepalive is None:
                try:
                    message = "lease acquire is succeed."
                    self.lease_client.acquire()

                except ResourceAlreadyClaimedError:
                    message = "the robot is already standing via the tablet. Will take over from the tablet."
                    self.lease_client.take()

                self._lease_keepalive = LeaseKeepAlive(self.lease_client,
                                                       must_acquire=True,
                                                       return_at_exit=True,
                                                       on_failure_callback=self.lease_keepalive_failure_callback)
            else:
                message = "return lease is succeed."
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None
        else:
            message = "Must be connect the robot."

        return message

    def lease_keepalive_failure_callback(self, exc):
        print("resuming check-in")
        print(exc)
        try:
            self._lease_keepalive.shutdown()
        except Exception as e:
            print(e)
        self._lease_keepalive = None
        # self.lease_client.return_lease(self.lease_client.lease_wallet.get_lease())

    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            # self.add_message('Could not toggle power because power state is unknown')
            print('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            # try_grpc_async("powering-on", self._request_power_on)
            result = self._request_power_on()
            result = result.result().status
            if result == 1:
                result = 'POWER_STATUS_OK'
        else:
            result = try_grpc("powering-off", self._safe_power_off)

        return result

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self):
        if self._lease_keepalive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = self._lease_keepalive.lease_wallet.get_lease()
                lease = '{}:{}'.format(_lease.lease_proto.resource, _lease.lease_proto.sequence)
            except bosdyn.client.lease.Error:
                lease = '...'
            except bosdyn.client.LeaseUseError as e:
                lease = e

            if self._lease_keepalive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return '{} THREAD:{}'.format(lease, alive)

    def _power_state_str(self):
        if not self._robot_state_task:
            return ''

        power_state = self._power_state()
        if power_state is None:
            state_str = ""
        else:
            state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return '{}'.format(state_str[6:])  # get rid of STATE_ prefix

    # def _estop_str(self):
    #     if not self.estop_client:
    #         thread_status = 'NOT ESTOP'
    #     else:
    #         thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
    #     estop_status = '??'
    #
    #     # if not self.robot_state
    #     if not self._robot_state_task:
    #         return 'Estop {} (thread: {})'.format(estop_status, thread_status)
    #
    #     state = self.robot_state
    #     if state:
    #         for estop_state in state.estop_states:
    #             if estop_state.type == estop_state.TYPE_SOFTWARE:
    #                 estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
    #                 break
    #     return 'Estop {} (thread: {})'.format(estop_status, thread_status)

    def _battery_str(self):
        # if not self.robot_state:
        #     return ''
        if not self._robot_state_task:
            return ''

        if self.robot_state is None:
            status    = ""
            bar_val   = 0
            bat_bar   = ""
            time_left = ""
        else:
            battery_state = self.robot_state.battery_states[0]
            status = battery_state.Status.Name(battery_state.status)
            status = status[7:]  # get rid of STATUS_ prefix
            if battery_state.charge_percentage.value:
                bar_val = battery_state.charge_percentage.value
                bar_len = int(bar_val) // 5
                bat_bar = '|{}{}|'.format('=' * bar_len, ' ' * (20 - bar_len))
            else:
                bar_val = 0
                bat_bar = ""
            time_left = ""
            if battery_state.estimated_runtime:
                # time_left = ' ({})'.format(secs_to_hms(battery_state.estimated_runtime.seconds))
                time_left = secs_to_hms(battery_state.estimated_runtime.seconds)
        # return '{} ({}) \n{} {}'.format(status, bar_val, bat_bar, time_left)
        return status, bar_val, time_left

    # endregion


LOGGER = logging.getLogger()


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__("robot_state", robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()


def blocking_dock_robot(robot, dock_id, num_retries=4, timeout=30):
    """Blocking helper that takes control of the robot and docks it.

    Args:
        robot: The instance of the robot to control.
        dock_id: The ID of the dock to dock at.
        num_retries: Optional, number of attempts.

    Returns:
        The number of retries required

    Raises:
        CommandFailedError: The robot was unable to be docked. See error for details.
    """
    docking_client = robot.ensure_client(DockingClient.default_service_name)

    attempt_number = 0
    docking_success = False

    # Try to dock the robot
    while attempt_number < num_retries and not docking_success:
        attempt_number += 1
        converter = robot.time_sync.get_robot_time_converter()
        start_time = converter.robot_seconds_from_local_seconds(now_sec())
        cmd_end_time = start_time + timeout
        cmd_timeout = cmd_end_time + 10  # client side buffer

        prep_pose = (docking_pb2.PREP_POSE_USE_POSE if
                     (attempt_number % 2) else docking_pb2.PREP_POSE_SKIP_POSE)

        try:
            cmd_id = docking_client.docking_command(dock_id, robot.time_sync.endpoint.clock_identifier,
                                                    seconds_to_timestamp(cmd_end_time), prep_pose)
        except ResponseError as exc:
            return exc.error_message

        while converter.robot_seconds_from_local_seconds(now_sec()) < cmd_timeout:
            feedback = docking_client.docking_command_feedback_full(cmd_id)
            maybe_raise(common_lease_errors(feedback))
            status = feedback.status
            if status == docking_pb2.DockingCommandFeedbackResponse.STATUS_IN_PROGRESS:
                # keep waiting/trying
                time.sleep(1)
            elif status == docking_pb2.DockingCommandFeedbackResponse.STATUS_DOCKED:
                docking_success = True
                break
            elif (status in [
                    docking_pb2.DockingCommandFeedbackResponse.STATUS_MISALIGNED,
                    docking_pb2.DockingCommandFeedbackResponse.STATUS_ERROR_COMMAND_TIMED_OUT,
            ]):
                # Retry
                break
            else:
                return CommandFailedError(
                    "Docking Failed, status: '%s'" %
                    docking_pb2.DockingCommandFeedbackResponse.Status.Name(status))

    if docking_success:
        return attempt_number - 1

    # Try and put the robot in a safe position
    try:
        blocking_go_to_prep_pose(robot, dock_id)
    except CommandFailedError:
        pass

    # Raise error on original failure to dock
    return CommandFailedError("Docking Failed, too many attempts")


def blocking_undock(robot, timeout=20):
    """Blocking helper that undocks the robot from the currently docked dock.

    Args:
        robot: The instance of the robot to control.

    Returns:
        None

    Raises:
        CommandFailedError: The robot was unable to undock. See error for details.
    """
    docking_client = robot.ensure_client(DockingClient.default_service_name)

    converter = robot.time_sync.get_robot_time_converter()
    start_time = converter.robot_seconds_from_local_seconds(now_sec())
    cmd_end_time = start_time + timeout
    cmd_timeout = cmd_end_time + 10  # client side buffer
    try:
        cmd_id = docking_client.docking_command(0, robot.time_sync.endpoint.clock_identifier,
                                                seconds_to_timestamp(cmd_end_time),
                                                docking_pb2.PREP_POSE_UNDOCK)
    except bosdyn.client.exceptions.ResponseError as e:
        return e.error_message
    except bosdyn.client.lease.NoSuchLease as e:
        return e

    while converter.robot_seconds_from_local_seconds(now_sec()) < cmd_timeout:
        feedback = docking_client.docking_command_feedback_full(cmd_id)
        maybe_raise(common_lease_errors(feedback))
        status = feedback.status
        if status == docking_pb2.DockingCommandFeedbackResponse.STATUS_IN_PROGRESS:
            # keep waiting/trying
            time.sleep(1)
        elif status == docking_pb2.DockingCommandFeedbackResponse.STATUS_AT_PREP_POSE:
            return
        else:
            raise CommandFailedError("Failed to undock the robot, status: '%s'" %
                                     docking_pb2.DockingCommandFeedbackResponse.Status.Name(status))

    raise CommandFailedError("Error undocking the robot, timeout exceeded.")
