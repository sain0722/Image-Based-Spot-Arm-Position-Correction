import threading
import time

from bosdyn import geometry
from bosdyn.api import world_object_pb2, geometry_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2, Vec3
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2

from spot.CarInspection.car_inspection_helpers import *
from spot.CarInspection.inspection_helpers import ArmSensorInspector


class SpotInspection:
    def __init__(self, robot):
        self.robot = robot
        self.move_with_fiducial = MoveWithFiducial(robot)
        self.car_inspection = CarInspection(robot)


class MoveWithFiducial:
    def __init__(self, robot):
        self.robot = robot
        self.robot_command_client = robot.robot_command_client
        self.world_object_client = robot.world_object_client

        self._tag_offset = float(0.5) + 1.1 / 2.0  # meters
        self._limit_speed = True  # Limit the robot's walking speed.
        self._avoid_obstacles = True  # Disable obstacle avoidance.

        # Epsilon distance between robot and desired go-to point.
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .075

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # Latest detected fiducial's position in the world.
        self._current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the fiducial.
        self._angle_desired = None

    def position_body_over_fiducial(self, world_object_client, robot_command_client):
        """Center the body over the fiducial in a stance."""
        fiducials = world_object_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]).world_objects

        assert len(fiducials) >= 1
        goal_fiducial = get_nearest_fiducial(fiducials)
        if not goal_fiducial:
            print("없음")
            return None, None, None

        odom_tform_fiducial = frame_helpers.get_se2_a_tform_b(
            goal_fiducial.transforms_snapshot, frame_helpers.ODOM_FRAME_NAME,
            goal_fiducial.apriltag_properties.frame_name_fiducial)

        stow_cmd = RobotCommandBuilder.arm_stow_command()

        # First walk the robot to the fiducial. Center the body on the fiducial.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=odom_tform_fiducial.x, goal_y=odom_tform_fiducial.y,
            goal_heading=odom_tform_fiducial.angle, frame_name=frame_helpers.ODOM_FRAME_NAME,
            build_on_command=stow_cmd)

        end_time = 10.0
        cmd_id = robot_command_client.robot_command(command=robot_cmd,
                                                    end_time_secs=time.time() + end_time)

        should_continue = block_for_trajectory_cmd(robot_command_client, cmd_id, timeout_sec=end_time)
        if not should_continue and not threading.current_thread().is_alive():
            # Only stop if the thread was explicitly cancelled. Otherwise, we probably just
            # timed out on the trajectory command.
            print("Stopping after the blocking trajectory command because it was cancelled.")
            return should_continue, None, None

        # This example ues the position of the fiducial and specifies the stance offsets relative to
        # the center of the fiducial.
        # Stance offsets from fiducial position.
        x_offset = 0.25
        y_offset = 0.25

        pos_fl_rt_odom = odom_tform_fiducial * math_helpers.SE2Pose(x_offset, y_offset, 0)
        pos_fr_rt_odom = odom_tform_fiducial * math_helpers.SE2Pose(x_offset, -y_offset, 0)
        pos_hl_rt_odom = odom_tform_fiducial * math_helpers.SE2Pose(-x_offset, y_offset, 0)
        pos_hr_rt_odom = odom_tform_fiducial * math_helpers.SE2Pose(-x_offset, -y_offset, 0)

        stance_cmd = RobotCommandBuilder.stance_command(
            frame_helpers.ODOM_FRAME_NAME, pos_fl_rt_odom.position, pos_fr_rt_odom.position,
            pos_hl_rt_odom.position, pos_hr_rt_odom.position)
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stance_robot_command = RobotCommandBuilder.build_synchro_command(stow_cmd, stance_cmd)

        end_time = 7.0  # seconds
        print("Issuing the stance command.")
        cmd_id_stance = robot_command_client.robot_command(command=stance_robot_command,
                                                           end_time_secs=time.time() + end_time)

        should_continue = block_for_stance_cmd(robot_command_client, cmd_id_stance, end_time)
        return should_continue, goal_fiducial, stance_cmd

    def set_mobility_params(self, body_control=None):
        """Set robot mobility params to disable obstacle avoidance."""
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                    disable_vision_foot_obstacle_avoidance=True,
                                                    disable_vision_foot_constraint_avoidance=True,
                                                    obstacle_avoidance_padding=.001)
        if not body_control:
            body_control = set_default_body_control()
        if self._limit_speed:
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
            if not self._avoid_obstacles:
                mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            else:
                mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        elif not self._avoid_obstacles:
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        else:
            # When set to none, RobotCommandBuilder populates with good default values
            mobility_params = None
        return mobility_params

    def trajectory_cmd(self, goto, heading, body_height=0):
        rotation = geometry.EulerZXY().to_quaternion()
        NOMINAL_HEIGHT = 0.515
        position = Vec3(x=0.0, y=0.0, z=0.0)
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        heading_yaw = Quat(w=heading.w, x=heading.x, y=heading.y, z=heading.z).to_yaw()
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

        # set mobility params
        mobility_params = self.set_mobility_params(body_control=body_control)

        # command
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=goto.x,
            goal_y=goto.y,
            goal_heading=heading_yaw,
            frame_name=frame_helpers.ODOM_FRAME_NAME,
            params=mobility_params
        )

        # Execute
        end_time = 5
        cmd_id = self.robot_command_client.robot_command(command=robot_cmd,
                                                         end_time_secs=time.time() + end_time)
        return block_for_trajectory_cmd(self.robot_command_client, cmd_id, end_time)

    def walk_to_fiducial(self, fiducial, dist_margin=1.0):
        if not fiducial:
            print('[walk_to_fiducial] No fiducial nearby.')
            return None

        # SE3 tform
        odom_tform_fiducial = get_a_tform_b(
            fiducial.transforms_snapshot, frame_helpers.ODOM_FRAME_NAME,
            fiducial.apriltag_properties.frame_name_fiducial_filtered
        )

        if is_fiducial_horizontal(odom_tform_fiducial):
            print("fiducial is horizontal")
            return False, False
        else:
            goto, heading = offset_tag_pose_with_rotation(odom_tform_fiducial, dist_margin)
            odom_T_body = math_helpers.SE3Pose(x=goto.x, y=goto.y, z=goto.z, rot=heading)
            fiducial_T_body = odom_tform_fiducial.inverse() * odom_T_body

            attmept_number = 0
            num_retries = 1
            while attmept_number < num_retries:
                attmept_number += 1
                self.trajectory_cmd(goto, heading)

    def get_fiducial(self):
        fiducials = self.world_object_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]).world_objects

        assert len(fiducials) >= 1
        goal_fiducial = get_nearest_fiducial(fiducials)

        return goal_fiducial

    def get_target_fiducial(self, fiducial_id):
        fiducials = self.world_object_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]).world_objects

        assert len(fiducials) >= 1
        goal_fiducial = get_target_fiducial(fiducials, fiducial_id)

        if len(goal_fiducial) == 1:
            return goal_fiducial
        else:
            return None

    def centering_on_nearest_fiducial(self, dist_margin):
        """ Run centering on fiducial service:
            Taking the fid_id and dist_margin
            1. Finds for the waypoint associated with te fiducial
            2. Then, moves the robot to the waypoint
            3. Detects and centers the robot on the closet fiducial in a stable stance.
        """
        # Take the first argument as the destination fiducial number.
        # if len(args) < 1:
        #     print("Wrong input. [fid-id] [dist-margin]")
        #     return

        # should_continue, goal_fiducial, _ = self.position_body_over_fiducial(self.Fiducial.world_object_client,
        #                                                                      self.Fiducial.robot_command_client)
        goal_fiducial = self.get_fiducial()
        # goal_fiducial = self.get_fiducial()
        print(goal_fiducial)
        self.walk_to_fiducial(goal_fiducial, dist_margin)

    def centering_on_target_fiducial(self, fiducial_id, dist_margin):
        """ Run centering on fiducial service:
            Taking the fid_id and dist_margin
            1. Finds for the waypoint associated with te fiducial
            2. Then, moves the robot to the waypoint
            3. Detects and centers the robot on the closet fiducial in a stable stance.
        """
        # Take the first argument as the destination fiducial number.
        # if len(args) < 1:
        #     print("Wrong input. [fid-id] [dist-margin]")
        #     return

        # should_continue, goal_fiducial, _ = self.position_body_over_fiducial(self.Fiducial.world_object_client,
        #                                                                      self.Fiducial.robot_command_client)
        goal_fiducial = self.get_target_fiducial(fiducial_id)
        print(goal_fiducial)
        if goal_fiducial is not None:
            goal_fiducial = goal_fiducial[0]
            self.walk_to_fiducial(goal_fiducial, dist_margin)


def is_boolean(input_string: str):
    """ A helper function to check if a string is 'True' or 'False'
        - Args:
            - input_string(string): an input string
        - Returns:
            - Boolean indicating if an input string is either 'True' or 'False'
    """
    return input_string in ('True', 'False')


class CarInspection:
    def __init__(self, robot):
        self.robot = robot

        # Create ArmSensorInspector object
        self._arm_sensor_inspector = None
        self.init_car_inspection()

    def init_car_inspection(self):
        if not self.robot:
            print("[CarInspection] robot 초기화 필요.")
            return

        # Create ArmSensorInspector object
        self._arm_sensor_inspector = ArmSensorInspector(self.robot)

    def set_upload_filepath(self, filepath):
        self._arm_sensor_inspector.set_upload_filepath(filepath)

    def go_to_inspection_waypoint(self, inspection_id):
        """ A function that commands the robot to go to a given inspection_id's waypoint.
            - Args:
                - inspection_id(int): the desired inspection_id
        """
        self._arm_sensor_inspector.go_to_inspection_waypoint(inspection_id=inspection_id)

    def partial_inspection(self, inspection_ids, dock_at_the_end, stow_in_between):
        """ A function that commands the robot to capture data at one or many inspection points,
            and monitors feedback.
            - Args:
                - inspection_ids(list of int): a list of ints that indicate an inspection point ID number
                - dock_at_the_end(Boolean): tells robot to dock at the end of inspection.
                                          It should be the second to last argument.
                - stow_in_between(Boolean): tells robot to stow arm in between inspection actions.
                                          It should be the last argument.
        """
        # if len(args) < 3:
        #     self.robot.logger.warning("CarInspectionClient: Invalid number of arguments! ")
        #     return

        # Cast inspection_ids to int
        # The second to last argument should be a boolean to indicate dock_at_the_end
        if is_boolean(dock_at_the_end):
            dock_at_the_end = (dock_at_the_end == 'True')
        else:
            self.robot.logger.warning(
                "CarInspectionClient: Invalid arguments. The second to last argument should be a boolean."
            )
            print("CarInspectionClient: Invalid arguments. The second to last argument should be a boolean.")
            return
        # The last argument should be a boolean to indicate dock_at_the_end stow_in_between
        if is_boolean(stow_in_between):
            stow_in_between = (stow_in_between == 'True')
        else:
            self.robot.logger.warning(
                "CarInspectionClient: Invalid arguments. The last argument should be a boolean.")
            print("CarInspectionClient: Invalid arguments. The last argument should be a boolean.")
            return
        # Send partial_inspection request
        self._arm_sensor_inspector.partial_inspection(inspection_ids, dock_at_the_end,
                                                      stow_in_between)

    def full_inspection(self, dock_at_the_end, stow_in_between):
        """ A function that commands the robot to run the full mission.
           - Args:
                - dock_at_the_end(Boolean): tells robot to dock at the end of inspection.
                                          It should be the second to last argument.
                - stow_in_between(Boolean): tells robot to stow arm in between inspection actions.
                                          It should be the last argument.
        """
        # if len(args) < 2:
        #     self.robot.logger.warning("CarInspectionClient: Invalid number of arguments! ")
        #     return

        # The second to last argument should be a boolean to indicate dock_at_the_end
        if is_boolean(dock_at_the_end):
            dock_at_the_end = (dock_at_the_end == 'True')
        else:
            self.robot.logger.warning(
                "CarInspectionClient: Invalid arguments. The second to last argument should be a boolean."
            )
            return
        # The last argument should be a boolean to indicate dock_at_the_end stow_in_between
        if is_boolean(stow_in_between):
            stow_in_between = (stow_in_between == 'True')
        else:
            self.robot.logger.warning(
                "CarInspectionClient: Invalid arguments. The last argument should be a boolean.")
            return
        self._arm_sensor_inspector.full_inspection(dock_at_the_end, stow_in_between)

    def periodic_inspection(self, inspection_interval, number_of_cycles):
        """ A function that commands the robot to perfrom full_inspection() every given inspection minute
            for given number of cycles. Robot spends (inspection_interval - robot inspection cycle time) minutes
            on the dock charging before proceeding to the next cycle.

            - Args:
                - inspection_interval(float): the periodicty of the inspection in minutes
                - number_of_cycles(int) : the frequency of the inspection in number of cycles
        """
        # if len(args) < 2:
        #     self.robot.logger.warning("CarInspectionClient: Invalid number of arguments! ")
        #     return
        self._arm_sensor_inspector.periodic_inspection(inspection_interval=inspection_interval,
                                                       number_of_cycles=number_of_cycles)
