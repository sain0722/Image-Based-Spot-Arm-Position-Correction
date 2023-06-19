import copy
import time

from bosdyn.api import trajectory_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2, geometry_pb2
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.util import seconds_to_duration


class SpotArm:

    def __init__(self, robot):
        # current joint state
        # sh0, sh1, el0, el1, wr0, wr1
        self.robot = robot
        self.joint_params = None
        self.robot_command_executor = robot.robot_commander

        self.JOINT_MOVE_RATE = 0.1  # arm joint move rate
        self.JOINT_TIME_SEC  = 1.5  # arm control speed

    @property
    def joint_move_rate(self):
        return self.JOINT_MOVE_RATE

    @joint_move_rate.setter
    def joint_move_rate(self, value):
        self.JOINT_MOVE_RATE = value

    @property
    def joint_time_sec(self):
        return self.JOINT_TIME_SEC

    @joint_time_sec.setter
    def joint_time_sec(self, value):
        self.JOINT_TIME_SEC = value

    def stow(self):
        stow_command    = RobotCommandBuilder.arm_stow_command()
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, stow_command)

        return self.robot_command_executor.start_robot_command('stow', synchro_command,
                                                               end_time_secs=10.0)

    def unstow(self):
        ready_command   = RobotCommandBuilder.arm_ready_command()
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, ready_command)

        return self.robot_command_executor.start_robot_command('unstow', synchro_command,
                                                               end_time_secs=10.0)

    def gripper_open(self):
        return self.robot_command_executor.start_robot_command('gripper_open',
                                                               RobotCommandBuilder.claw_gripper_open_command(),
                                                               end_time_secs=6.0)

    def gripper_close(self):
        return self.robot_command_executor.start_robot_command('gripper_close',
                                                               RobotCommandBuilder.claw_gripper_close_command(),
                                                               end_time_secs=6.0)

    def gripper_check(self):
        gripper_open_percentage = self.robot.robot_state.manipulator_state.gripper_open_percentage

    def is_arm_unstow(self):
        if self.robot.robot_state.manipulator_state.gripper_open_percentage >= 10:
            return True
        else:
            return False

    def joint_move(self, target):
        self.joint_params = self.robot.get_current_joint_state()
        if target == "sh0_right":
            self.joint_params['sh0'] = self.joint_params['sh0'] - self.JOINT_MOVE_RATE

        elif target == "sh0_left":
            self.joint_params['sh0'] = self.joint_params['sh0'] + self.JOINT_MOVE_RATE

        elif target == "sh1_up":
            self.joint_params['sh1'] = self.joint_params['sh1'] - self.JOINT_MOVE_RATE

        elif target == "sh1_down":
            self.joint_params['sh1'] = self.joint_params['sh1'] + self.JOINT_MOVE_RATE

        elif target == "el0_up":
            self.joint_params['el0'] = self.joint_params['el0'] - self.JOINT_MOVE_RATE

        elif target == "el0_down":
            self.joint_params['el0'] = self.joint_params['el0'] + self.JOINT_MOVE_RATE

        elif target == "el1_right":
            self.joint_params['el1'] = self.joint_params['el1'] + self.JOINT_MOVE_RATE

        elif target == "el1_left":
            self.joint_params['el1'] = self.joint_params['el1'] - self.JOINT_MOVE_RATE

        elif target == "wr0_up":
            self.joint_params['wr0'] = self.joint_params['wr0'] - self.JOINT_MOVE_RATE

        elif target == "wr0_down":
            self.joint_params['wr0'] = self.joint_params['wr0'] + self.JOINT_MOVE_RATE

        elif target == "wr1_right":
            self.joint_params['wr1'] = self.joint_params['wr1'] - self.JOINT_MOVE_RATE

        elif target == "wr1_left":
            self.joint_params['wr1'] = self.joint_params['wr1'] + self.JOINT_MOVE_RATE

        # elif target == "hr0":
        #     self.joint_params['hr0'] = self.joint_params['hr0'] + self.JOINT_MOVE_RATE
        #
        # elif target == "f1x":
        #     self.joint_params['f1x'] = self.joint_params['f1x'] + self.JOINT_MOVE_RATE

        return self.robot_command_executor.joint_move_cmd_helper(desc=target,
                                                                 params=self.joint_params.values(),
                                                                 time_secs=self.JOINT_TIME_SEC)

    def joint_move_manual(self, params):
        return self.robot_command_executor.joint_move_cmd_helper(desc="joint_move_manual",
                                                                 params=params,
                                                                 time_secs=self.JOINT_TIME_SEC)

    def move_to_frame_hand(self, frame_tform_hand, frame_name, end_seconds=3.0):
        arm_command = RobotCommandBuilder.arm_pose_command(
            frame_tform_hand.x, frame_tform_hand.y, frame_tform_hand.z,
            frame_tform_hand.rot.w, frame_tform_hand.rot.x,
            frame_tform_hand.rot.y, frame_tform_hand.rot.z,
            frame_name, end_seconds)

        # Open the gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Build the proto
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # self.robot_command_executor.start_robot_command('move_to_frame_hand', command)

        # Send the request
        cmd_id = self.robot_command_executor.start_robot_command('move_to_frame_hand', command,
                                                                 end_time_secs=end_seconds)

        block_until_arm_arrives(self.robot_command_executor.robot_command_client, cmd_id, end_seconds)

    def trajectory_manual(self, body_tform_hand, axis, move_rate, end_seconds=3.0):
        t = 0
        if type(body_tform_hand) == geometry_pb2.SE3Pose:
            body_tform_hand = math_helpers.SE3Pose.from_obj(body_tform_hand)

        if axis == 'x':
            t = body_tform_hand.x
        elif axis == 'y':
            t = body_tform_hand.y
        elif axis == 'z':
            t = body_tform_hand.z

        t += move_rate

        if axis == 'x':
            body_tform_hand.x = t
        elif axis == 'y':
            body_tform_hand.y = t
        elif axis == 'z':
            body_tform_hand.z = t

        arm_command = RobotCommandBuilder.arm_pose_command(
            body_tform_hand.x, body_tform_hand.y, body_tform_hand.z,
            body_tform_hand.rot.w, body_tform_hand.rot.x,
            body_tform_hand.rot.y, body_tform_hand.rot.z,
            frame_helpers.BODY_FRAME_NAME, end_seconds)

        # Open the gripper
        # gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Build the proto
        # command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
        command = arm_command
        # self.robot_command_executor.start_robot_command('move_to_frame_hand', command)

        # Send the request
        cmd_id = self.robot_command_executor.start_robot_command('move_to_frame_hand', command,
                                                                 end_time_secs=end_seconds)

        block_until_arm_arrives(self.robot_command_executor.robot_command_client, cmd_id)

    def trajectory_rotation_manual(self, body_tform_hand, rotation, end_time=1.0):
        position = body_tform_hand.position
        hand_pose = math_helpers.SE3Pose(x=position.x, y=position.y, z=position.z, rot=rotation)
        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose.to_proto(), time_since_reference=seconds_to_duration(end_time))

        # Build the trajectory proto by combining the points.
        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1])

        # Build the command by taking the trajectory and specifying the frame it is expressed
        # in.
        #
        # In this case, we want to specify the trajectory in the body's frame, so we set the
        # root frame name to the flat body frame.
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_traj, root_frame_name=BODY_FRAME_NAME)

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Keep the gripper opened the whole time.
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0, build_on_command=robot_command)

        # Send the trajectory to the robot.
        cmd_id = self.robot_command_executor.start_robot_command('trajectory', robot_command,
                                                                 end_time_secs=10.0)

        self.robot_command_executor.wait_until_arm_arrives(cmd_id, timeout=3)
        time.sleep(0.5)
        return cmd_id

    # def trajectory(self, x, y1, y2, y3, z):
    #     # Use the same rotation as the robot's body.
    #     rotation = math_helpers.Quat()
    #
    #     # Define times (in seconds) for each point in the trajectory.
    #     t_first_point = 4.0  # first point starts at t = 0 for the trajectory.
    #     t_second_point = 4.0
    #     t_third_point = 8.0
    #
    #     # Build the points in the trajectory.
    #     hand_pose1 = math_helpers.SE3Pose(x=x, y=y1, z=z, rot=rotation)
    #     hand_pose2 = math_helpers.SE3Pose(x=x, y=y2, z=z, rot=rotation)
    #     hand_pose3 = math_helpers.SE3Pose(x=x, y=y3, z=z, rot=rotation)
    #
    #     # Build the points by combining the pose and times into protos.
    #     traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
    #         pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(t_first_point))
    #     traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
    #         pose=hand_pose2.to_proto(), time_since_reference=seconds_to_duration(t_second_point))
    #     traj_point3 = trajectory_pb2.SE3TrajectoryPoint(
    #         pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(t_third_point))
    #
    #     # Build the trajectory proto by combining the points.
    #     hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2, traj_point3])
    #
    #     # Build the command by taking the trajectory and specifying the frame it is expressed
    #     # in.
    #     #
    #     # In this case, we want to specify the trajectory in the body's frame, so we set the
    #     # root frame name to the flat body frame.
    #     arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
    #         pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)
    #
    #     # Pack everything up in protos.
    #     arm_command = arm_command_pb2.ArmCommand.Request(
    #         arm_cartesian_command=arm_cartesian_command)
    #
    #     synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
    #         arm_command=arm_command)
    #
    #     robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
    #
    #     # Keep the gripper closed the whole time.
    #     # robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
    #     #     0, build_on_command=robot_command)
    #
    #     # Send the trajectory to the robot.
    #     cmd_id = self.robot_command_executor.start_robot_command('trajectory', robot_command,
    #                                                              end_time_secs=10.0)
    #
    #     self.robot_command_executor.wait_until_arm_arrives(cmd_id)

    # 메인 Trajectory 함수
    def trajectory(self, position, rotation, frame_name, end_time=2.0):
        # Use the same rotation as the robot's body.
        rotation = math_helpers.Quat(w=rotation['w'], x=rotation['x'], y=rotation['y'], z=rotation['z'])
        # rotation = self.robot.get_current_hand_position("hand").rotation
        # position = self.robot.get_current_hand_position("hand").position

        # x = position.x
        # y = position.y
        # z = position.z
        # # Build the points in the trajectory.
        hand_pose = math_helpers.SE3Pose(x=position['x'], y=position['y'], z=position['z'], rot=rotation)

        # position = geometry_pb2.Vec3(x=x, y=y, z=z)
        # this_se3_pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)

        # print("[Debug] position\n" + self.robot.get_current_hand_position("hand").position.__str__())
        # print("[Debug] rotation\n" + self.robot.get_current_hand_position("hand").rotation.__str__())
        # Build the points by combining the pose and times into protos.
        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose.to_proto(), time_since_reference=seconds_to_duration(end_time))

        # Build the trajectory proto by combining the points.
        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1])

        # Build the command by taking the trajectory and specifying the frame it is expressed
        # in.
        #
        # In this case, we want to specify the trajectory in the body's frame, so we set the
        # root frame name to the flat body frame.
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_traj, root_frame_name=frame_name)

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Keep the gripper opened the whole time.
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0, build_on_command=robot_command)

        # Send the trajectory to the robot.
        cmd_id = self.robot_command_executor.start_robot_command('trajectory', robot_command,
                                                                 end_time_secs=10.0)

        self.robot_command_executor.wait_until_arm_arrives(cmd_id)
        time.sleep(0.5)
        return cmd_id

    def trajectory_pos_rot(self, x, y, z, rot_x=0, rot_y=0, rot_z=0, rot_w=1):
        # Use the same rotation as the robot's body.
        rotation = math_helpers.Quat(w=rot_w, x=rot_x, y=rot_y, z=rot_z)
        # rotation = self.robot.get_current_hand_position("hand").rotation
        # position = self.robot.get_current_hand_position("hand").position

        # x = position.x
        # y = position.y
        # z = position.z
        # # Build the points in the trajectory.
        hand_pose = math_helpers.SE3Pose(x=x, y=y, z=z, rot=rotation)

        # position = geometry_pb2.Vec3(x=x, y=y, z=z)
        # this_se3_pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)

        # print("[Debug] position\n" + self.robot.get_current_hand_position("hand").position.__str__())
        # print("[Debug] rotation\n" + self.robot.get_current_hand_position("hand").rotation.__str__())
        # Build the points by combining the pose and times into protos.
        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose.to_proto(), time_since_reference=seconds_to_duration(2.0))

        # Build the trajectory proto by combining the points.
        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1])

        # Build the command by taking the trajectory and specifying the frame it is expressed
        # in.
        #
        # In this case, we want to specify the trajectory in the body's frame, so we set the
        # root frame name to the flat body frame.
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_traj, root_frame_name=BODY_FRAME_NAME)

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Keep the gripper closed the whole time.
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0, build_on_command=robot_command)

        # Send the trajectory to the robot.
        cmd_id = self.robot_command_executor.start_robot_command('trajectory', robot_command,
                                                                 end_time_secs=10.0)

        self.robot_command_executor.wait_until_arm_arrives(cmd_id)
        time.sleep(0.5)
        return cmd_id

    def trajectory_odometry(self, x, y, z, rot_x=0, rot_y=0, rot_z=0, rot_w=1):
        rotation = math_helpers.Quat(w=rot_w, x=rot_x, y=rot_y, z=rot_z)
        hand_pose = math_helpers.SE3Pose(x=x, y=y, z=z, rot=rotation)

        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(pose=hand_pose.to_proto(),
                                                        time_since_reference=seconds_to_duration(2.5))

        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1])

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(pose_trajectory_in_task=hand_traj,
                                                                            root_frame_name=ODOM_FRAME_NAME)

        arm_command = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0, build_on_command=robot_command)

        cmd_id = self.robot_command_executor.start_robot_command('trajectory', robot_command, end_time_secs=10.0)
        self.robot_command_executor.wait_until_arm_arrives(cmd_id)

        # command = RobotCommandBuilder.freeze_command()
        # cmd_id = self.robot_command_executor.start_robot_command('freeze', command, end_time_secs=10.0)

        return cmd_id
