import time

from bosdyn.api import arm_command_pb2, synchronized_command_pb2, robot_command_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.robot_command import RobotCommandBuilder


def try_grpc(desc, thunk):
    try:
        return thunk()
    except (ResponseError, RpcError) as err:
        # self.add_message("Failed {}: {}".format(desc, err))
        # message = "{}: {}".format(desc, err)
        message = err.error_message
        print(err)
        return message
    except LeaseBaseError as err:
        message = err
        return message
        # return "Failed {}: {}".format(desc, err)


def try_grpc_async(desc, thunk):
    def on_future_done(fut):
        try:
            fut.result()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            # self.add_message("Failed {}: {}".format(desc, err))
            message = "{}: {}".format(desc, err)
            print(message)
            return message

    future = thunk()
    future.add_done_callback(on_future_done)


def make_robot_command(arm_joint_traj, gripper_open_flag=False):
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)

    if gripper_open_flag:
        # Keep the gripper open the whole time, so we can get an image.
        arm_sync_robot_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0, build_on_command=arm_sync_robot_cmd)

    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


class RobotCommandExecutor:
    def __init__(self, client):
        self.robot_command_client  = client
        self.VELOCITY_CMD_DURATION = 0.6  # seconds

    def start_robot_command(self, desc, command_proto, end_time_secs=None):
        def _start_command():
            return self.robot_command_client.robot_command(lease=None, command=command_proto,
                                                           end_time_secs=end_time_secs)

        return try_grpc(desc, _start_command)

    def velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        return self.start_robot_command(
                desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
                end_time_secs=time.time() + self.VELOCITY_CMD_DURATION)

    def joint_move_cmd_helper(self, params, desc='', time_secs=1.0, flag=False):
        sh0, sh1, el0, el1, wr0, wr1 = params
        # time_secs = JOINT_TIME_SEC
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=time_secs)

        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        arm_command = make_robot_command(arm_joint_traj, flag)

        # Open the gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Build the proto
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        cmd_id = self.start_robot_command(desc=desc, command_proto=command)
        return cmd_id

    def wait_until_arm_arrives(self, cmd_id, timeout=5):
        # Wait until the arm arrives at the goal.
        start_time = time.time()
        end_time = start_time + timeout
        while time.time() < end_time:
            feedback_resp = self.robot_command_client.robot_command_feedback(cmd_id)
            print('Distance to final point: ' + '{:.2f} meters'.format(
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                arm_cartesian_feedback.measured_pos_distance_to_goal) + ', {:.2f} radians'.format(
                    feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                    arm_cartesian_feedback.measured_rot_distance_to_goal))

            if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                # if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal < 0.03:
                print('Move complete.')
                break

            # if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_pos_distance_to_goal == 0 and \
            #    feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal == 0:
            #     print("Move Arrived.")
            #     break

            time.sleep(0.1)

    def feedback_test(self, cmd_id):
        feedback_resp = self.robot_command_client.robot_command_feedback(cmd_id)

        return feedback_resp
