import time

import bosdyn.client.util
from bosdyn.api import basic_command_pb2
from bosdyn.client import UnableToConnectToRobotError, TimedOutError
from bosdyn.client.power import CommandTimedOutError
from bosdyn.client.robot_command import RobotCommandBuilder, CommandFailedError


def create_robot(hostname, username, password):
    try:
        sdk = bosdyn.client.create_standard_sdk('TWIM')
        robot = sdk.create_robot(hostname)
        # bosdyn.client.util.authenticate(robot)
        robot.authenticate(username=username, password=password)
        return robot

    except UnableToConnectToRobotError as exc:
        print(exc)
        # quit()


def blocking_stand(command_client, timeout_sec=10, update_frequency=1.0, params=None):
    """Helper function which uses the RobotCommandService to stand.

    Blocks until robot is standing, or raises an exception if the command times out or fails.

    Args:
        command_client: RobotCommand client.
        timeout_sec: Timeout for the command in seconds.
        update_frequency: Update frequency for the command in Hz.
        params(spot.MobilityParams): Spot specific parameters for mobility commands to optionally set say body_height

    Raises:
        CommandFailedError: Command feedback from robot is not STATUS_PROCESSING.
        bosdyn.client.robot_command.CommandTimedOutError: Command took longer than provided
            timeout.
    """

    start_time = time.time()
    end_time = start_time + timeout_sec
    update_time = 1.0 / update_frequency

    stand_command = RobotCommandBuilder.synchro_stand_command(params=params)
    command_id = command_client.robot_command(stand_command, timeout=timeout_sec)

    now = time.time()
    while now < end_time:
        time_until_timeout = end_time - now
        rpc_timeout = max(time_until_timeout, 1)
        start_call_time = time.time()
        try:
            response = command_client.robot_command_feedback(command_id, timeout=rpc_timeout)
            mob_feedback = response.feedback.synchronized_feedback.mobility_command_feedback
            mob_status = mob_feedback.status
            stand_status = mob_feedback.stand_feedback.status
        except TimedOutError:
            # Excuse the TimedOutError and let the while check bail us out if we're out of time.
            pass
        else:
            if mob_status != basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING:
                raise CommandFailedError('Stand (ID {}) no longer processing (now {})'.format(
                    command_id,
                    basic_command_pb2.RobotCommandFeedbackStatus.Status.Name(mob_status)))
            if stand_status == basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING:
                return
        delta_t = time.time() - start_call_time
        time.sleep(max(min(delta_t, update_time), 0.0))
        now = time.time()

    raise CommandTimedOutError(
        "Took longer than {:.1f} seconds to assure the robot stood.".format(now - start_time))
