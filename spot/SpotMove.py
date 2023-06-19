from bosdyn.api import basic_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder


class SpotMove:

    def __init__(self, robot_commander):
        self.robot_commander = robot_commander
        self.VELOCITY_BASE_SPEED   = 0.5  # m/s
        self.VELOCITY_BASE_ANGULAR = 0.8  # rad/sec

    @property
    def velocity_base_speed(self):
        return self.VELOCITY_BASE_SPEED

    @velocity_base_speed.setter
    def velocity_base_speed(self, value):
        self.VELOCITY_BASE_SPEED = value

    @property
    def velocity_base_angular(self):
        return self.VELOCITY_BASE_ANGULAR

    @velocity_base_angular.setter
    def velocity_base_angular(self, value):
        self.VELOCITY_BASE_ANGULAR = value

    def sit(self):
        return self.robot_commander.start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def stand(self):
        return self.robot_commander.start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())

    def move_forward(self):
        return self.robot_commander.velocity_cmd_helper('move_forward', v_x=self.VELOCITY_BASE_SPEED)

    def move_backward(self):
        return self.robot_commander.velocity_cmd_helper('move_backward', v_x=-self.VELOCITY_BASE_SPEED)

    def strafe_left(self):
        return self.robot_commander.velocity_cmd_helper('strafe_left', v_y=self.VELOCITY_BASE_SPEED)

    def strafe_right(self):
        return self.robot_commander.velocity_cmd_helper('strafe_right', v_y=-self.VELOCITY_BASE_SPEED)

    def turn_left(self):
        return self.robot_commander.velocity_cmd_helper('turn_left', v_rot=self.VELOCITY_BASE_ANGULAR)

    def turn_right(self):
        return self.robot_commander.velocity_cmd_helper('turn_right', v_rot=-self.VELOCITY_BASE_ANGULAR)

    def selfright(self):
        return self.robot_commander.start_robot_command('selfright', RobotCommandBuilder.selfright_command())

    def battery_change_pose(self):
        cmd = RobotCommandBuilder.battery_change_pose_command(
            dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT)
        return self.robot_commander.start_robot_command('battery_change_pose', cmd)
