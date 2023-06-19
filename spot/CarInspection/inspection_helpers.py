''' Class to enable autonomous inspection with the gripper camera
    Refer to "car_inspection.py" for an example of how to use this class. 
'''
import csv
import os
import shutil
import ssl
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bosdyn.api.geometry_pb2 import Quaternion
from bosdyn.client import math_helpers
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.mission.client
from bosdyn.api import geometry_pb2, gripper_camera_param_pb2, robot_state_pb2
from bosdyn.api.autowalk.walks_pb2 import (Action, ActionWrapper, BatteryMonitor, Element,
                                           FailureBehavior, Walk)
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.api.mission import mission_pb2, util_pb2
from bosdyn.api.robot_state_pb2 import ManipulatorState
from bosdyn.client.autowalk import AutowalkClient
from bosdyn.client.data_acquisition_helpers import make_time_query_params, _LOGGER
from bosdyn.client.docking import DockingClient, docking_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.power import PowerClient, power_on
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import secs_to_hms

ACTION_NAME = "Arm Pointing"
ANSWER_TRYAGAIN = "Try Again"
ANSWER_SKIP = "Skip"
ANSWER_DOCK = "Return to Dock and End Mission"
# Answers questions raised during mission execution on the robot - set to ANSWER_SKIP
MISSION_QUESTION_ANSWER_CHOICE = ANSWER_SKIP


class ArmSensorInspector:
    def __init__(self, robot):
        """
            - Args:
                - robot(BD SDK Robot): a robot object
                - upload_filepath(string): a filepath to an Autowalk .walk folder that contains
                                     edge_snapshots, waypoint_snapshots, missions,
                                      autowalk_metadata, and graph
        """

        self.robot = robot

        # BD SDK Robot
        self._robot = robot.robot

        # Filepath for edge_snapshots, waypoint_snapshots, missions, autowalk_metadata, and graph.
        self._upload_filepath = ""

        self._power_client = None
        self._graph_nav_client = None
        self._autowalk_service_client = None
        self._mission_client = None
        self._lease_client = None
        self._robot_command_client = None
        self._robot_state_client = None
        self._docking_client = None
        self._base_mission = None
        self._base_inspection_elements = None
        self._base_inspection_ids = None
        self._num_of_inspection_elements = None
        self._inspection_data_header = None
        self._inspection_data = None
        self._summary_header = None

        # Maps inspection_ids  to node_ids to provide element-wise feedback for missions
        self._node_map = {}
        # Store the most recent knowledge of the state of the self._robot based on rpc calls.
        self._current_graph = None
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot

        # Folder to save inspection data locally
        self._inspection_folder = os.getcwd() + '/'
        # String to differentiate the captured images
        self._image_suffix = ''

        # Number of inspection points completed -  this is computed by counting number of images
        self._inspection_elements_completed = 0
        # Number of failed inspections due to arm pointing failure
        self._arm_pointing_failure_count = 0

    def init_arm_sensor_inspector(self):
        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()
        # Create a power client for the robot.
        self._power_client = self.robot.power_client
        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self.robot.graph_nav_client
        # Create the client for Autowalk Service
        self._autowalk_service_client = self._robot.ensure_client(
            AutowalkClient.default_service_name)
        # Create the client for Mission Service
        self._mission_client = self.robot.mission_client
        # Create the client for lease
        self._lease_client = self.robot.lease_client
        # Create the robot command client
        self._robot_command_client = self.robot.robot_command_client
        # Create the robot state client
        self._robot_state_client = self.robot.robot_state_client

        # Create a docking client
        self._docking_client = self._robot.ensure_client(DockingClient.default_service_name)
        # Base mission
        self._base_mission = self._load_mission(self._upload_filepath)
        # Dict of inspection_elements and a list inspection_ids
        self._base_inspection_elements, self._base_inspection_ids = self._get_base_inspections()
        # Upload the graph and snapshots to the robot.
        self._run_with_fallback(self._upload_map_and_localize)
        # Number of inspection points completed
        self._num_of_inspection_elements = self._get_num_of_inspection_elements(self._base_mission)
        # The data header for inspection data saved in csv
        self._inspection_data_header = [
            'Cycle #', 'Inspection Start Time', 'Inspection End Time', 'Cycle Time in min',
            '# of required inspections', '# of completed inspections', '# of failed inspections',
            '# of arm pointing failures', 'Time Spent Docked',
            'Battery Level at the start of the cycle', 'Battery Level at the end of the cycle',
            'Battery Consumption', 'Battery Min Temperature at the start of the cycle',
            'Battery Max Temperature at the start of the cycle',
            'Battery Min Temperature at the end of the cycle',
            'Battery Max Temperature at the end of the cycle', 'Mission Succeeded?'
        ]
        # Initialize the inspection data which is saved to csv during periodic inspection
        self._inspection_data = [0 for i in range(len(self._inspection_data_header))]
        # Initialize 'Mission Succeeded?' in self._inspection_data to False
        self._inspection_data[self._inspection_data_header.index("Mission Succeeded?")] = False
        # The header for summarizing a periodic inspection
        self._summary_header = [
            "Periodic mission start datetime", "Periodic mission end datetime",
            "Periodic mission duration(hours)", "Cycles Required", "Cycles Completed",
            "Cycles Failed", "Inspections Required", "Inspections Completed", "Inspection Failures",
            "Arm Pointing Failures", "Average Cycle Time(minutes)", "Median Cycle Time(minutes)",
            "STDEV Cycle Time(minutes)", "Q1 Cycle Time(minutes)", "Q3 Cycle Time(minutes)",
            "Min cycle time", "Max cycle time", "Average Battery", "Median Battery",
            "STDEV Battery", "Q1 Battery", "Q3 Battery", "Min Battery", "Max Battery"
        ]

    def set_upload_filepath(self, upload_filepath):
        self._upload_filepath = upload_filepath
        self.init_arm_sensor_inspector()

    # def get_current_robot_state(self):
    #     self._async_tasks.update()
    #     return self._robot_state_task.proto

    def Move_and_Take_a_picture(self, sel_axis):
        state = self.get_current_robot_state()
        # ARM의 현재 위치 GET
        state_list = state.kinematic_state.ListFields()
        pose_info_depth_4 = state_list[4][1]
        pose_info_depth_3 = pose_info_depth_4.ListFields()[0][1]
        pose_info_depth_2 = pose_info_depth_3.get('hand')
        pose_info_depth_1 = pose_info_depth_2.ListFields()[1][1]
        print("현재 Arm 위치 : X=%f,   Y=%f,    Z=%f" %(pose_info_depth_1.position.x,
                                                    pose_info_depth_1.position.y,
                                                    pose_info_depth_1.position.z)
              )

        if sel_axis == 'x' or 'X':
            hand_x = pose_info_depth_1.position.x - 0.05
            #print("hand_x : ", hand_x)

        elif sel_axis == 'y' or 'Y':
            hand_y = pose_info_depth_1.position.y - 0.05
            #print("hand_y : ", hand_y)

        elif sel_axis == 'z' or 'Z':
            hand_z = pose_info_depth_1.position.z - 0.05
            #print("hand_z : ", hand_z)
        else:
            return False

        print("이동 Arm 위치 : X=%f,   Y=%f,    Z=%f" % (hand_x, hand_y, hand_x))
        hand_rx = pose_info_depth_1.rotation.x
        #print("hand_rx : ", hand_rx)
        hand_ry = pose_info_depth_1.rotation.y
        #print("hand_ry : ", hand_ry)
        hand_rz = pose_info_depth_1.rotation.z
        #print("hand_rz : ", hand_rz)
        hand_w = pose_info_depth_1.rotation.w
        #print("hand_w : ", hand_w)

        hand_ewrt_flat_body = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z)
        flat_body_Q_hand = Quaternion(w=hand_w, x=hand_rx, y=hand_ry, z=hand_rz)
        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        odom_T_flat_body = get_a_tform_b(state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
                                                           odom_T_hand.rot.w, odom_T_hand.rot.x, odom_T_hand.rot.y,
                                                           odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds=2)

        # # Make the open gripper RobotCommand
        # gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(self.hand_claw)
        #
        # # Combine the arm and gripper commands into one RobotCommand
        # command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        self._robot_command_client.robot_command(arm_command)
        time.sleep(5)
        return True

    def full_inspection(self, dock_at_the_end=True, stow_in_between=False):
        ''' A function that commands the robot to run the full mission and downloads captured images at the end.
            - Args:
                - dock_at_the_end(Boolean) : tells robot to dock at the end of inspection
                - stow_in_between(Boolean) :  tells robot to stow arm in between inspection actions
            - Returns:
                - Boolean indicating if inspection is successful
        '''
        self._robot.logger.info('ArmSensorInspector: Running full_inspection')
        # Check to see if the self._inspection_folder is set by periodic_inspection fucntion
        # This is because periodic_inspection calls full_inspection.
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        timestamp_mdy = datetime.now().strftime("%m%d%Y")
        if self._inspection_folder.find("periodic_inspection") == -1:
            # It is not set by periodic_inspection, so set inspection folder as full_inspection
            self._inspection_folder = os.getcwd(
            ) + '/full_inspections/full_inspection_' + timestamp_mdy + '/'
        # Set the mission as the self._base_mission
        mission = self._base_mission
        # Check if mission is valid
        if not self._base_mission:
            self._robot.logger.error('ArmSensorInspector: Invalid Mission!')
            return
        # Set the mission_name
        self._set_mission_name(mission, mission_name='Full_Inspection')
        # Determine docking behavior at the end of missions
        if dock_at_the_end:
            # Tell Robot to dock after completion
            self._enable_dock_after_completion(mission)
        else:
            # Tell Robot not to dock after completion
            self._disable_dock_after_completion(mission)
        # Determine stowing behavior in between inspection actions
        if stow_in_between:
            # Force stow arm in between inspection actions
            self._enable_stow_arm_in_between_inspection_actions(mission)
        else:
            # Tell Robot not to stow arm in between inspection actions
            self._disable_stow_arm_in_between_inspection_actions(mission)
        # Set the speed for the joint move
        self._set_joint_move_speed(mission, joint_move_speed="FAST")
        # Set travel_speed
        self._set_travel_speed(mission, travel_speed="FAST")
        # Set failure behaviour
        self._set_failure_behavior(mission)
        # Set global mission parameters
        self._set_global_parameters(mission)
        # Execute mission on robot
        start_time = time.time()
        success = self._execute_mission_on_robot(mission)
        end_time = time.time()
        # Log status
        self._log_command_status(command_name="full_inspection", status=success)
        # Reset the number of inspection points completed - this is computed by counting number of images
        self._inspection_elements_completed = 0
        # Retrieve the images from the DAQ
        self._download_image_from_robot(start_time, end_time, timestamp)
        return success

    def partial_inspection(self, inspection_ids, dock_at_the_end=True, stow_in_between=False):
        ''' A function that commands the robot to capture data at one or many inspection points, 
            and monitors feedback. It also downloads captured images at the end.
            - Args:
                - inspection_ids(list): a list of ints that indicate an inspection point ID number
                - dock_at_the_end(Boolean) : tells robot to dock at the end of inspection
                - stow_in_between(Boolean) :  tells robot to stow arm in between inspection actions
            - Returns:
                - Boolean indicating if inspection is successful
        '''
        self._robot.logger.info('ArmSensorInspector: Running partial_inspection')
        print('ArmSensorInspector: Running partial_inspection')
        # Set inspection folder as periodic_inspections
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        self._inspection_folder = os.getcwd(
        ) + '/partial_inspections/partial_inspection_' + timestamp + '/'
        # Curate a mission based on inspection_ids
        curated_mission = self._construct_mission_given_inspection_ids(inspection_ids)
        # Check if curated mission is valid
        if not curated_mission:
            self._robot.logger.error('ArmSensorInspector: Invalid Mission!')
            return
        # Set the mission_name
        self._set_mission_name(curated_mission, mission_name='Partial_Inspection')
        # Determine docking behavior at the end of missions
        if dock_at_the_end:
            # Tell Robot to dock after completion
            self._enable_dock_after_completion(curated_mission)
        else:
            # Tell Robot not to dock after completion
            self._disable_dock_after_completion(curated_mission)
        # Determine stowing behavior in between inspection actions
        if stow_in_between:
            # Force stow arm in between inspection actions
            self._enable_stow_arm_in_between_inspection_actions(curated_mission)
        else:
            # Tell Robot not to stow arm in between inspection actions
            self._disable_stow_arm_in_between_inspection_actions(curated_mission)
        # Set the speed for the joint move
        self._set_joint_move_speed(curated_mission)
        # Set travel_speed
        self._set_travel_speed(curated_mission)
        # Set failure behaviour
        self._set_failure_behavior(curated_mission)
        # Set global mission parameters
        self._set_global_parameters(curated_mission)
        # Execute mission on robot
        start_time = time.time()
        success = self._execute_mission_on_robot(curated_mission)
        end_time = time.time()
        # Log status
        self._log_command_status(command_name="partial_inspection", status=success)
        # Reset the number of inspection points completed
        self._inspection_elements_completed = 0
        # Retrieve the images from the DAQ
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

        self._download_image_from_robot(start_time, end_time, timestamp)
        return success

    def periodic_inspection(self, inspection_interval, number_of_cycles):
        ''' A function that commands the robot to perform full_inspection() every given inspection minute
            for given number of cycles. Robot spends (inspection_interval - robot inspection cycle time) minutes
            on the dock charging before proceeding to the next cycle.
            - Args:
                - inspection_interval(double): the periodicty of the inspection in minutes
                - number_of_cycles(int) : the frequency of the inspection in number of cycles
            - Returns:
                - Boolean indicating if inspection is successful
        '''
        self._robot.logger.info('ArmSensorInspector: Running periodic_inspection')
        # Set inspection folder as periodic_inspections
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        self._inspection_folder = os.getcwd(
        ) + '/periodic_inspections/periodic_inspection_' + timestamp + '/'
        # Set the name for the csv
        csv_filepath = Path(self._inspection_folder + 'inspection_data_' + timestamp + '.csv')
        # Write the HEADER to the csv
        self._write_inspection_data(self._inspection_folder, csv_filepath,
                                    self._inspection_data_header)
        # Set periodic_mission_start_datetime
        periodic_mission_start_datetime = datetime.now()
        # Initialize cycle number
        cycle = 1
        # Run full_inspection for given number_of_cycles
        while cycle <= number_of_cycles:
            # Initialize time
            start_time = datetime.now()
            inspection_cycle_time = timedelta(minutes=inspection_interval, seconds=0)
            inspection_cycle_endtime = start_time + inspection_cycle_time
            while datetime.now() < inspection_cycle_endtime:
                # Set the image suffix as the cycle num to create a seperate dir for the images taken in a cycle
                self._image_suffix = "cycle_" + str(cycle) + "_" + datetime.now().strftime(
                    "%m%d%Y_%H%M%S") + "/"
                self._robot.logger.info("ArmSensorInspector: Performing inspection cycle#: " +
                                        str(cycle))
                # Check the status of the battery before the inspection begins
                start_battery, start_min_temp, start_max_temp = self._run_with_fallback(
                    self._battery_status)
                start_t = datetime.now()
                # Send the full_inspection request
                cycle_success = self.full_inspection(dock_at_the_end=True)
                end_t = datetime.now()
                # Check the status of the battery after the inspection is done
                end_battery, end_min_temp, end_max_temp = self._run_with_fallback(
                    self._battery_status)
                cycle_time = (end_t - start_t).total_seconds() / 60  # mins
                self._robot.logger.info("ArmSensorInspector: Completed cycle#: " + str(cycle) +
                                        " in (mins): " + str(cycle_time))

                # Calculate the time to spend on the dock before proceeding to the next inspection_cycle
                wait_time = (inspection_cycle_endtime - datetime.now()).total_seconds()  #seconds
                wait_timeout = time.time() + wait_time
                # Wait and charge on the dock
                self._robot.logger.info("ArmSensorInspector: Waiting on the dock for " +
                                        str(wait_time / 60) + " mins...")
                # Log inspection data
                battery_drop = end_battery - start_battery
                # Log failed inspection
                failed_inspections = self._num_of_inspection_elements - self._inspection_elements_completed
                # Update self._inspection_data
                self._inspection_data = [
                    cycle,
                    start_t,
                    end_t,
                    cycle_time,
                    self._num_of_inspection_elements,
                    self._inspection_elements_completed,
                    failed_inspections,
                    self._arm_pointing_failure_count,
                    wait_time / 60,
                    start_battery,
                    end_battery,
                    battery_drop,
                    start_min_temp,
                    start_max_temp,
                    end_min_temp,
                    end_max_temp,
                    cycle_success,
                ]
                # Write the inspection self._inspection_data to the csv
                self._write_inspection_data(self._inspection_folder, csv_filepath,
                                            self._inspection_data)
                while time.time() < wait_timeout:
                    time.sleep(1)
                # Increment cycle number
                cycle += 1
                # Reset the number of inspection points completed for the upcoming cycle
                self._inspection_elements_completed = 0
        # Set periodic_mission_end_datetime to now
        periodic_mission_end_datetime = datetime.now()
        # Computing periodic Mission Summary metrics
        self._write_periodic_mission_summary(csv_filepath, number_of_cycles,
                                             periodic_mission_start_datetime,
                                             periodic_mission_end_datetime)
        self._robot.logger.info("ArmSensorInspector: Completed the requested " +
                                str(number_of_cycles) + " inspection cycles!")

    def go_to_dock(self, dock_id=None):
        ''' A function that commands the robot to go to a given dock_id if the robot is not already 
            at that dock_id.
            - Args:
                - dock_id(int): the ID associated with the dock
            - Returns:
                - Boolean indicating if docking operation is successful
        '''
        self._robot.logger.info('ArmSensorInspector: Running go_to_dock')
        # Create a mission without any element. This should automatically
        # succeed and return to dock
        curated_mission = self._create_identical_mission_without_elements()
        # Making sure we enable_dock_after_completion
        self._enable_dock_after_completion(curated_mission)
        # Set travel_speed
        self._set_travel_speed(curated_mission)
        # Set failure behaviour
        self._set_failure_behavior(curated_mission)
        # Set global mission parameters
        self._set_global_parameters(curated_mission)
        # Set the mission_name
        self._set_mission_name(curated_mission, mission_name='Go_To_Dock')
        # Check if the mission contains docks
        if (len(curated_mission.docks) == 0):
            self._robot.logger.info("ArmSensorInspector: Mission has no docks registered. Quit.")
            return False
        # If the mission contains a single dock and dock_id is not specified, go to the dock in the mission
        elif (len(curated_mission.docks) == 1) and (dock_id is None):
            # Check if the robot is docked
            if (self._docking_client.get_docking_state().status ==
                    docking_pb2.DockState.DOCK_STATUS_DOCKED):
                self._robot.logger.info("ArmSensorInspector: the robot is already on the dock!")
                return True
            success = self._execute_mission_on_robot(curated_mission)
            # Log status
            self._log_command_status(command_name="go_to_dock", status=success)
            return success
        # Check if there are multiple docks in the mission and a dock id is not specified
        elif (len(curated_mission.docks) > 1) and (dock_id is None):
            self._robot.logger.info(
                "ArmSensorInspector: Mission has multiple dock locations. Specify dock number.")
            return False
        # Find the specified Dock ID in the mission docks and go to the dock if it exists in the mission
        self._robot.logger.info("ArmSensorInspector: Dock ID is specified as " + str(dock_id))
        for dock in curated_mission.docks:
            if (dock.dock_id == dock_id):
                # Check if the robot is docked on the given dock_id
                if (self._docking_client.get_docking_state().status ==
                        docking_pb2.DockState.DOCK_STATUS_DOCKED):
                    self._robot.logger.info("ArmSensorInspector: the robot is already on the dock!")
                    return False
                # Pick this dock as the destination and create a new mission
                mission_name = self._base_mission.mission_name
                global_parameters = self._base_mission.global_parameters
                playback_mode = self._base_mission.playback_mode
                curated_mission = Walk(mission_name=mission_name,
                                       global_parameters=global_parameters,
                                       playback_mode=playback_mode, docks=[dock])
                # command the robot to go to the dock
                success = self._execute_mission_on_robot(curated_mission)
                # Log status
                self._log_command_status(command_name="go_to_dock", status=success)
                return success
        self._robot.logger.info("ArmSensorInspector: The specified Dock ID is not in the mission")
        return False

    def go_to_inspection_waypoint(self, inspection_id):
        ''' A function that commands the robot to go to a given inspection_id's waypoint. 
            - Args:
                - inspection_id(int): the ID associated with the inspection
            - Returns:
                - Boolean indicating if docking operation is successful
        '''
        self._robot.logger.info(
            'ArmSensorInspector: Running go_to_inspection_waypoint for inspection_id ' +
            str(inspection_id))
        # Create a mission without any element
        curated_mission = self._create_identical_mission_without_elements()
        # Get mission_element using the inspection_id
        mission_element = self._base_inspection_elements.get(inspection_id)
        # Quit if inspection_id is invalid
        if not mission_element:
            self._robot.logger.info((
                'ArmSensorInspector: Invalid inspection_id: {}! It is not in the list of inspection_ids : {}! '
            ).format(inspection_id, self._base_inspection_ids))
            return
        # Get the target corresponding to the inspection_id
        target = mission_element.target
        # Create an ActionWrapper that tells the robot to pose at the inspection_waypoint
        wrapper = ActionWrapper(robot_body_pose=ActionWrapper.RobotBodyPose())
        # Create an action that tells the robot to wait in place
        action = Action(sleep=Action.Sleep(duration=duration_pb2.Duration(seconds=1)))
        # Create a mission element using the info above add it to the empty curated_mission
        element = Element(target=target, action=action, action_wrapper=wrapper)
        curated_mission.elements.append(element)
        # Set the mission_name
        self._set_mission_name(curated_mission, mission_name='Go_To_Inspection_Waypoint')
        # Tell Robot not to dock after completion
        self._disable_dock_after_completion(curated_mission)
        # Set failure behaviour
        self._set_failure_behavior(curated_mission)
        # Set global mission parameters
        self._set_global_parameters(curated_mission)
        # command the robot to go_to_inspection_waypoint
        success = self._execute_mission_on_robot(curated_mission)
        # Log status
        self._log_command_status(command_name="go_to_inspection_waypoint", status=success)
        return True

    def _load_mission(self, upload_filepath):
        ''' A helper function that loads a mission folder from disk.
            - Args:
                - upload_filepath(string): a filepath to Autowalk .walk folder that contains map data and mission folder
            - Returns:
                - base_mission(walks_pb2.Walk): the original/base mission obtained in the uploaded .walk folder
        '''
        with open(upload_filepath + '/missions/autogenerated.walk', "rb") as mission_file:
            # Load the Autowalk .walk file that contains Arm Sensor Pointing actions
            data = mission_file.read()
            base_mission = Walk()
            base_mission.ParseFromString(data)
            self._robot.logger.info("ArmSensorInspector: Loaded AutoWalk File")
        return base_mission

    def _execute_mission_on_robot(self, mission, mission_timeout=60):
        ''' A helper function that loads and plays a mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - mission_timeout(seconds): a time when the mission should pause execution in seconds
            - Returns:
                - Boolean indicating if mission execution is successful
        '''
        # Forcibly take the lease
        self._lease_client.take()
        # Load the mission onto the robot
        load_mission_success = self._run_with_fallback(self._load_mission_to_robot, mission)
        # Return if mission loading is not successful
        if not load_mission_success:
            self._robot.logger.info("ArmSensorInspector: Mission loading failed!")
            print("ArmSensorInspector: Mission loading failed!")
            return False
        # Print the relevant mission info
        self._print_mission_info(mission)
        # Play the mission if loading is successful and return status
        return self._run_with_fallback(self._play_mission, mission_timeout)

    def _load_mission_to_robot(self, mission):
        ''' A helper function that loads a mission to the robot by sending it to the autowalk service.
            This function ensures that robot motor power is on and robot is localized to uploaded map.
            It also disables battery monitor to prevent battery level requirements to start a mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
            - Returns:
                - Boolean indicating if mission loading is successful
        '''
        # Determine if the robot motors are powered on. If not, turn them on
        self._ensure_motor_power_is_on()
        # Determine if the robot is localized to the uploaded graph. If not, localize the robot.
        self._ensure_robot_is_localized()
        # Disable battey monitor
        self._disable_battery_monitor(mission)
        # Prepare body lease in order to load mission to the robot
        body_lease = self._lease_client.lease_wallet.advance()
        # Load the mission to the robot
        load_autowalk_response = self._autowalk_service_client.load_autowalk(mission, [body_lease])
        if load_autowalk_response.status == load_autowalk_response.STATUS_OK:
            self._robot.logger.info("ArmSensorInspector: successfuly loaded the mission to robot!")
            # Associate element identifiers to mission element's name
            self._node_map = self._generate_node_map(mission.elements,
                                                     load_autowalk_response.element_identifiers)
            return True
        self._robot.logger.error(
            "ArmSensorInspector: Problem loading the mission to robot! Status_code: " +
            str(load_autowalk_response.status))
        return False

    def _play_mission(self, mission_timeout=60):
        ''' A helper function that plays a mission that is already on the robot
            - Args:
                - mission_timeout(seconds): a time when the mission should pause execution
            - Returns:
                - Boolean indicating if mission playing is successful
        '''
        self._robot.logger.info('ArmSensorInspector: Running mission...')
        print('ArmSensorInspector: Running mission...')
        # Initialize self._arm_pointing_failure_count
        self._arm_pointing_failure_count = 0
        # Initialize the status of the mission
        status = self._mission_client.get_state().status
        # Initialize inspection_action_start_time to now
        inspection_action_start_time = time.time()
        # Initialize play request times
        play_request_rate = 0.5  # seconds
        play_request_time = time.time() + play_request_rate
        # Play the mission
        while status in (mission_pb2.State.STATUS_NONE, mission_pb2.State.STATUS_RUNNING):
            # Check for element-wise feedback
            mission_element_name, element_feedback = self._element_wise_feedback()
            if mission_element_name is not None and element_feedback:
                self._robot.logger.info('ArmSensorInspector: Completed ' + mission_element_name)
                print('ArmSensorInspector: Completed ' + mission_element_name)
                # Set inspection_action_end_time to now
                inspection_action_end_time = time.time()
                # Retrieve the images from the DAQ if mission execution is successful
                timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
                timestamp_mdy = datetime.now().strftime("%m%d%Y")

                self._download_image_from_robot(inspection_action_start_time,
                                                inspection_action_end_time,
                                                timestamp_mdy)
                # Reset inspection_action_start_time to now
                inspection_action_start_time = time.time()
            # Mission fails if any operator questions are persistent
            mission_questions = self._mission_client.get_state().questions
            if mission_questions:
                # Answer mission questions
                if not self._run_with_fallback(self._answer_mission_question, mission_questions):
                    # If answering the questions did not help, fail the mission
                    self._robot.logger.error(
                        'ArmSensorInspector: Mission failed by triggering operator question.')
                    return False
            # Send play request every play_request_rate
            if (time.time() > play_request_time):
                # Advance body lease
                body_lease = self._lease_client.lease_wallet.advance()
                # Send play request
                play_mission_response = self._mission_client.play_mission(
                    pause_time_secs=time.time() + mission_timeout, leases=[body_lease])
                # Return if play requested fails
                if not play_mission_response:
                    self._robot.logger.info("ArmSensorInspector: Mission Play Request Failed!")
                    print("ArmSensorInspector: Mission Play Request Failed!")
                    return False
                # Update the next play_request_time
                play_request_time = time.time() + play_request_rate
            print(f"[Debug] Mission Executing... {datetime.now()}")
            status = self._mission_client.get_state().status
        self._robot.logger.info("ArmSensorInspector: Mission Execution Successful? " +
                                str(status == mission_pb2.State.STATUS_SUCCESS))
        print("ArmSensorInspector: Mission Execution Successful? " +
                                str(status == mission_pb2.State.STATUS_SUCCESS))
        return status == mission_pb2.State.STATUS_SUCCESS

    def _answer_mission_question(self, questions):
        ''' A helper function that answers questions raised during mission execution on the robot.
            - Args:
                - questions(list of mission_pb2.State.Question): a list of mission questions obtained via self._mission_client.get_state().questions
            - Returns:
                - Boolean indicating if we have answered the mission question successful
        '''
        for question in questions:
            try:
                if (question.text.find("Robot is waiting") != -1):
                    # Respect the wait time. The mission playback_mode is periodic.
                    return True
                if (question.text.find(ACTION_NAME) != -1):
                    # Increment self._arm_pointing_failure_count
                    self._arm_pointing_failure_count += 1
                # Stow the arm to make sure that the arm is not deployed while robot is executing the next mission node
                self._stow_arm()
                self._robot.logger.error('ArmSensorInspector: ' + str(question.text))
                print('ArmSensorInspector: ' + str(question.text))
                answer_code = self._get_answer_code_to_mission_question(
                    question.options, MISSION_QUESTION_ANSWER_CHOICE)
                self._mission_client.answer_question(question.id, answer_code)
                self._robot.logger.info('ArmSensorInspector: Answered mission question  with ' +
                                        str(MISSION_QUESTION_ANSWER_CHOICE))
                print('ArmSensorInspector: Answered mission question  with ' +
                                        str(MISSION_QUESTION_ANSWER_CHOICE))

            except (bosdyn.client.RpcError, bosdyn.client.ResponseError) as err:
                # Log Exception
                text_message = "ArmSensorInspector: Exception raised: [{}] {} - file: {} at line ({}) Trying again!".format(
                    type(err), str(err), err.__traceback__.tb_frame.f_code.co_filename,
                    err.__traceback__.tb_lineno)
                self._robot.logger.error(text_message)
                self._robot.logger.error('ArmSensorInspector: Error while answering ' +
                                         str(question.text))
        # Check if mission questions are still persistent
        if self._mission_client.get_state().questions:
            self._robot.logger.error('ArmSensorInspector: Mission questions are still persistent!')
            print('ArmSensorInspector: Mission questions are still persistent!')
            return False
        return True

    def _get_answer_code_to_mission_question(self, options, answer_string):
        ''' A helper function that returns the answer code associated with a string. 
            - Args:
                answer_string(string): a string input used to query for answer code to a mission question
                options(list of nodes_pb2.Prompt.Option): answer options for mission questions 
                                                                    obtained via self._mission_client.get_state().questions[i].options
            - Returns:
                answer_code(int): a code associated with the string if string exists if not returns 2 which is "Try Again"
        '''
        answer_code = 2  # default as 2 which is "Try Again"
        for option in options:
            if (option.text.find(answer_string) != -1):
                # We have found the answer we want. Go ahead assign the answer_code associated
                answer_code = option.answer_code
        return answer_code

    def _element_wise_feedback(self):
        ''' A helper function that provide element-wise feedback during mission execution. 
            - Returns:
                - mission_element_name(string):  return the name of the mission element that is being executed
                - Boolean indicating if an element is successful
        '''

        # Return if there are no inspections
        if not self._node_map:
            return None, False
        try:
            # Get mission state
            mission_state = self._mission_client.get_state()
            if mission_state.history:
                # Look at the most recent mission_state.history which is always the first index
                for node_state in mission_state.history[0].node_states:
                    # Query for element name given the node_state.id
                    mission_element_name = self._node_map.get(node_state.id)
                    # Return the mission_element_name and result if node_state.id is in self._node_map
                    if mission_element_name is not None:
                        return mission_element_name, (node_state.result == util_pb2.RESULT_SUCCESS)
            # Did not find node_state.id is in self._node_map
            return None, False
        except Exception as err:
            # Log Exception
            text_message = "ArmSensorInspector: Exception raised: [{}] {} - file: {} at line ({})".format(
                type(err), str(err), err.__traceback__.tb_frame.f_code.co_filename,
                err.__traceback__.tb_lineno)
            self._robot.logger.error(text_message)
            return None, False

    def _generate_node_map(self, mission_elements, element_identifiers):
        ''' A helper function that generates a map that associates mission element's name to element identifiers
            to provide element-wise feedback for missions.
            - Args:
                - mission_elements(a list of autowalk_pb2.Element):  a list of mission elements that correspond to the mission loaded onto the robot
                - element_identifiers(autowalk_pb2.ElementIdentifier):  element_identifiers obtained from load_autowalk response
            - Returns:
                - node_map(dict): a dictionary with key:element_identifier.action_id.node_id and value: inspection_id
        '''
        # Check the number of elements is equivalent to the number of element_identifiers
        if len(mission_elements) != len(element_identifiers):
            return None
        # Generate a node map that assoicates mission element names with their correpsonding element_identifiers
        node_map = {}
        for i in range(len(element_identifiers)):
            element_identifier = element_identifiers[i]
            # Add the node id assocated with the element_identifier's action id
            if (len(element_identifier.action_id.user_data_id) > 0):
                node_map[element_identifier.action_id.node_id] = mission_elements[i].name
        return node_map

    def _run_with_fallback(self, cmd_func, input_to_cmd_func=None):
        ''' A helper function that runs and handles exceptions for an input function given its argument.
            - Args:
                - cmd_func(Function): a function name - the function should return a boolean to indicate success
                - input_to_cmd_func(Function Input): a single argument for cmd_func
            - Returns:
                - return_value(Function return) : the return value associated it with cmd_func
        '''
        return_value = None
        success = False
        attempt_number = 0
        num_retries = 4
        while attempt_number < num_retries and not success:
            attempt_number += 1
            try:
                if not input_to_cmd_func:
                    return_value = cmd_func()
                else:
                    return_value = cmd_func(input_to_cmd_func)
                success = True
            except Exception as err:
                # Log Exception
                text_message = "ArmSensorInspector: Exception raised running [{}] : [{}] {} - file: {} at line ({}) Trying again!".format(
                    str(cmd_func), type(err), str(err),
                    err.__traceback__.tb_frame.f_code.co_filename, err.__traceback__.tb_lineno)
                self._robot.logger.error(text_message)
                print(text_message)
            time.sleep(1) if not success else time.sleep(0)
        return return_value

    def _create_identical_mission_without_elements(self):
        ''' A helper function that creates a new mission that has the same fields as the loaded mission, but has no elements 
            - Returns:
                - curated_mission(walks_pb2.Walk): a new mission that has the same fields as the loaded mission excluding mission elements
        '''
        mission_name = self._base_mission.mission_name
        docks = self._base_mission.docks
        global_parameters = self._base_mission.global_parameters
        playback_mode = self._base_mission.playback_mode
        # create a new mission without elements
        curated_mission = Walk(mission_name=mission_name, global_parameters=global_parameters,
                               playback_mode=playback_mode, docks=docks)
        return curated_mission

    def _extract_inspection_id_from_mission_element_name(self, mission_element_name):
        ''' A helper function that returns the inspection_id given the mission element name
            - Args:
                - mission_element_name(string): the string linked with the mission_element_name
            - Returns:
                - inspection_id(int) corresponding to the mission_element_name
        '''
        # For now ASP is prinitng as 'Arm Pointing 1' so its easier to find the digit as follows
        return ''.join(filter(str.isdigit, mission_element_name))

    def _construct_mission_given_inspection_ids(self, inspection_ids):
        ''' A helper function that returns the mission consisting of elements associated with the
            inspection_ids. This function should fail if all elements requested are not present in the 
            mission. 
            - Args:
                - inspection_ids(list of int): a list of ints that indicate an inspection point ID number.
            - Returns:
                - curated_mission(walks_pb2.Walk): a mission that has only the inspection_elements 
                                                    corresponding to the inspection_ids
        '''
        # Create an mission without elements
        curated_mission = self._create_identical_mission_without_elements()

        # Fill with elements that correspond to the inspection ids.
        chosen_elements = self._select_inspection_elements(inspection_ids)
        # Check if chosen_elements is None
        if not chosen_elements:
            self._robot.logger.info(
                'ArmSensorInspector: All elements requested are not present in the mission.')
            return
        # Given the chosen elements append each to curated_mission.elements
        for chosen_element in chosen_elements:
            curated_mission.elements.append(chosen_element)
        return curated_mission

    def _get_base_inspections(self):
        ''' A helper function that returns a dictionary of inspection mission elements with 
            their respective inspection_id and a list of inspection IDs.
            - Returns:
                - arm_sensor_pointing_elements(dict): a dict  with key - inspection_id and value - Autowalk Element
                - inspection_ids(list): a list of ids associated with each inspection mission element
         '''
        arm_sensor_pointing_elements = {}
        inspection_ids = []
        for mission_element in self._base_mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                # We have found the action name we are interested in
                # So go ahead and parse the line to extract waypoints
                inspection_id = self._extract_inspection_id_from_mission_element_name(
                    mission_element.name)
                # Add the mission element to the arm_sensor_pointing_elements dict
                arm_sensor_pointing_elements[int(inspection_id)] = mission_element
                # Add the inspection_id to inspection_ids list
                inspection_ids.append(inspection_id)
        return arm_sensor_pointing_elements, inspection_ids

    def _select_inspection_elements(self, inspection_ids):
        ''' A helper function that returns the mission elements associated with the
            inspection_ids. This function should fail if all elements requested are not present in the
            mission.
            - Args:
                - inspection_ids(list of int): a list of ints that indicate an inspection point ID number.
            - Returns:
                - inspection_elements(list of AutoWalk Element): a list of AutoWalk Elements corresponding to the inspection_ids
         '''
        # Using the self._base_inspection_elements find the mission elements corresponding to inspection_ids
        inspection_elements = []
        for inspection_id in inspection_ids:
            inspection_element = self._base_inspection_elements.get(int(inspection_id))
            # Quit if inspection_id is invalid
            if not inspection_element:
                self._robot.logger.info((
                    'ArmSensorInspector: Invalid inspection_id: {}! It is not in the list of inspection_ids : {}! '
                ).format(inspection_id, self._base_inspection_ids))
                return
            # Append if inspection_element is not None
            inspection_elements.append(inspection_element)
        return inspection_elements

    def _clear_graph(self):
        ''' A helper function that clears the state of the map on the robot, removing all waypoints and edges.
            - Returns:
                - Boolean indicating if the given map is cleared successfuly or not.
        '''
        self._robot.logger.info("ArmSensorInspector: Cleared the pre-existing map on the robot")
        return self._graph_nav_client.clear_graph()

    def _upload_map_and_localize(self):
        ''' A helper function that uploads the graph and snapshots to the robot and localizes the robot to the map.
           - Returns:
                - Boolean indicating if the given map is uploaded and localized to the robot successfuly or not.
        '''
        # Clear the preexisitng map on the robot
        self._clear_graph()
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot

        # Upload the graph to the robot.
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Return if status is not okay
        if response.status != graph_nav_pb2.UploadGraphResponse.STATUS_OK:
            self._robot.logger.info("ArmSensorInspector: Problem uploading graph to robot!")
            return False

        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
        # Localize the robot to the map
        return self._ensure_robot_is_localized()

    def _download_image_from_robot(self, start_time, end_time, time_s, additional_params=None):
        ''' A helper function to retrieve all images from the DataBuffer REST API and write them to files in local disk.
            - Args:
                - start_time(timestamp): the timestamp of the desired start time needed to download images
                - end_time(timestamp): the timestamp of the desired end time needed to download images 
                - additional_params(dict): Additional GET parameters to append to the URL.
            - Returns:
                - Boolean indicating if the data was downloaded successfuly or not.
        '''
        # Query parameters to use to retrieve metadata from the DataStore service.
        query_params = make_time_query_params(start_time, end_time, self._robot)
        # Hostname to specify in URL where the DataBuffer service is running.
        hostname = self._robot.address
        # User token to specify in https GET request for authentication.
        token = self._robot.user_token
        # Folder where to download the data.
        folder = self._inspection_folder
        # String to differentiate the captured images
        image_suffix = self._image_suffix

        try:
            url = 'https://{}/v1/data-buffer/daq-data/'.format(hostname)
            headers = {"Authorization": "Bearer {}".format(token)}
            get_params = additional_params or {}
            if query_params.HasField('time_range'):
                get_params.update({
                    'from_nsec': query_params.time_range.from_timestamp.ToNanoseconds(),
                    'to_nsec': query_params.time_range.to_timestamp.ToNanoseconds()
                })
            chunk_size = 10 * (1024**2)  # This value is not guaranteed.

            url = url + '?{}'.format(urlencode(get_params))
            request = Request(url, headers=headers)
            context = ssl._create_unverified_context()
            with urlopen(request, context=context) as resp:
                # This is the default file name used to download data, updated from response.
                if resp.status == 204:
                    self._robot.logger.info("ArmSensorInspector: " + str(
                        "No content available for the specified download time range (in seconds): "
                        "[%d, %d]" %
                        (query_params.time_range.from_timestamp.ToNanoseconds() / 1.0e9,
                         query_params.time_range.to_timestamp.ToNanoseconds() / 1.0e9)))
                    print("ArmSensorInspector: " + str(
                        "No content available for the specified download time range (in seconds): "
                        "[%d, %d]" %
                        (query_params.time_range.from_timestamp.ToNanoseconds() / 1.0e9,
                         query_params.time_range.to_timestamp.ToNanoseconds() / 1.0e9)))

                    return False

                # Set path for image_dir
                image_dir = folder + 'images/' + str(image_suffix)
                # Check whether the  specified path is an existing file
                if not os.path.isdir(image_dir):
                    os.makedirs(image_dir, exist_ok=True)
                zip_folder_name = "download_" + datetime.now().strftime("%m%d%Y_%H%M%S") + ".zip"
                download_file = Path(image_dir, zip_folder_name)
                content = resp.headers['Content-Disposition']
                if not content or len(content) < 2:
                    self._robot.logger.error(
                        "ArmSensorInspector: Content-Disposition is not set correctly")
                    print(
                        "ArmSensorInspector: Content-Disposition is not set correctly")
                    return False
                else:
                    start_ind = content.find('\"')
                    if start_ind == -1:
                        self._robot.logger.error(
                            "ArmSensorInspector: Content-Disposition does not have a \"")
                        print(
                            "ArmSensorInspector: Content-Disposition does not have a \"")
                        return False
                    else:
                        start_ind += 1

                with open(str(download_file), 'wb') as fid:
                    while True:
                        chunk = resp.read(chunk_size)
                        if len(chunk) == 0:
                            break
                        print('.', end='', flush=True)
                        fid.write(chunk)
        except URLError as rest_error:
            self._robot.logger.error("ArmSensorInspector: REST Exception:" + str(rest_error))
            print("ArmSensorInspector: REST Exception:" + str(rest_error))
            return False
        except IOError as io_error:
            self._robot.logger.error("ArmSensorInspector: IO Exception:" + str(io_error))
            print("ArmSensorInspector: IO Exception:" + str(io_error))
            return False
        # Data downloaded and saved to local disc successfuly in a zip.
        count = 0
        with zipfile.ZipFile(download_file, 'r') as zip_folder:
            files = zip_folder.namelist()
            # Extract the image file to its corresponding folder
            for file_name in files:
                if file_name.endswith('.jpg'):
                    zip_folder.extract(file_name, image_dir)
                    # Split the file_name by "/" and query the image name and add a timestamp
                    img_name = file_name.split("/")[1]
                    # Extract the image name without the jpg extension
                    img_name_without_jpg_ext = img_name.split(".")[0]
                    # Extract the timestamps based on the state of image_suffix
                    if image_suffix == '':
                        # timestamp = ''.join(filter(str.isdigit, folder))
                        timestamp = time_s
                    else:
                        # This image is most likely part of a periodic inspection
                        # Hence, use image_suffix which contains the cycle number
                        timestamp = ''.join(filter(str.isdigit, image_suffix))
                    # Apppend the timestamp to create new_img_name
                    new_img_name = img_name_without_jpg_ext + "_" + str(timestamp) + ".jpg"
                    # Rename filename such that the image moves up a directory
                    shutil.move(image_dir + file_name, image_dir + new_img_name)
                    # os.rename(image_dir + file_name, image_dir + new_img_name)
                    # Remove the folder that the image came in from the robot
                    os.rmdir(image_dir + file_name.split("/")[0])
                    count += 1
        # Update the number of inspection points completed
        self._inspection_elements_completed = count
        # Remove the zip folder after extraction
        os.remove(download_file)
        return True

    def _battery_status(self, silent_print=False):
        ''' A helper function that provides the status of the robot battery. 
            - Args:
                - silent_print(boolean): a boolean to silence printing the battery status
            - Returns:
                - battery_level(double): the current battery level percentage 
                - min_battery_temp(double): the current min temperature for the battery
                - max_battery_temp(double): the current max temperature for the battery
        '''
        robot_state = self._robot_state_client.get_robot_state()
        battery_states = robot_state.battery_states
        # Check if battery_states is available
        if not battery_states:
            self._robot.logger.error("ArmSensorInspector: Problem in querying for battery_states!")
            # -1 indicates an error
            battery_level = min_battery_temp = max_battery_temp = -1
            return battery_level, min_battery_temp, max_battery_temp
        # Take a look at the most recent battery_state
        battery_state = battery_states[0]
        # Query the battery status
        status = battery_state.Status.Name(battery_state.status)
        # Get rid of the STATUS_ prefix
        status = status[7:]
        battery_temperatures = battery_state.temperatures
        # Check if battery_temperatures is available
        if battery_temperatures:
            min_battery_temp = min(battery_temperatures)
            max_battery_temp = max(battery_temperatures)
        else:
            # -1 indicates an invalid temp
            min_battery_temp = max_battery_temp = -1
        battery_level = battery_state.charge_percentage.value
        # Check if battery level is available
        if battery_level:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = '|{}{}|'.format('=' * bar_len, ' ' * (10 - bar_len))
        else:
            # -1 indicates an invalid battery level
            battery_level = -1
            bat_bar = ''
        time_left = ''
        # Check if battery_state.estimated_runtime is available
        if battery_state.estimated_runtime:
            time_left = ' ({})'.format(secs_to_hms(battery_state.estimated_runtime.seconds))
        # Print the battery status if print is not silenced
        if not silent_print:
            print_1 = (' Battery: {}{}{}{}'.format(status, bat_bar, battery_level, time_left))
            print_2 = (" Min Battery Temp: {} Max Battery Temp: {}".format(
                min_battery_temp, max_battery_temp))
            print_3 = " Battery voltage: " + str(battery_state.voltage.value)
            text_message = "ArmSensorInspector: " + print_1 + print_2 + print_3
            self._robot.logger.info(text_message)
        return battery_level, min_battery_temp, max_battery_temp

    def _write_inspection_data(self, inspection_data_path, file_name, data):
        ''' A helper function that writes inspection data to a csv file.
            - Args:
                - inspection_data_path(string): the path to save the csv
                - file_name(string): the name of the csv file 
                - data(list): a list of values to be printed to the csv
        '''
        # If folder path is not given use the current working dir
        if inspection_data_path is None:
            inspection_data_path = os.getcwd() + '/'
        # If file_name is none
        if file_name is None:
            return
        # Check whether the  specified path is an existing file
        if not os.path.isdir(inspection_data_path):
            os.makedirs(inspection_data_path, exist_ok=True)
        file_name.touch(exist_ok=True)
        with open(file_name, 'a+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def _write_periodic_mission_summary(self, csv_filepath, number_of_cycles,
                                        periodic_mission_start_datetime,
                                        periodic_mission_end_datetime):
        ''' A helper function that computes metrics and provides a summary for a  periodic_mission. 
            - Args:
                - csv_filepath(string): the file path to the mission data stored as a csv file
                - number_of_cycles(int) : the number of cycles used to run the periodic_mission 
                - periodic_mission_start_datetime(datetime): a datetime object created when the periodic_mission started
                - periodic_mission_end_datetime(datetime): a datetime object created when the periodic_mission ended
            - Returns:
                - Boolean indicating if writing periodic_mission_summary is successful
        '''
        try:
            # Assign csv_header to the data header for inspection data saved in csv
            csv_header = self._inspection_data_header
            # Read the data saved in the csv_filepath
            inspection_data = pd.read_csv(csv_filepath)
            # Periodic_mission Stats
            periodic_mission_duration = (
                periodic_mission_end_datetime -
                periodic_mission_start_datetime).total_seconds() / 3600  #hrs
            # Cycle stats. Note that cycles_completed_index should refer to 'Mission Succeeded?'
            cycles_completed_index = csv_header.index("Mission Succeeded?")
            cycles_required = number_of_cycles
            # Initialize cycles_completed
            cycles_completed = 0
            # Compute the frequency of strings linked with cycles_completed_index
            value_counts = inspection_data[csv_header[cycles_completed_index]].value_counts()
            # Only change the cycles_completed if the "Mission Succeeded?" column has the field "True"
            if value_counts.get(True):
                cycles_completed = value_counts[True]
            cycles_failed = cycles_required - cycles_completed
            # Inspection stats.
            # Note that inspections_required_index '# of required inspections'
            inspections_required_index = csv_header.index("# of required inspections")
            inspections_required = inspection_data[csv_header[inspections_required_index]].sum()
            # Note that inspections_completed_index should refer to '# of completed inspections'
            inspections_completed_index = csv_header.index("# of completed inspections")
            inspections_completed = inspection_data[csv_header[inspections_completed_index]].sum()
            # Note that inspections_failed_index should refer to '# of failed inspections'
            inspections_failed_index = csv_header.index("# of failed inspections")
            inspections_failed = inspection_data[csv_header[inspections_failed_index]].sum()
            # Arm pointing failure stats. Note that inspections_completed_index should refer to '# of arm pointing failures'
            inspections_failure_index = csv_header.index("# of arm pointing failures")
            arm_pointing_failures = inspection_data[csv_header[inspections_failure_index]].sum()
            # Cycle time stats. Note that csv_header[3] should refer to 'Cycle Time in min'
            cycle_time_index = csv_header.index("Cycle Time in min")
            cycle_times = inspection_data[csv_header[cycle_time_index]]
            average_cycle_time = cycle_times.mean()
            median_cycle_time = cycle_times.median()
            std_cycle_time = cycle_times.std()
            cycle_time_Q1 = cycle_times.quantile(0.25)
            cycle_time_Q3 = cycle_times.quantile(0.75)
            min_cycle_time = cycle_times.min()
            max_cycle_time = cycle_times.max()
            # Battery stats. Note that battery_index should refer to 'Battery Consumption'
            battery_index = csv_header.index("Battery Consumption")
            battery_data = inspection_data[csv_header[battery_index]]
            battery_mean = abs(battery_data).mean()
            battery_median = abs(battery_data).median()
            battery_std = abs(battery_data).std()
            battery_Q1 = abs(battery_data).quantile(0.25)
            battery_Q3 = abs(battery_data).quantile(0.75)
            battery_min = abs(battery_data).min()
            battery_max = abs(battery_data).max()
            # Prepare summary data for csv
            summary_data = [
                periodic_mission_start_datetime, periodic_mission_end_datetime,
                periodic_mission_duration, cycles_required, cycles_completed, cycles_failed,
                inspections_required, inspections_completed, inspections_failed,
                arm_pointing_failures, average_cycle_time, median_cycle_time, std_cycle_time,
                cycle_time_Q1, cycle_time_Q3, min_cycle_time, max_cycle_time, battery_mean,
                battery_median, battery_std, battery_Q1, battery_Q3, battery_min, battery_max
            ]
            self._write_inspection_data(self._inspection_folder, csv_filepath, self._summary_header)
            # Write the summary data to the csv
            self._write_inspection_data(self._inspection_folder, csv_filepath, summary_data)
            # Reset variables associated with inspection folder
            self._inspection_folder = os.getcwd() + '/'
            self._image_suffix = ''
            return True
        except Exception as err:
            # Log Exception
            text_message = "ArmSensorInspector: Problem in writing in _periodic_mission_summary! Exception raised: [{}] {} - file: {} at line ({})".format(
                type(err), str(err), err.__traceback__.tb_frame.f_code.co_filename,
                err.__traceback__.tb_lineno)
            self._robot.logger.error(text_message)
            return False

    def _get_global_parameters(self, mission):
        ''' A function that returns the global_parameters used for inspection missions.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
            - Returns:
                - The field named "global_parameters" in the given mission
        '''
        return mission.global_parameters

    def _get_mission_name(self, mission):
        ''' A function that returns the mission_name.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
            - Returns:
                - The field named "mission_name" in the given mission
        '''
        return mission.mission_name

    def _get_playback_mode(self, mission):
        ''' A function that returns the playback_mode for the mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
            - Returns:
                - The field named "playback_mode" in the given mission
        '''
        return mission.playback_mode

    def _get_num_of_inspection_elements(self, mission):
        ''' A function that returns the number of mission elements associated with the
            with ACTION_NAME = "Arm Pointing" in the given mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
            - Returns:
                - num_of_inspection_elements(int): number of inspection mission elements
         '''
        # Initialize count
        num_of_inspection_elements = 0
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                # we have found action name we are interested in
                # so go ahead increment the num_of_inspection_elements
                num_of_inspection_elements += 1
        return num_of_inspection_elements

    def _print_mission_info(self, mission):
        ''' A function that prints relevant info based on the provided mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        self._robot.logger.info((
            'ArmSensorInspector: Mission Name : {} Number of Inspection Actions: {} Playback Mode: {} Global Params: {}'
        ).format(self._get_mission_name(mission), self._get_num_of_inspection_elements(mission),
                 self._get_playback_mode(mission), self._get_global_parameters(mission)))

        print((
            'ArmSensorInspector: Mission Name : {} Number of Inspection Actions: {} Playback Mode: {} Global Params: {}'
        ).format(self._get_mission_name(mission), self._get_num_of_inspection_elements(mission),
                 self._get_playback_mode(mission), self._get_global_parameters(mission)))

    def _set_mission_name(self, mission, mission_name):
        ''' A function that sets mission_name for a given mission.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - mission_name(string): the desired mission_name
        '''
        mission.mission_name = mission_name
        self._robot.logger.info(
            ("ArmSensorInspector: Completed setting mission name to {} !").format(mission_name))
        print(f"ArmSensorInspector: Completed setting mission name to {mission_name} !")

    def _set_global_parameters(self, mission, self_right_attempts=0):
        ''' A function that sets global_parameters.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - self_right_attempts(int): attempts to automatically self-rights the robot 
                                          if robot experiences a fall.
        '''
        mission.global_parameters.self_right_attempts = self_right_attempts
        self._robot.logger.info((
            "ArmSensorInspector: Completed setting mission global parameters! self_right_attempts = {}"
        ).format(self_right_attempts))

        print((
            "ArmSensorInspector: Completed setting mission global parameters! self_right_attempts = {}"
        ).format(self_right_attempts))

    def _set_failure_behavior(self, mission, behavior="proceed_if_able", retry_count=1,
                              prompt_duration_secs=10, try_again_delay_secs=60):
        ''' A helper function that sets failure behaviors for the robot to handle
            failures during a mission execution. 
            - Some of the possible failures that could happen are the following. 
                - System Faults:   indicates a hardware or software fault on the robot.
                - Behavior Faults: faults related to behavior commands and issue warnings 
                                    if a certain behavior fault will prevent execution of subsequent commands.
                - Service Faults:  third party payloads and services may encounter unexpected issues
                                on hardware or software connected to Spot.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - behavior(string): the desired failure behavior for inspection actions.
                    - "safe_power_off": the robot will sit down and power off. This is the safest option.
                    - "proceed_if_able": the robot will proceed to the next action if able to do so.
                    - "return_to_start_and_terminate": the robot will return to the start, dock and terminate the mission if able to do so. Only available in missions with a dock!
                    - "return_to_start_and_try_again_later": the robot will return to the start and dock. If successful, the robot will try again later after try_again_delay_secs.
                - retry_count(int): the number of times the robot should try running the mission element
                - prompt_duration_secs(seconds - min 10s): the duration of the prompt for user 
                                                           before defaulting to failure behaviors.
                - try_again_delay_secs(seconds- min 60s): the wait time before trying again.                      
        '''
        behavior_choices = [
            "safe_power_off", "proceed_if_able", "return_to_start_and_terminate",
            "return_to_start_and_try_again_later"
        ]

        if behavior not in behavior_choices:
            text = " Choose [safe_power_off, proceed_if_able, return_to_start_and_terminate, return_to_start_and_try_again_later] "
            self._robot.logger.error(
                "ArmSensorInspector: " + str(behavior) + " is an invalid input!" + text +
                "Continuing with default failure behaviors set during map recording!")
            return
        # Set prompt_duration and try_again_delay in seconds
        prompt_duration = duration_pb2.Duration(seconds=prompt_duration_secs)
        try_again_delay = duration_pb2.Duration(seconds=try_again_delay_secs)
        # Set retry count and prompt duration for the failure_behavior
        failure_behavior = FailureBehavior(retry_count=retry_count, prompt_duration=prompt_duration)
        # Determine the failure behavior requested
        if behavior == "safe_power_off":
            failure_behavior.safe_power_off.SetInParent()
        elif behavior == "proceed_if_able":
            failure_behavior.proceed_if_able.SetInParent()
        elif behavior == "return_to_start_and_terminate":
            failure_behavior.return_to_start_and_terminate.SetInParent()
        elif behavior == "return_to_start_and_try_again_later":
            failure_behavior.return_to_start_and_try_again_later.try_again_delay.CopyFrom(
                try_again_delay)
        # Set this failure_behavior to mission_elements in the given mission
        for mission_element in mission.elements:
            # Action is what the robot should do at that location. Here, we are setting its failure behavior.
            mission_element.action_failure_behavior.CopyFrom(failure_behavior)
            # Target is the location the robot should navigate to. Here, we are setting its failure behavior.
            mission_element.target_failure_behavior.CopyFrom(failure_behavior)
        self._robot.logger.info("ArmSensorInspector: Completed setting failure behavior to " +
                                behavior + " !")
        print("ArmSensorInspector: Completed setting failure behavior to " + behavior + " !")

    def _set_gripper_camera_parameters(self, mission, inspection_id_input=None, resolution=None,
                                       brightness=None, contrast=None, gain=None, saturation=None,
                                       manual_focus=None, auto_focus=None, exposure=None,
                                       auto_exposure=None, hdr_mode=None, led_mode=None,
                                       led_torch_brightness=None):
        ''' A helper function that sets the gripper camera parameters using the provided settings.
            If inspection_id_input is provided, this function changes the gripper camera params for
            the inspection action that corresponds with the inspection_id_input. If not provided, the
            function applies the given camera params to all inspection actions.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - inspection_id_input(int): the desired inspection_id
                - resolution(string): resolution of the camera
                    - choices: '640x480', '1280x720','1920x1080', '3840x2160', '4096x2160', '4208x3120'
                - brightness(double): brightness value in (0.0 - 1.0)
                - contrast(double):   contrast value in (0.0 - 1.0)
                - gain(double):       gain value in (0.0 - 1.0)
                - saturation(double): saturation value in (0.0 - 1.0)
                - manual_focus(double): manual focus value in (0.0 - 1.0)
                - auto_focus(string): Enable/disable auto-focus
                    - choices: 'on', 'off'
                - exposure(double):   exposure value in (0.0 - 1.0)
                - auto_exposure(string): enable/disable auto-exposure
                    - choices: 'on', 'off'
                - hdr_mode(string):   on-camera high dynamic range (HDR) setting.  manual1-4 modes enable HDR with 1 the minimum HDR setting and 4 the maximum'
                    - choices: 'off','auto','manual1','manual2','manual3','manual4'   
                - led_mode(string):   LED mode. 
                    - choices:
                        - 'off': off all the time.
                        - 'torch': on all the time.
                        - 'flash': on during snapshots.
                        - 'both': on all the time and at a different brightness for snapshots.
                - led_torch_brightness(double): LED brightness value in (0.0 - 1.0) when led_mode is on all the time
            '''
        # Find the mission element that corresponds to the given inspection id
        mission_element_for_inspection_id_input = None
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                inspection_id = self._extract_inspection_id_from_mission_element_name(
                    mission_element.name)
                # Check if inspection_id_input is not provided
                if inspection_id_input is None:
                    # Change the gripper camera params for all mission elements related with arm sensor pointing
                    self._set_gripper_camera_parameters_for_mission_element(
                        mission_element, resolution, brightness, contrast, gain, saturation,
                        manual_focus, auto_focus, exposure, auto_exposure, hdr_mode, led_mode,
                        led_torch_brightness)
                # Found mission element associated with inspection_id_input
                elif int(inspection_id_input) == int(inspection_id):
                    # Now set the cam params for this element
                    mission_element_for_inspection_id_input = mission_element
                    self._set_gripper_camera_parameters_for_mission_element(
                        mission_element_for_inspection_id_input, resolution, brightness, contrast,
                        gain, saturation, manual_focus, auto_focus, exposure, auto_exposure,
                        hdr_mode, led_mode, led_torch_brightness)
        # Return with an error if we did not find the corresponding mission_element for inspection_id_input
        if not mission_element_for_inspection_id_input:
            self._robot.logger.info((
                'ArmSensorInspector: Invalid inspection_id: {}! It is not in the list of inspection_ids! '
            ).format(inspection_id_input))
            return

    def _set_gripper_camera_parameters_for_mission_element(
            self, mission_element, resolution=None, brightness=None, contrast=None, gain=None,
            saturation=None, manual_focus=None, auto_focus=None, exposure=None, auto_exposure=None,
            hdr_mode=None, led_mode=None, led_torch_brightness=None):
        ''' A helper function that sets the gripper camera parameters using the provided settings 
            a given mission element.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                mission_element(walks_pb2.Element): the desired mission_element
                - resolution(string): resolution of the camera
                    - choices: '640x480', '1280x720','1920x1080', '3840x2160', '4096x2160', '4208x3120'
                - brightness(double): brightness value in (0.0 - 1.0)
                - contrast(double):   contrast value in (0.0 - 1.0)
                - gain(double):       gain value in (0.0 - 1.0)
                - saturation(double): saturation value in (0.0 - 1.0)
                - manual_focus(double): manual focus value in (0.0 - 1.0)
                - auto_focus(string): Enable/disable auto-focus
                    - choices: 'on', 'off'
                - exposure(double):   exposure value in (0.0 - 1.0)
                - auto_exposure(string): enable/disable auto-exposure
                    - choices: 'on', 'off'
                - hdr_mode(string):   on-camera high dynamic range (HDR) setting.  manual1-4 modes enable HDR with 1 the minimum HDR setting and 4 the maximum'
                    - choices: 'off','auto','manual1','manual2','manual3','manual4'   
                - led_mode(string):   LED mode. 
                    - choices:
                        - 'off': off all the time.
                        - 'torch': on all the time.
                        - 'flash': on during snapshots.
                        - 'both': on all the time and at a different brightness for snapshots.
                - led_torch_brightness(double): LED brightness value in (0.0 - 1.0) when led_mode is on all the time
            '''
        # For this specific inspection ID set the following GripperCameraParams if they are provided
        # Set the resolution
        if resolution is not None:
            if resolution in ('640x480', '1280x720', '1920x1080', '3840x2160', '4096x2160',
                              '4208x3120'):
                if resolution == '640x480':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480_120FPS_UYVY
                elif resolution == '1280x720':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1280_720_60FPS_UYVY
                elif resolution == '1920x1080':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1920_1080_60FPS_MJPG
                elif resolution == '3840x2160':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_3840_2160_30FPS_MJPG
                elif resolution == '4096x2160':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4096_2160_30FPS_MJPG
                elif resolution == '4208x3120':
                    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4208_3120_20FPS_MJPG
                mission_element.action_wrapper.gripper_camera_params.params.camera_mode = camera_mode
            else:
                text = " Choose ['640x480', '1280x720', '1920x1080', '3840x2160', '4096x2160','4208x3120'] "
                self._robot.logger.error(
                    "ArmSensorInspector: " + str(led_mode) + " is an invalid input!" + text +
                    "Continuing with default led mode set during map recording!")
        # Set the brightness
        if brightness is not None:
            mission_element.action_wrapper.gripper_camera_params.params.brightness.value = brightness
        # Set the contrast
        if contrast is not None:
            mission_element.action_wrapper.gripper_camera_params.params.contrast.value = contrast
        # Set the saturation
        if saturation is not None:
            mission_element.action_wrapper.gripper_camera_params.params.saturation.value = saturation
        # Set the gain
        if gain is not None:
            mission_element.action_wrapper.gripper_camera_params.params.gain.value = gain
        #  Check manual_focus and auto_focus restrictions
        if manual_focus is not None and auto_focus and auto_focus == 'on':
            self._robot.logger.warn(
                'ArmSensorInspector: Can not specify both a manual focus value and enable auto-focus. Setting auto_focus = off now'
            )
            # Set  auto_focus off
            mission_element.action_wrapper.gripper_camera_params.params.focus_auto.value = False
        # Set the manual_focus
        if manual_focus:
            mission_element.action_wrapper.gripper_camera_params.params.focus_absolute.value = manual_focus
            # Set  auto_focus off because we can not specify both a manual focus value and enable auto-focus
            mission_element.action_wrapper.gripper_camera_params.params.focus_auto.value = False
        # Set  auto_focus
        if auto_focus is not None:
            if auto_focus in ('on', 'off'):
                auto_focus_enabled = (auto_focus == 'on')
                mission_element.action_wrapper.gripper_camera_params.params.focus_auto.value = auto_focus_enabled
            else:
                text = " Choose ['on','off'] "
                self._robot.logger.error(
                    "ArmSensorInspector: " + str(auto_focus) + " is an invalid input!" + text +
                    "Continuing with default auto focus set during map recording!")
        if exposure is not None and auto_exposure and auto_exposure == 'on':
            self._robot.logger.warn(
                'ArmSensorInspector: Can not specify both manual exposure &enable auto-exposure. Setting auto_exposure = off now'
            )
            # Set  auto_exposure off
            mission_element.action_wrapper.gripper_camera_params.params.exposure_auto.value = False
        if exposure is not None:
            # Set the exposure
            mission_element.action_wrapper.gripper_camera_params.params.exposure_absolute.value = exposure
            # Set auto_exposure off because we can not specify both manual exposure &enable auto-exposure
            mission_element.action_wrapper.gripper_camera_params.params.exposure_auto.value = False
        # Set  auto_exposure
        if auto_exposure:
            if auto_exposure in ('on', 'off'):
                auto_exposure_enabled = (auto_exposure == 'on')
                mission_element.action_wrapper.gripper_camera_params.params.exposure_auto.value = auto_exposure_enabled
            else:
                text = " Choose ['on','off'] "
                self._robot.logger.error(
                    "ArmSensorInspector: " + str(auto_exposure) + " is an invalid input!" + text +
                    "Continuing with default auto exposure set during map recording!")
        # Set the hdr mode
        if hdr_mode is not None:
            if hdr_mode in ('off', 'auto', 'manual1', 'manual2', 'manual3', 'manual4'):
                if hdr_mode == 'off':
                    hdr = gripper_camera_param_pb2.HDR_OFF
                elif hdr_mode == 'auto':
                    hdr = gripper_camera_param_pb2.HDR_AUTO
                elif hdr_mode == 'manual1':
                    hdr = gripper_camera_param_pb2.HDR_MANUAL_1
                elif hdr_mode == 'manual2':
                    hdr = gripper_camera_param_pb2.HDR_MANUAL_2
                elif hdr_mode == 'manual3':
                    hdr = gripper_camera_param_pb2.HDR_MANUAL_3
                elif hdr_mode == 'manual4':
                    hdr = gripper_camera_param_pb2.HDR_MANUAL_4
                mission_element.action_wrapper.gripper_camera_params.params.hdr = hdr
            else:
                text = " Choose ['off', 'auto', 'manual1', 'manual2', 'manual3', 'manual4'] "
                self._robot.logger.error(
                    "ArmSensorInspector: " + str(hdr_mode) + " is an invalid input!" + text +
                    "Continuing with default hdr mode set during map recording!")
        # Set the led_mode
        if led_mode is not None:
            if led_mode in ('off', 'torch', 'flash', 'both'):
                if led_mode == 'off':
                    led = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_OFF
                elif led_mode == 'torch':
                    led = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_TORCH
                elif led_mode == 'flash':
                    led = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_FLASH
                elif led_mode == 'both':
                    led = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_FLASH_AND_TORCH
                mission_element.action_wrapper.gripper_camera_params.params.led_mode = led
            else:
                text = " Choose ['off', 'torch', 'flash', 'both'] "
                self._robot.logger.error(
                    "ArmSensorInspector: " + str(led_mode) + " is an invalid input!" + text +
                    "Continuing with default led mode set during map recording!")
        # Set the led_torch_brightness
        if led_torch_brightness is not None:
            mission_element.action_wrapper.gripper_camera_params.params.led_torch_brightness.value = led_torch_brightness
        # Log status
        self._robot.logger.info("ArmSensorInspector: Completed setting gripper camera parameters!")

    def _set_travel_speed(self, mission, travel_speed="MEDIUM"):
        ''' A helper function that sets travel parameters for navigation.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - travel_speed(string): the speed used by the robot to navigate the map.
                    - The base speeds are:
                        - robot_velocity_max_yaw = 1.13446 rad/s
                        - robot_velocity_max_x = 1.6  m/s
                        - robot_velocity_max_y = 0.5  m/s
                    - Choices: 
                        - "FAST":  100% base speed 
                        - "MEDIUM":66% base speed 
                        - "SLOW":  33% base speed
        '''
        # Base robot travel speeds
        robot_velocity_max_yaw = 1.13446  # rad/s
        robot_velocity_max_x = 1.6  # m/s
        robot_velocity_max_y = 0.5  # m/s
        # Velocity limits for navigation (optional)
        if travel_speed == "FAST":
            nav_velocity_max_yaw = robot_velocity_max_yaw
            nav_velocity_max_x = robot_velocity_max_x
            nav_velocity_max_y = robot_velocity_max_y
            self._robot.logger.info("ArmSensorInspector: Travel speed is set to FAST!")
            print("ArmSensorInspector: Travel speed is set to FAST!")
        elif travel_speed == "MEDIUM":
            nav_velocity_max_yaw = 0.66 * robot_velocity_max_yaw
            nav_velocity_max_x = 0.66 * robot_velocity_max_x
            nav_velocity_max_y = 0.66 * robot_velocity_max_y
            self._robot.logger.info("ArmSensorInspector: Travel speed is set to MEDIUM!")
            print("ArmSensorInspector: Travel speed is set to MEDIUM!")
        elif travel_speed == "SLOW":
            nav_velocity_max_yaw = 0.33 * robot_velocity_max_yaw
            nav_velocity_max_x = 0.33 * robot_velocity_max_x
            nav_velocity_max_y = 0.33 * robot_velocity_max_y
            self._robot.logger.info("ArmSensorInspector: Travel speed is set to SLOW!")
            print("ArmSensorInspector: Travel speed is set to SLOW!")
        else:
            self._robot.logger.error(
                "ArmSensorInspector: " + str(travel_speed) +
                " is an invalid input! Choose ['FAST', 'MEDIUM', 'SLOW'] Continuing with default travel speed set during map recording!"
            )
            return
        nav_velocity_limits = geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=nav_velocity_max_x, y=nav_velocity_max_y),
                angular=nav_velocity_max_yaw), min_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=-nav_velocity_max_x, y=-nav_velocity_max_y),
                    angular=-nav_velocity_max_yaw))
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                mission_element.target.navigate_route.travel_params.velocity_limit.CopyFrom(
                    nav_velocity_limits)

    def _set_joint_move_speed(self, mission, joint_move_speed="MEDIUM"):
        ''' A helper function that sets the speed for the arm joint move.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - joint_move_speed(string): the speed used by the robot to deploy the arm via a joint move
                    - The base settings are:
                        - maximum_velocity = 4.0 rad/s
                        - maximum_acceleration = 50.0 rad/s^2
                    - Choices: 
                        - "FAST":  100% base settings 
                        - "MEDIUM":66% base settings 
                        - "SLOW":  33% base settings
            '''
        # Velocity limits for arm movements
        if joint_move_speed == "FAST":
            maximum_velocity = 4.0  # rad/s
            maximum_acceleration = 50.0  #rad/s^2
            self._robot.logger.info("ArmSensorInspector: Joint move speed is set to FAST!")
            print("ArmSensorInspector: Joint move speed is set to FAST!")
        elif joint_move_speed == "MEDIUM":
            maximum_velocity = 2.64  # rad/s
            maximum_acceleration = 33.0  #rad/s^2
            self._robot.logger.info("ArmSensorInspector: Joint move speed is set to MEDIUM!")
            print("ArmSensorInspector: Joint move speed is set to MEDIUM!")
        elif joint_move_speed == "SLOW":
            maximum_velocity = 1.32  # rad/s
            maximum_acceleration = 16.5  #rad/s^2
            self._robot.logger.info("ArmSensorInspector: Joint move speed is set to SLOW!")
            print("ArmSensorInspector: Joint move speed is set to SLOW!")
        else:
            self._robot.logger.error(
                "ArmSensorInspector: " + str(joint_move_speed) +
                " is an invalid input! Choose ['FAST', 'MEDIUM', 'SLOW'] Continuing with default joint move speed set during map recording!"
            )
            print("ArmSensorInspector: " + str(joint_move_speed) + " is an invalid input! Choose ['FAST', 'MEDIUM', 'SLOW'] Continuing with default joint move speed set during map recording!")
            return
        # Using the above setting, set arm velocity and accelerations
        self._set_max_arm_velocity_and_acceleration(mission, maximum_velocity, maximum_acceleration)

    def _set_max_arm_velocity_and_acceleration(self, mission, maximum_velocity=2.5,
                                               maximum_acceleration=15):
        ''' A helper function that sets the maximum_velocity and maximum_acceleration for the arm joint move.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - maximum_velocity(double): the maximum velocity in rad/s that any joint is allowed to achieve.
                                          If this field is not set, the default value 2.5  will be used.
                - maximum_acceleration(double): the maximum acceleration in rad/s^2 that any joint is allowed to
                                              achieve. If this field is not set, the default value 15 will be used
        '''
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                mission_element.action_wrapper.arm_sensor_pointing.joint_trajectory.maximum_velocity.value = maximum_velocity
                mission_element.action_wrapper.arm_sensor_pointing.joint_trajectory.maximum_acceleration.value = maximum_acceleration
        self._robot.logger.info(
            "ArmSensorInspector: Completed '_set_max_arm_velocity_and_acceleration'!")

    def _enable_stow_arm_in_between_inspection_actions(self, mission):
        ''' A helper function that forces the arm to stow in between inspection actions.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                mission_element.action_wrapper.arm_sensor_pointing.force_stow_override = True
        self._robot.logger.info(
            "ArmSensorInspector: Completed '_enable_stow_arm_in_between_inspection_actions' !")
        print("ArmSensorInspector: Completed '_enable_stow_arm_in_between_inspection_actions' !")

    def _disable_stow_arm_in_between_inspection_actions(self, mission):
        ''' A helper function that forces the arm to stow in between inspection actions.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        for mission_element in mission.elements:
            if (mission_element.name.find(ACTION_NAME) != -1):
                mission_element.action_wrapper.arm_sensor_pointing.force_stow_override = False
        self._robot.logger.info(
            "ArmSensorInspector: Completed '_disable_stow_arm_in_between_inspection_actions' !")
        print("ArmSensorInspector: Completed '_disable_stow_arm_in_between_inspection_actions' !")

    def _disable_battery_monitor(self, mission):
        ''' A helper function that disables battery monitor before starting mission execution. 
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        for mission_element in mission.elements:
            mission_element.battery_monitor.CopyFrom(
                BatteryMonitor(battery_start_threshold=0, battery_stop_threshold=0))
        self._robot.logger.info("ArmSensorInspector: Completed '_disable_battery_monitor' !")

    def _enable_battery_monitor(self, mission, battery_start_threshold=60,
                                battery_stop_threshold=10):
        ''' A helper function that enables battery monitor before starting mission execution given
            the thresholds for start and stop. 
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - battery_start_threshold(double): the robot will continue charging on the dock 
                                                until the robot battery is above this threshold 
                - battery_stop_threshold(double): the robot will stop and return to the dock
                                                if the robot battery is below this threshold
                                                (Note: this only works in continuous missions at this time.)
        '''
        for mission_element in mission.elements:
            mission_element.battery_monitor.CopyFrom(
                BatteryMonitor(battery_start_threshold=battery_start_threshold,
                               battery_stop_threshold=battery_stop_threshold))
        self._robot.logger.info("ArmSensorInspector: Completed '_enable_battery_monitor '!")

    def _disable_dock_after_completion(self, mission):
        ''' A helper function that tells the robot to not dock after completion.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        mission.playback_mode.once.skip_docking_after_completion = True
        self._robot.logger.info("ArmSensorInspector: Completed '_disable_dock_after_completion'!")
        print("ArmSensorInspector: Completed '_disable_dock_after_completion'!")

    def _enable_dock_after_completion(self, mission):
        ''' A helper function that tells the robot to dock after completion.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        mission.playback_mode.once.skip_docking_after_completion = False
        self._robot.logger.info("ArmSensorInspector: Completed '_enable_dock_after_completion '!")
        print("ArmSensorInspector: Completed '_enable_dock_after_completion '!")

    def _set_playback_mode_once(self, mission):
        ''' A helper function that sets the autowalk playback_mode to once.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        mission.playback_mode.once.SetInParent()
        self._robot.logger.info("ArmSensorInspector: Completed '_set_playback_mode_once'!")

    def _set_playback_mode_periodic(self, mission, inspection_interval, number_of_cycles):
        ''' A function that sets the autowalk playback_mode to periodic. 
            Mission runs periodicly every given interval for the given number of cycles. 
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
                - inspection_interval(double): the periodicty of the mission playback in minutes
                - number_of_cycles(int) : the frequency of the inspection in number of cycles
        '''
        interval_seconds = int(inspection_interval * 60)
        duration = duration_pb2.Duration(seconds=interval_seconds // 1, nanos=int(
            (interval_seconds % 1) * 1e9))
        mission.playback_mode.periodic.interval.CopyFrom(duration)
        mission.playback_mode.periodic.repetitions = number_of_cycles
        self._robot.logger.info("ArmSensorInspector: Completed '_set_playback_mode_periodic'!")

    def _set_playback_mode_continuous(self, mission):
        ''' A helper function that sets the autowalk playback_mode to continuous. 
            Mission runs continuously, only stopping when it needs to.
            - Args:
                - mission(walks_pb2.Walk): a mission input to be executed on the robot
        '''
        mission.playback_mode.continuous.SetInParent()
        self._robot.logger.info("ArmSensorInspector: Completed '_set_playback_mode_continuous'!")

    def _ensure_robot_is_localized(self):
        ''' A helper function that localizes robot to the uploaded graph if not localized already. 
            Make sure the robot is in front of the dock or other fiducials within the map.
            - Returns:
                - Boolean indicating if the robot is localized to the uploaded map
        '''
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            self._robot.logger.info(
                'ArmSensorInspector: the robot is not localized to the uploaded graph. Localizing now!'
            )
            try:
                localization = nav_pb2.Localization()
                self._graph_nav_client.set_localization(initial_guess_localization=localization)
                self._robot.logger.info("ArmSensorInspector: the robot is localized!")
                return True
            except Exception as err:
                # Log Exception
                text_message = "ArmSensorInspector: Exception raised in graph_nav_client.set_localization(): [{}] {} - file: {} at line ({})".format(
                    type(err), str(err), err.__traceback__.tb_frame.f_code.co_filename,
                    err.__traceback__.tb_lineno)
                self._robot.logger.error(text_message)
                return False
        return True

    def _ensure_motor_power_is_on(self):
        ''' A helper function that powers robot motors on if not powered on already. 
            - Returns:
                - Boolean indicating if the motors are on
        '''
        try:
            power_state = self._robot_state_client.get_robot_state().power_state
            is_powered_on = (power_state.motor_power_state == power_state.MOTOR_POWER_STATE_ON)
            if is_powered_on:
                self._robot.logger.info('ArmSensorInspector: the robot motors are on already!')
                return True
            if not is_powered_on:
                self._robot.logger.info(
                    'ArmSensorInspector: the robot motors are off! Turning Robot motors on now!')
                # Power on the robot up before proceeding with mission execution
                power_on(self._power_client)
                motors_on = False
                # Wait until motors are on within the feedback_end_time
                fdbk_end_time = 60
                feedback_end_time = time.time() + fdbk_end_time
                while (time.time() < feedback_end_time) or not motors_on:
                    future = self._robot_state_client.get_robot_state_async()
                    state_response = future.result(
                        timeout=10)  # 10 second timeout for waiting for the state response.
                    # Set motors_on state
                    motors_on = (state_response.power_state.motor_power_state ==
                                 robot_state_pb2.PowerState.MOTOR_POWER_STATE_ON)
                    if motors_on:
                        self._robot.logger.info('ArmSensorInspector: the robot motors are on!')
                        return True
                    else:
                        # Motors are not yet fully powered on.
                        time.sleep(.25)
                self._robot.logger.error('ArmSensorInspector: Turn motor power on command timeout!')
                return False
        except Exception as err:
            # Log Exception
            text_message = "ArmSensorInspector: Exception raised in _ensure_motor_power_is_on: [{}] {} - file: {} at line ({})".format(
                type(err), str(err), err.__traceback__.tb_frame.f_code.co_filename,
                err.__traceback__.tb_lineno)
            self._robot.logger.error(text_message)

    def _log_command_status(self, command_name, status):
        ''' A helper function that logs the status of a given command. 
            - Args: 
                - command_name(string): the name of the command
                - status(boolean): indicates if the given command is successful
        '''
        if not status:
            self._robot.logger.info(('ArmSensorInspector: Failed to run {}').format(command_name))
        else:
            self._robot.logger.info(('ArmSensorInspector: Completed {} ').format(command_name))

    def _stow_arm(self):
        ''' A helper function that stows the arm.
            - Returns: 
                - Boolean indicating if inspection is successful
        '''
        state = self._robot_state_client.get_robot_state()
        # return if already stowed
        if state.manipulator_state.stow_state == ManipulatorState.STOWSTATE_STOWED:
            self._robot.logger.info("ArmSensorInspector: Arm is already stowed!")
            return
        stow = RobotCommandBuilder.arm_stow_command()
        close_and_stow = RobotCommandBuilder.claw_gripper_close_command(stow)
        self._robot_command_client.robot_command(close_and_stow)

        # Wait until the arm arrives at the goal within the feedback_end_time
        fdbk_end_time = 10
        feedback_end_time = time.time() + fdbk_end_time
        while (time.time() < feedback_end_time):
            state = self._robot_state_client.get_robot_state()
            if state.manipulator_state.stow_state != ManipulatorState.STOWSTATE_STOWED:
                self._robot.logger.info("ArmSensorInspector: Arm is stowed!")
                return True
            time.sleep(0.1)
        self._robot.logger.info("ArmSensorInspector: _stow_arm command timeout exceeded!")
        return False
