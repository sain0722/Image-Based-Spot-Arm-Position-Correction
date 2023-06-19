import math
import threading
import time

import numpy as np
from bosdyn import geometry
from bosdyn.api import basic_command_pb2, geometry_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import Vec3
from bosdyn.client import frame_helpers
from bosdyn.client.math_helpers import Quat
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2


def make_orthogonal(primary, secondary):
    # Gram schmidt
    p = primary / np.linalg.norm(primary, ord=2, axis=0, keepdims=True)
    s = secondary / np.linalg.norm(secondary, ord=2, axis=0, keepdims=True)

    u = np.subtract(s, np.multiply(np.dot(p, s) / np.dot(s, s), p))

    normalized_u = u / np.linalg.norm(u, ord=2, axis=0, keepdims=True)
    return normalized_u


def is_fiducial_horizontal(odom_tform_fiducial):
    zhat_ewrt_fiducial = Vec3(x=0, y=0, z=1)
    zhat_ewrt_odom = odom_tform_fiducial.rot.transform_vec3(zhat_ewrt_fiducial)
    zhat_ewrt_odom_np = np.array([
        zhat_ewrt_odom.x, zhat_ewrt_odom.y, zhat_ewrt_odom.z
    ])

    xhat_ewrt_odom_np = np.array([0.0, 0.0, 1.0])
    # Check to see if the fiducial Z-axis is aligned with gravity (0, 0, 1)
    if abs(np.dot(xhat_ewrt_odom_np, zhat_ewrt_odom_np)) > 0.95:
        # The fiducial is probably on the floor
        return True
    return False


def get_nearest_fiducial(fiducials):
    """Determine the fiducial closest to the body position."""
    assert len(fiducials) >= 1
    odom_tform_body = frame_helpers.get_odom_tform_body(fiducials[0].transforms_snapshot)

    def calculate_3d_distance(transform):
        return math.sqrt(transform.x ** 2 + transform.y ** 2 + transform.z ** 2)

    filtered_fiducials = [fid for fid in fiducials if fid.apriltag_properties.tag_id < 500]
    body_tform_fiducials = [(fid, frame_helpers.get_a_tform_b(fid.transforms_snapshot,
                                                              frame_helpers.BODY_FRAME_NAME,
                                                              fid.apriltag_properties.frame_name_fiducial_filtered))
                            for fid in filtered_fiducials]

    closest_fiducial, _ = min(body_tform_fiducials, key=lambda item: calculate_3d_distance(item[1]))

    return closest_fiducial


def get_target_fiducial(fiducials, fiducial_id):
    filtered_fiducials = [fid for fid in fiducials if fid.apriltag_properties.tag_id < 500]
    target_fiducials = [fid for fid in filtered_fiducials if str(fid.apriltag_properties.tag_id) == fiducial_id]
    return target_fiducials


def offset_tag_pose_with_rotation(odom_tform_fiducial, dist_margin):
    zhat_ewrt_fiducial = Vec3(x=0, y=0, z=1)
    zhat_ewrt_odom = odom_tform_fiducial.rot.transform_vec3(zhat_ewrt_fiducial)
    zhat_ewrt_odom_np = np.array(
        [zhat_ewrt_odom.x, zhat_ewrt_odom.y, zhat_ewrt_odom.z]
    )

    xhat_ewrt_odom_np = np.array([0.0, 0.0, 1.0])

    # Check to see if the fiducial Z-axis is aligned with gravity
    if is_fiducial_horizontal(odom_tform_fiducial):
        xhat_ewrt_fiducial = Vec3(x=1, y=0, z=0)

        # Vec3
        xhat_ewrt_odom = odom_tform_fiducial.rot.transform_vec3(xhat_ewrt_fiducial)
        xhat_ewrt_odom_np = np.array([xhat_ewrt_odom.x, xhat_ewrt_odom.y, xhat_ewrt_odom.z])
        zhat_ewrt_odom_np = np.array([0.0, 0.0, 1.0])

        xhat_ewrt_odom_np = make_orthogonal(zhat_ewrt_odom_np, xhat_ewrt_odom_np)
        goal_position_rt_fiducial = Vec3(x=-dist_margin, y=0, z=0)
    else:
        zhat_ewrt_odom_np = make_orthogonal(xhat_ewrt_odom_np, zhat_ewrt_odom_np)
        goal_position_rt_fiducial = Vec3(x=0, y=0, z=dist_margin)

    goal_position_rt_odom = odom_tform_fiducial.transform_vec3(goal_position_rt_fiducial)

    yhat = np.cross(zhat_ewrt_odom_np, xhat_ewrt_odom_np)
    mat = np.array([xhat_ewrt_odom_np, yhat, zhat_ewrt_odom_np]).transpose()
    heading = Quat.from_matrix(mat)

    goal_position_rt_odom_np = np.array([
        goal_position_rt_odom.x, goal_position_rt_odom.y, goal_position_rt_odom.z
    ])

    # return goal_position_rt_odom_np, heading
    return goal_position_rt_odom, heading


def set_default_body_control():
    """Set default body control params to current body position"""
    footprint_R_body = geometry.EulerZXY()
    position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
    rotation = footprint_R_body.to_quaternion()
    pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
    point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
    traj = trajectory_pb2.SE3Trajectory(points=[point])
    return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)


def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded."""
    end_time = time.time() + timeout_sec

    while (timeout_sec is None or time.time() < end_time) and threading.current_thread().is_alive():
        feedback_resp = command_client.robot_command_feedback(cmd_id)

        current_state = feedback_resp.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
        movement_state = feedback_resp.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.body_movement_status
        if current_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
            print("Successfully reached the goal!")
            return True

        elif (current_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_GOING_TO_GOAL and
              movement_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.BODY_STATUS_SETTLED):
            print("Body finished moving even though goal not fully achieved.")
            return True

        time.sleep(0.1)

    print("Trajectory command timeout exceeded (or cancelled).")
    return False


def block_for_stance_cmd(command_client, cmd_id, timeout_sec):
    """Helper that blocks until a stance command reaches STATUS_AT_GOAL or a timeout is
        exceeded."""
    end_time = time.time() + timeout_sec

    while (timeout_sec is None or time.time() < end_time) and threading.current_thread().is_alive():
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        current_state = feedback_resp.feedback.synchronized_feedback.mobility_command_feedback.stance_feedback.status
        if current_state == basic_command_pb2.StanceCommand.Feedback.STATUS_STANCED:
            print("Robot has achieved the desired stance.")
            return True

        time.sleep(0.1)

    return False
