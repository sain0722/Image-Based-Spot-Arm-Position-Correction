import math

import numpy as np
from bosdyn.client import math_helpers


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    rotation = math_helpers.Quat(w, x, y, z)
    return rotation


def pose_to_homogeneous_matrix(pose):
    # 회전 정보를 행렬로 변환
    rotation_matrix = np.array([
        [1 - 2 * (pose['rotation']['y'] ** 2) - 2 * (pose['rotation']['z'] ** 2),
         2 * (pose['rotation']['x'] * pose['rotation']['y'] - pose['rotation']['z'] * pose['rotation']['w']),
         2 * (pose['rotation']['x'] * pose['rotation']['z'] + pose['rotation']['y'] * pose['rotation']['w'])],
        [2 * (pose['rotation']['x'] * pose['rotation']['y'] + pose['rotation']['z'] * pose['rotation']['w']),
         1 - 2 * (pose['rotation']['x'] ** 2) - 2 * (pose['rotation']['z'] ** 2),
         2 * (pose['rotation']['y'] * pose['rotation']['z'] - pose['rotation']['x'] * pose['rotation']['w'])],
        [2 * (pose['rotation']['x'] * pose['rotation']['z'] - pose['rotation']['y'] * pose['rotation']['w']),
         2 * (pose['rotation']['y'] * pose['rotation']['z'] + pose['rotation']['x'] * pose['rotation']['w']),
         1 - 2 * (pose['rotation']['x'] ** 2) - 2 * (pose['rotation']['y'] ** 2)]
    ])

    # 동차 좌표계 행렬 생성
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = [pose['x'], pose['y'], pose['z']]

    return homogeneous_matrix


def homogeneous_matrix_to_pose(matrix):
    # 위치 정보 추출
    tx, ty, tz = matrix[:3, 3]

    # 회전 정보(사원수) 추출
    trace = np.trace(matrix[:3, :3])
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s

    rotation = {'w': w, 'x': x, 'y': y, 'z': z}

    return {'x': tx, 'y': ty, 'z': tz, 'rotation': rotation}


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q

    rotation_matrix = np.array([
        [1 - 2*(y**2) - 2*(z**2),     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*(x**2) - 2*(z**2),     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*(x**2) - 2*(y**2)]
    ])

    return rotation_matrix


def apply_transformation_to_target(transformation_matrix, target_pose):
    # 입력된 SE3Pose를 동차 좌표계 행렬로 변환
    target_homogeneous_matrix = pose_to_homogeneous_matrix(target_pose)

    # 변환 행렬을 소스 위치에서의 포인트 클라우드를 타겟 위치에서의 포인트 클라우드로 변환하는 행렬로 사용
    # transformed_target_homogeneous_matrix = np.dot(target_homogeneous_matrix, np.linalg.inv(transformation_matrix))
    transformed_target_homogeneous_matrix = np.dot(target_homogeneous_matrix, transformation_matrix)

    # 변환된 동차 좌표계 행렬을 다시 SE3Pose로 변환
    # print(transformed_target_homogeneous_matrix)
    corrected_target_pose = homogeneous_matrix_to_pose(transformed_target_homogeneous_matrix)

    return corrected_target_pose


def apply_spot_coordinate_matrix(transformation_matrix):
    # 변환 행렬의 역행렬
    transformation_matrix = np.linalg.inv(transformation_matrix)

    # 변환 행렬의 좌표계의 배열을 SPOT의 좌표계의 배열에 맞게 재배열합니다.
    # SPOT 좌표계의 y축 (실제 좌표계의 x축)은 대칭이동합니다.
    transformation_matrix[:3, 3] = transformation_matrix[:3, 3][[2, 0, 1]]
    transformation_matrix[0, 3] = -transformation_matrix[0, 3]

    # Transformation matrix의 회전 부분만 추출합니다.
    rotation_matrix = transformation_matrix[:3, :3]

    # SPOT의 x축 (실제 z축)은 대칭이동합니다.
    rotation_matrix[0, 1] = -rotation_matrix[0, 1]
    rotation_matrix[1, 0] = -rotation_matrix[1, 0]

    # 새로운 순서에 맞게 회전 행렬의 행을 재배열합니다.
    rotation_matrix = rotation_matrix[[2, 0, 1]]

    # 새로운 순서에 맞게 회전 행렬의 열을 재배열합니다.
    rotation_matrix = rotation_matrix[:, [2, 0, 1]]

    transformation_matrix[:3, :3] = rotation_matrix

    return transformation_matrix


def rotation_x_axis_test(rotation_matrix, rotation_angle_x):
    # x축 회전 행렬을 생성합니다.
    # 여기에서 회전 각도를 반대로 하고 2로 나누어 x축 회전 값을 조정합니다.
    rotation_angle_x = -rotation_angle_x / 2
    rotation_matrix_x = np.array([
        [np.cos(rotation_angle_x), -np.sin(rotation_angle_x), 0],
        [np.sin(rotation_angle_x), np.cos(rotation_angle_x), 0],
        [0, 0, 1]
    ])

    # x축 회전 행렬을 원본 회전 행렬에 곱합니다.
    rotation_matrix = np.dot(rotation_matrix, rotation_matrix_x)
    return rotation_matrix


def calculate_new_rotation(axis, angle, body_tform_hand_rotation):
    # 각도 변화 (라디안)
    angle = angle * (math.pi / 180)

    # 회전 축
    if axis == "x":
        axis = np.array([1, 0, 0])
    elif axis == "y":
        axis = np.array([0, 1, 0])
    elif axis == "z":
        axis = np.array([0, 0, 1])

    # 새로운 쿼터니언 생성
    new_quaternion = np.array([
        math.cos(angle / 2),
        axis[0] * math.sin(angle / 2),
        axis[1] * math.sin(angle / 2),
        axis[2] * math.sin(angle / 2)
    ])

    # 원래의 쿼터니언
    original_quaternion = [body_tform_hand_rotation.w,
                           body_tform_hand_rotation.x,
                           body_tform_hand_rotation.y,
                           body_tform_hand_rotation.z]

    # 쿼터니언 곱셈을 통해 새로운 회전 쿼터니언 계산
    new_rotation = quaternion_multiply(original_quaternion, new_quaternion)

    return new_rotation


def calculate_new_rotation_multi_axes(axes_angles, body_tform_hand_rotation):
    # 원래의 쿼터니언
    original_quaternion = [body_tform_hand_rotation.w,
                           body_tform_hand_rotation.x,
                           body_tform_hand_rotation.y,
                           body_tform_hand_rotation.z]

    # 각 축에 대해 회전을 계산
    for axis, angle in axes_angles.items():
        if type(original_quaternion) == math_helpers.Quat:
            original_quaternion = [original_quaternion.w,
                                   original_quaternion.x,
                                   original_quaternion.y,
                                   original_quaternion.z]

        # 각도 변화 (라디안)
        angle = angle * (math.pi / 180)

        # 회전 축
        if axis == "x":
            axis_vector = np.array([1, 0, 0])
        elif axis == "y":
            axis_vector = np.array([0, 1, 0])
        elif axis == "z":
            axis_vector = np.array([0, 0, 1])

        # 새로운 쿼터니언 생성
        new_quaternion = np.array([
            math.cos(angle / 2),
            axis_vector[0] * math.sin(angle / 2),
            axis_vector[1] * math.sin(angle / 2),
            axis_vector[2] * math.sin(angle / 2)
        ])

        # 쿼터니언 곱셈을 통해 새로운 회전 쿼터니언 계산
        original_quaternion = quaternion_multiply(original_quaternion, new_quaternion)

    return original_quaternion
