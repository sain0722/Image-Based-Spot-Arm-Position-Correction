import copy
import queue
from collections import deque

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib import pyplot as plt

fx = 217.19888305664062
fy = 217.19888305664062
# 90도 로테이션을 했기 때문에, cx, cy의 좌표값을 서로 바꾸어 줌. 기존: (111.-, 87.-)
cx = 87.27774047851562
cy = 111.68077850341797
depth_scale = 1000


class PointCloud:
    def __init__(self, depth_image, is_pcd=False):
        self.depth_image = depth_image
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.transformation_matrix = np.eye(4)  # 초기값은 항등행렬

        if not is_pcd:
            points, _ = self.transformation_depth_to_pcd(depth_image)
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
        else:
            self.depth_image = np.asarray(depth_image.points)
            self.pcd = depth_image

    def __call__(self, *args, **kwargs):
        return self.pcd

    @staticmethod
    def transformation_depth_to_pcd(depth):
        # depth 이미지의 크기를 구합니다.
        height, width = depth.shape[:2]

        # 카메라 파라미터를 이용하여 포인트 클라우드를 생성합니다.
        # 포인트 클라우드는 (h*w, 3) 크기의 NumPy 배열입니다.
        pointcloud = np.zeros((height * width, 3), dtype=np.float32)

        index = 0
        for v in range(height):
            for u in range(width):
                # 이미지 상의 (u, v) 좌표의 깊이 값을 가져옵니다.
                depth_value = depth[v, u]

                # 깊이 값이 0인 경우는 포인트 클라우드에 추가하지 않습니다.
                if depth_value == 0:
                    continue

                # 이미지 상의 (u, v) 좌표의 3차원 좌표 값을 계산합니다.
                Z = depth_value / depth_scale
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                pointcloud[index, :] = [X, Y, Z]
                index += 1

        # 포인트 클라우드가 저장된 NumPy 배열과 포인트 개수를 반환합니다.
        return pointcloud[:index, :], index

    @staticmethod
    def transformation_pcd_to_depth(pointcloud, index, height=224, width=171):
        # 빈 depth 이미지를 생성합니다.
        depth_image = np.zeros((height, width), dtype=np.float32)

        for i in range(index):
            # 포인트 클라우드의 각 점을 깊이 이미지에 투영합니다.
            X, Y, Z = pointcloud[i, :]
            if Z == 0:
                continue

            # 3차원 좌표를 이미지 좌표 (u, v)로 변환합니다.
            u = int(np.round((X * fx / Z) + cx))
            v = int(np.round((Y * fy / Z) + cy))

            # 이미지 좌표가 이미지 내에 있는 경우, 깊이 값을 저장합니다.
            if 0 <= u < width and 0 <= v < height:
                depth_value = Z * depth_scale
                depth_image[v, u] = depth_value

        # 변환된 depth 이미지를 반환합니다.
        return depth_image

    @staticmethod
    def transformation_pcd_to_depth_vectorized(pointcloud, height=224, width=171):
        depth_image = np.zeros((height, width), dtype=np.float32)

        # Z 값이 0인 포인트 제거
        valid_points = pointcloud[pointcloud[:, 2] != 0]

        # 3D 좌표를 이미지 좌표 (u, v)로 변환
        u = np.round((valid_points[:, 0] * fx / valid_points[:, 2]) + cx).astype(int)
        v = np.round((valid_points[:, 1] * fy / valid_points[:, 2]) + cy).astype(int)

        # 이미지 범위 내의 좌표만 선택
        valid_indices = np.where((0 <= u) & (u < width) & (0 <= v) & (v < height))

        # 이미지에 깊이 값을 저장
        depth_values = valid_points[valid_indices][:, 2] * depth_scale
        depth_image[v[valid_indices], u[valid_indices]] = depth_values

        return depth_image

    def apply_transformation_matrix(self, R, T):
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T

        self.transformation_matrix = np.dot(H, self.transformation_matrix)
        print(self.transformation_matrix)
        # 포인트 클라우드 변환
        self.pcd.transform(H)

    def apply_transformation_matrix_H(self, H):
        self.transformation_matrix = np.dot(H, self.transformation_matrix)
        # print(self.transformation_matrix)
        # 포인트 클라우드 변환
        self.pcd.transform(H)

    def apply_colors(self, channel: str):
        pcd_colors = np.zeros((len(self.pcd.points), 3))
        if channel == 'R':
            channel = 0
        elif channel == 'G':
            channel = 1
        elif channel == 'B':
            channel = 2
        else:
            channel = 0

        pcd_colors[:, channel] = 1.0

        self.pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    def matrix_translation(self, tx=0, ty=0, tz=0):
        R = np.eye(3)

        translation = [tx, ty, tz]
        # self.transformation_matrix[:, :3] += np.array(translation)
        self.apply_transformation_matrix(R, translation)

    def matrix_rotation(self, angle_x=0, angle_y=0, angle_z=0):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(angle_x), -np.sin(angle_x)],
                        [0, np.sin(angle_x), np.cos(angle_x)]])
        R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                        [0, 1, 0],
                        [-np.sin(angle_y), 0, np.cos(angle_y)]])
        R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                        [np.sin(angle_z), np.cos(angle_z), 0],
                        [0, 0, 1]])

        R = np.dot(R_z, np.dot(R_y, R_x))

        translation = [0, 0, 0]
        self.apply_transformation_matrix(R, translation)

    def save(self, fname, ext):
        saved_name = fname + '.' + ext
        o3d.io.write_point_cloud(saved_name, self.pcd, write_ascii=True)

    def apply_sor_filter(self, nb_neighbors=20, std_ratio=2.0):
        # SOR 필터 적용
        pcd, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 정제된 포인트 클라우드 반환
        return pcd


class ICP:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.trans_init = np.eye(4)
        self.reg_p2l = None

        self.target_buffer = deque()
        self.correspondences_pcd_buffer = deque()
        self.lineset_buffer = deque()
        self.transformation_buffer = deque()
        self.icp_result_buffer = deque()

    def set_source(self, pcd):
        self.source = pcd

    def set_target(self, pcd):
        self.target = pcd

    def set_init_transformation(self, tform):
        self.trans_init = tform

    def robust_icp(self, iteration=20, sigma=0.05, threshold=0.005, is_show=False):
        # print("Using the noisy source pointcloud to perform robust ICP.\n")
        # print("Robust point-to-plane ICP, threshold={}:".format(threshold))
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        # print("Using robust loss:", loss)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        # self.source.estimate_normals()
        # self.target.estimate_normals()

        for i in range(iteration):
            self.reg_p2l = o3d.pipelines.registration.registration_icp(
                self.source, self.target, threshold, self.trans_init, p2l)

            # self.source.transform(self.reg_p2l.transformation)

            # print(reg_p2l)
            # print("Transformation is:")
            # print(reg_p2l.transformation)

            self.trans_init = self.reg_p2l.transformation
            if is_show:
                correspondences_array, correspondences_pcd = self.create_correspondence(self.reg_p2l)
                lineset = self.create_lineset(correspondences_array)
                self.draw_registration_result_with_cor(self.source,
                                                       self.target,
                                                       self.trans_init,
                                                       correspondences_pcd,
                                                       lineset)

        correspondences_array, correspondences_pcd = self.create_correspondence(self.reg_p2l)
        lineset = self.create_lineset(correspondences_array)

        self.correspondences_pcd_buffer.append(correspondences_pcd)
        self.lineset_buffer.append(lineset)
        self.target_buffer.append(self.target)
        self.transformation_buffer.append(self.reg_p2l.transformation)
        self.icp_result_buffer.append(self.reg_p2l)

    def create_correspondence(self, icp_result):
        correspondences_array = np.asarray(icp_result.correspondence_set)
        correspondences_color = np.zeros((len(correspondences_array), 3))
        correspondences_color[:, 0] = 1  # 빨간색으로 설정

        source_points = np.asarray(self.source.points)

        correspondences_pcd = o3d.geometry.PointCloud()
        correspondences_pcd.points = o3d.utility.Vector3dVector(source_points[correspondences_array[:, 0], :])
        correspondences_pcd.colors = o3d.utility.Vector3dVector(correspondences_color)

        return correspondences_array, correspondences_pcd

    def create_lineset(self, correspondences_array):
        source_points = np.asarray(self.source.points)
        target_points = np.asarray(self.target.points)

        lines = []
        for idx in range(len(correspondences_array)):
            pt1 = source_points[correspondences_array[idx, 0]]
            pt2 = target_points[correspondences_array[idx, 1]]
            lines.append([pt1, pt2])

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.asarray(lines).reshape(-1, 3))
        lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i + 1] for i in range(0, len(lines), 2)]))

        return lineset

    def draw_registration_result(self, idx):
        self.draw_registration_result_with_cor(self.source,
                                               self.target_buffer[idx],
                                               self.transformation_buffer[idx],
                                               self.correspondences_pcd_buffer[idx],
                                               self.lineset_buffer[idx])

    @staticmethod
    def draw_registration_result_with_cor(source, target, transformation, correspondences, lineset):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        correspondences = copy.deepcopy(correspondences)
        lineset = copy.deepcopy(lineset)

        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        translation = np.array([
            [1, 0, 0, -1.3],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]
        ])
        correspondences.transform(translation)
        lineset.transform(translation)
        source_temp_2 = copy.deepcopy(source_temp)
        target_temp_2 = copy.deepcopy(target_temp)
        source_temp_2.transform(translation)
        target_temp_2.transform(translation)

        # source_temp.transform(transformation)
        target_temp.transform(np.linalg.inv(transformation))

        o3d.visualization.draw_geometries([source_temp_2, target_temp_2, lineset, source_temp, target_temp],
                                          width=1440, height=968, left=50, top=50,
                                          front=[0.026079887874658959, -0.16142771866709718, -0.98653987810649701],
                                          lookat=[-0.6325727553215863, -0.063579100789774134, 0.88815143938110086],
                                          up=[0.0090927819603467599, -0.98679641988701194, 0.1617100708502654],
                                          zoom=0.53999999999999981)

    def run(self):
        sigma = 0.05
        threshold = 0.005
        print("Using the noisy source pointcloud to perform robust ICP.\n")
        print("Robust point-to-plane ICP, threshold={}:".format(threshold))
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        print("Using robust loss:", loss)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        self.target.estimate_normals()
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.source, self.target, threshold, self.trans_init, p2l)
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        self.draw_registration_result(reg_p2l.transformation)

    # def draw_registration_result(self, transformation):
    #     source_temp = copy.deepcopy(self.source)
    #     target_temp = copy.deepcopy(self.target)
    #     source_temp.paint_uniform_color([1, 0.706, 0])
    #     target_temp.paint_uniform_color([0, 0.651, 0.929])
    #     source_temp.transform(transformation)
    #     o3d.visualization.draw([source_temp, target_temp])


def get_trans_init(M, source_depth_median, target_depth_median):
    trans_init = np.identity(4)
    print(f"[SURF] x: {M[0][2]}, y: {M[1][2]}")
    # tx = M[0][2] * 0.0013458950201884
    tx = M[0][2] * 0.002
    # ty = M[1][2] * 0.0025720680142442
    ty = M[1][2] * 0.002
    # 초기행렬의 tz값 추출 (중앙값 계산)
    tz = (target_depth_median - source_depth_median) / 1000
    trans_init[:3, 3] = [tx, ty, tz]

    return trans_init


def execute_surf(img1, img2, ratio_threshold=0.3):
    surf = cv2.xfeatures2d.SURF_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # (3) Create flann matcher
    # print("## (3) Create flann matcher")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

    # print("## (4) Detect keypoints and compute keypointer descriptors")
    kpts1, descs1 = surf.detectAndCompute(gray1, None)
    kpts2, descs2 = surf.detectAndCompute(gray2, None)

    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)

    # (6) Ratio test, to get good matches.
    # print("## (6) Ratio test, to get good matches.")
    good = [m1 for (m1, m2) in matches if m1.distance < ratio_threshold * m2.distance]
    M = None
    found = None

    if len(good) > 4:
        # (queryIndex for the small object, trainIndex for the scene )
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # outlier가 제거된 좋은 매칭 포인트들의 위치들을 추출합니다.
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        # 추출된 좋은 매칭 포인트들의 위치를 사용하여 homography matrix를 다시 추정합니다.
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
        found = cv2.warpPerspective(img2, perspectiveM, (w, h))

        src_coord = []
        dst_coord = []
        for src_pt in src_pts[:]:
            x, y = src_pt[0]
            src_coord.append((x, y))

        for dst_pt in dst_pts[:]:
            x, y = dst_pt[0]
            dst_coord.append((x, y))

        diff = [np.array(s_coord) - np.array(d_coord)
                for s_coord, d_coord in zip(src_coord, dst_coord)]
        x_diff, y_diff = np.transpose(diff)

        x_diff_without_outliers = remove_outlier(x_diff, q1=40, q3=60)
        y_diff_without_outliers = remove_outlier(y_diff, q1=40, q3=60)
        # print(np.mean(x_diff_without_outliers))
        # print(np.mean(y_diff_without_outliers))

    return M, found


def remove_outlier(coord, q1, q3):
    # Calculate the first and third quartile (Q1 and Q3)
    Q1, Q3 = np.percentile(coord, [q1, q3])

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Remove the outliers.
    # data_without_outliers = [x for x in x_diff if lower_bound <= x <= upper_bound]
    data_without_outliers = [x for x in coord if lower_bound <= x <= upper_bound]

    return data_without_outliers


def visualization(*args):
    # pcd = args[0]
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(
    #         pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    #
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    o3d.visualization.draw_geometries(args, left=0, top=30, width=1080, height=720)
    # o3d.visualization.draw([args[0]])
