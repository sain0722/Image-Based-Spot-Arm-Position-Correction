import copy

from IPython.display import display
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from control.PointCloud import PointCloud, visualization
import open3d as o3d

fx = 217.19888305664062
fy = 217.19888305664062
cy = 111.68077850341797
cx = 87.27774047851562
depth_scale = 1000


class DepthAccumulator:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.n_accumulate = 0
        self.prev_data = None
        self.curr_data = None

        # 누적 처리된 데이터 초기화
        self.cumulative_data = np.zeros((224, 171))

    def add_data(self, data):
        # 이전 데이터 저장
        self.prev_data = self.curr_data
        # 현재 데이터 저장
        self.curr_data = data.copy()

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
            idx = self.buffer_size - 1
        else:
            idx = self.n_accumulate
        self.buffer.append(data)

        if self.n_accumulate == 0:
            self.cumulative_data = self.buffer[idx]
        elif self.n_accumulate == 1:
            self.cumulative_data = self.processing(self.buffer[idx-1], self.buffer[idx])
        elif self.n_accumulate >= 2:
            self.cumulative_data = self.processing(self.cumulative_data, self.buffer[idx])

        self.n_accumulate += 1

    def add_data_2(self, data, threshold):
        data = self.outlier_processing(data, threshold)

        # 이전 데이터 저장
        self.prev_data = self.curr_data
        # 현재 데이터 저장
        self.curr_data = data.copy()

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
            idx = self.buffer_size - 1
        else:
            idx = self.n_accumulate
        self.buffer.append(data)

        if self.n_accumulate == 0:
            self.cumulative_data = self.buffer[idx]
        elif self.n_accumulate == 1:
            self.cumulative_data = self.processing(self.buffer[idx-1], self.buffer[idx])
        elif self.n_accumulate >= 2:
            self.cumulative_data = self.processing(self.cumulative_data, self.buffer[idx])

        self.n_accumulate += 1

    def filter_data(self):
        if len(self.buffer) == 0:
            return None
        return np.mean(self.buffer, axis=0).astype(np.uint16)

    def filter_data_median(self):
        if len(self.buffer) == 0:
            return None
        return np.median(self.buffer, axis=0).astype(np.uint16)

    def filter_with_filling(self):
        data = np.mean(self.buffer, axis=0).astype(np.uint16)

        return data

    def get_filtered_data(self, is_remove_outlier, threshold=1.2):
        cumulative_data = copy.deepcopy(self.cumulative_data)
        if is_remove_outlier:
            cumulative_data = self.outlier_processing(cumulative_data, threshold=threshold)
        return cumulative_data

    def outlier_processing(self, data, threshold=1.2):
        # depth_data에서 0을 제외한 값들을 추출합니다.
        nonzero_values = data[data != 0]

        if len(nonzero_values) == 0:
            return data

        # 추출된 값들의 평균과 표준편차를 계산합니다.
        mean = np.mean(nonzero_values)
        # median = np.median(nonzero_values)
        std = np.std(nonzero_values)

        # 추출된 값들 중에서, 평균에서 표준편차의 threshold배 이상 벗어난 값을 0으로 대체합니다.
        threshold = threshold * std
        outliers = np.abs(nonzero_values - mean) > threshold
        nonzero_values[outliers] = 0

        # 대체된 값을 다시 depth_data에 할당합니다.
        data[data != 0] = nonzero_values
        return data

    def processing(self, prev_data, curr_data):
        cumulative_data = np.zeros((224, 171))

        # 이전 데이터와 현재 데이터 비교하여 적절한 데이터 선택
        mask = (prev_data == 0) & (curr_data != 0)
        cumulative_data[mask] = curr_data[mask]
        mask = (prev_data != 0) & (curr_data == 0)
        cumulative_data[mask] = prev_data[mask]
        mask = (prev_data != 0) & (curr_data != 0)
        cumulative_data[mask] = (prev_data[mask] + curr_data[mask]) / 2

        cumulative_data = cumulative_data.astype(np.uint16)
        return cumulative_data

    def clear(self):
        self.buffer.clear()
        self.n_accumulate = 0
        self.cumulative_data = np.zeros((224, 171))

    def save_data(self, filename):
        # 파일명과 확장자 분리
        name, ext = os.path.splitext(filename)

        # 새로운 파일명 생성
        new_filename = name
        # while os.path.exists(new_filename + ext):
        #     new_filename = f"{name}_{self.n_accumulated}"
        new_filename = f"{name}_{self.n_accumulate}"

        new_filename += ext

        # 배열 저장
        data = self.filter_data_median()
        np.save(new_filename, data)

    def evaluate_accuracy(self, ground_truth):
        filtered_data = self.filter_data()
        if filtered_data is None:
            return None

        rmse = np.sqrt(np.mean((filtered_data - ground_truth) ** 2))
        return rmse


class MeanShiftFilter:
    def __init__(self, buffer_size, radius):
        self.buffer = []
        self.buffer_size = buffer_size
        self.radius = radius

    def add_data(self, data):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def filter_data(self):
        if len(self.buffer) == 0:
            return None
        center_data = self.buffer[-1]
        center_data = center_data.astype(np.float)
        shifted_data = np.zeros_like(center_data)

        for i in range(self.radius):
            kernel_size = 2 * i + 1
            kernel = np.zeros(kernel_size)
            kernel_half = kernel_size // 2
            kernel[kernel_half] = 1

            kernel = signal.gaussian(kernel_size, std=i, sym=True)
            kernel = kernel / np.sum(kernel)

            for j in range(center_data.shape[1]):
                shifted_data[:, j] = np.convolve(center_data[:, j], kernel, mode='same')

        return shifted_data

    def _filter_data(self):
        if len(self.buffer) == 0:
            return None
        center_data = self.buffer[-1]
        center_data = center_data.astype(np.float)
        shifted_data = np.zeros((center_data.shape[0], 3))
        for i in range(self.radius):
            distances = np.sqrt(np.sum((center_data - shifted_data) ** 2, axis=1))
            kernel = distances <= i
            kernel = kernel / np.sum(kernel)
            shifted_data = np.zeros((center_data.shape[0], 3))
            for j in range(center_data.shape[0]):
                shifted_data[j, :] = np.sum(center_data[j, :] * kernel[:, np.newaxis], axis=0)
        return shifted_data

    def filter_data_median(self):
        if len(self.buffer) == 0:
            return None
        return np.median(self.buffer, axis=0)

    def clear(self):
        self.buffer.clear()

    def save_data(self, filename):
        # 파일명과 확장자 분리
        name, ext = os.path.splitext(filename)

        # 새로운 파일명 생성
        new_filename = name
        # while os.path.exists(new_filename + ext):
        #     new_filename = f"{name}_{self.n_accumulated}"
        new_filename = f"{name}_{len(self.buffer)}"

        new_filename += ext

        # 배열 저장
        data = self.filter_data()
        np.save(new_filename, data)


def transformation_depth_to_pcd(depth_image):

    # depth 이미지의 크기를 구합니다.
    height, width = depth_image.shape[:2]

    # 카메라 파라미터를 이용하여 포인트 클라우드를 생성합니다.
    # 포인트 클라우드는 (h*w, 3) 크기의 NumPy 배열입니다.
    pointcloud = np.zeros((height * width, 3), dtype=np.float32)

    index = 0
    for v in range(height):
        for u in range(width):
            # 이미지 상의 (u, v) 좌표의 깊이 값을 가져옵니다.
            depth_value = depth_image[v, u]

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


def blur_filter(image, ksize=(5, 5)):
    """
    블러링 필터링 함수

    Parameters:
        image (ndarray): 입력 이미지 배열
        ksize (tuple): 커널 크기 (기본값: (5,5))

    Returns:
        ndarray: 필터링된 이미지 배열
    """
    blurred = cv2.GaussianBlur(image, ksize, 0)
    return blurred


def mean_shift_filtering(depth_data1, depth_data2):

    pcd_depth_1 = PointCloud(depth_data1)
    pcd_depth_2 = PointCloud(depth_data2)
    # MeanShiftFilter 인스턴스 생성
    filter = MeanShiftFilter(buffer_size=10, radius=5)

    # 첫 번째 depth data 필터링
    filter.add_data(np.asarray(pcd_depth_1.pcd.points))
    filtered_data1 = filter.filter_data()

    # 두 번째 depth data 필터링
    filter.add_data(np.asarray(pcd_depth_2.pcd.points))
    filtered_data2 = filter.filter_data()

    # 필터링된 depth data 출력
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    axes[0, 0].imshow(depth_data1, cmap='gray')
    axes[0, 0].set_title("Depth Data 1")

    pcd1 = PointCloud(filtered_data1, True)
    pcd1.apply_colors('red')
    axes[0, 1].set_aspect("equal")
    # axes[0, 1].set_title("Filtered Data 1 (Depth Map)")

    axes[0, 2].hist(filtered_data1.flatten(), bins=50)
    axes[0, 2].set_title("Filtered Data 1 Histogram")

    axes[1, 0].imshow(depth_data2, cmap='gray')
    axes[1, 0].set_title("Depth Data 2")

    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("Filtered Depth Data 2")

    axes[1, 2].hist(filtered_data2.flatten(), bins=50)
    axes[1, 2].set_title("Filtered Data 2 Histogram")

    plt.show()
    pcd_depth_1.apply_colors('G')
    o3d.visualization.draw_geometries([pcd_depth_1.pcd ,pcd1.pcd], width=640, height=480, window_name="Original Point Cloud 1", left=0, top=500)
    # o3d.visualization.draw_geometries([pcd1], width=640, height=480, window_name="Filtered Point Cloud 1", left=0, top=500)


def blur_flitering(image):
    # 블러링 필터링 적용
    blurred = blur_filter(image, ksize=(5, 5))

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(blurred, cmap="gray")
    plt.title("Blurred Image")
    # plt.show()

    pcd = PointCloud(blurred)
    visualization(pcd.pcd)


# 쿼터니언 값의 정규화 함수
def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm


# 쿼터니언 값의 역(quaternion conjugate) 함수
def conjugate_quaternion(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


# 쿼터니언 값의 곱(quaternion product) 함수
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=np.float64)


if __name__ == '__main__':
    # 예시 데이터
    depth_data1 = np.load("../depth_acm_1_2.npy")
    depth_data2 = np.load("../depth_acm_20_2.npy")

    intrinsic = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]

    pcd1 = PointCloud(depth_data1)
    pcd2 = PointCloud(depth_data2)
    pcd2.matrix_translation(tx=0.1)
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1.pcd, pcd2.pcd, 0.02)

    print(reg_p2p)

    # 정합된 포인트클라우드를 다시 Depth 데이터로 변환
    transformation = reg_p2p.transformation
    transformation = np.multiply(transformation, 1000)
    print(transformation)
    pcd2_registered = pcd2.pcd.transform(transformation)

    visualization(pcd1.pcd, pcd2_registered)
    visualization(pcd2.pcd, pcd2_registered)
