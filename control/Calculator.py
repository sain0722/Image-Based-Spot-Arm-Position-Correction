import copy
import os
import numpy as np


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
