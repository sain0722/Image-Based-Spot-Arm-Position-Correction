import pandas as pd
from matplotlib import pyplot as plt, animation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_transformation_diff(transformations):
    # 이동 벡터와 회전 벡터 추출
    translations = [t[:3, 3] for t in transformations]
    rotations = [t[:3, :3] for t in transformations]

    # 차분 계산
    translation_diffs = np.diff(translations, axis=0)
    rotation_diffs = np.diff(rotations, axis=0)

    # 차분 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(translation_diffs)
    plt.title("Translation Differences")
    plt.xlabel("Transformation Index")
    plt.ylabel("Difference")
    plt.legend(['x', 'y', 'z'])

    plt.subplot(2, 1, 2)
    plt.plot(rotation_diffs.ravel())
    plt.title("Rotation Differences")
    plt.xlabel("Transformation Index")
    plt.ylabel("Difference")

    plt.tight_layout()
    plt.show()


def plot_3d_translations(transformations):
    # 이동 성분 추출
    translations = [t[:3, 3] for t in transformations]  # 이동 성분

    # 이동 벡터의 누적값 계산
    cumulative_translations = np.cumsum(translations, axis=0)

    # 이동 성분을 3D 공간에 플롯
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # x, y, z 성분 분리
    xs = cumulative_translations[:, 0]
    ys = cumulative_translations[:, 1]
    zs = cumulative_translations[:, 2]

    # 각 이동 벡터를 화살표로 표시
    for i in range(len(translations) - 1):
        ax.quiver(xs[i], ys[i], zs[i], translations[i][0], translations[i][1], translations[i][2],
                  color='b', alpha=0.5, length=0.8, arrow_length_ratio=0.05)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Translation Vectors with Directions')

    plt.show()


def plot_heatmap_translations(transformations):
    # 이동 성분 추출
    translations = [t[:3, 3] for t in transformations]  # 이동 성분

    # 데이터 프레임 생성
    df = pd.DataFrame(translations, columns=['x', 'y', 'z'])

    # 히트맵 그리기
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, cmap='coolwarm')
    plt.title("Heatmap of Translation Vectors")
    plt.show()


def view_continuous_depth_data_plot(data_list):
    # 가정: 'depth_data_list'는 연속적인 2차원 깊이 데이터 리스트
    fig = plt.figure()

    def animate(i):
        plt.clf()  # 이전 프레임 지우기
        plt.imshow(data_list[i], cmap='hot', interpolation='nearest')
        plt.colorbar(label='Depth')
        plt.title(f'Depth Heatmap {i}')

    ani = animation.FuncAnimation(fig, animate, frames=len(data_list))

    plt.show()


def view_continuous_color_data_plot(data_list):
    fig = plt.figure()

    def animate(i):
        plt.clf()  # 이전 프레임 지우기
        plt.imshow(data_list[i])
        plt.title(f'Color Image {i}')

    ani = animation.FuncAnimation(fig, animate, frames=len(data_list))

    plt.show()


def compare_continuous_color_data_plot(data_list1, data_list2, title="Compare Data"):
    assert len(data_list1) == len(data_list2), "두 이미지 리스트의 길이가 같아야 합니다."

    fig, axs = plt.subplots(1, 2, figsize=(18, 12))  # 1행 2열의 subplot 생성
    fig.suptitle(title)

    def animate(i):
        axs[0].cla()  # 첫 번째 subplot 초기화
        axs[1].cla()  # 두 번째 subplot 초기화

        axs[0].imshow(data_list1[i])
        axs[0].set_title(f'Target - {i+1}')

        axs[1].imshow(data_list2[i])
        axs[1].set_title(f'Corrected - {i+1}')

    ani = animation.FuncAnimation(fig, animate, frames=len(data_list1), interval=500)

    plt.show()


def visualize_transformations(transformations):
    # 회전과 이동 성분 추출
    rotations = [t[:3, :3] for t in transformations]  # 회전 성분
    translations = [t[:3, 3] for t in transformations]  # 이동 성분

    # 각 변환 행렬의 회전과 이동 성분을 시간에 따라 플롯
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 회전 성분 플롯
    for i in range(3):
        for j in range(3):
            axs[0].plot([r[i, j] for r in rotations], label=f"R{i}{j}")
    axs[0].set_title("Rotations")
    axs[0].legend()

    # 이동 성분 플롯
    for i in range(3):
        axs[1].plot([t[i] for t in translations], label=f"T{i}")
    axs[1].set_title("Translations")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


class Animate:
    def __init__(self, data_list1, data_list2):
        assert len(data_list1) == len(data_list2), "두 이미지 리스트의 길이가 같아야 합니다."
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.paused = False

    def set_title(self, title):
        self.fig.suptitle(title)

    def animate(self, i):
        if not self.paused:
            self.axs[0].cla()
            self.axs[1].cla()

            self.axs[0].imshow(self.data_list1[i])
            self.axs[0].set_title(f'Target - {i+1}')

            self.axs[1].imshow(self.data_list2[i])
            self.axs[1].set_title(f'Corrected - {i+1}')

    def onClick(self, event):
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.data_list1), interval=500)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        plt.show()