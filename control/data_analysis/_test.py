import os

import cv2
import numpy as np
from natsort import natsorted

from control.data_analysis.visualize_utils import *


def load_npy_files(path):
    npy_files = []
    file_names = natsorted(os.listdir(path))

    # 디렉토리 내의 모든 파일에 대해
    for filename in file_names:
        # 파일이 .npy로 끝나면
        if filename.endswith(".npy"):
            # 파일을 읽어서 리스트에 추가
            depth = np.load(os.path.join(path, filename))
            npy_files.append(depth)
    return npy_files


def load_jpg_files(path):
    jpg_files = []

    file_names = natsorted(os.listdir(path))
    # 디렉토리 내의 모든 파일에 대해
    for filename in file_names:
        # 파일이 .npy로 끝나면
        if filename.endswith(".jpg"):
            # 파일을 읽어서 리스트에 추가
            image_bgr = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            jpg_files.append(image_rgb)

    return jpg_files


def load_txt_files(path):
    transformations = []
    file_names = natsorted(os.listdir(path))
    # 디렉토리 내의 모든 파일에 대해
    for filename in file_names:
        # 파일이 .txt로 끝나면
        if filename.endswith(".txt"):
            # 파일을 읽어서 리스트에 추가
            txt_file = np.loadtxt(os.path.join(path, filename), delimiter=",")
            transformations.append(txt_file)
    return transformations


def compare_target_and_corrected(root):
    # target_dir = os.path.join(root, "target")
    corrected_dir = os.path.join(root, "corrected")

    overlapped_images = load_jpg_files(root)
    corrected_images = load_jpg_files(corrected_dir)

    title = os.path.basename(root)
    # compare_continuous_color_data_plot(overlapped_images, corrected_images, title)

    animate = Animate(overlapped_images, corrected_images)
    animate.set_title(title)
    animate.start()


def analysis_transformations(transformations):
    # 행렬 원소 추출
    elements = np.array([[mat[i, j] for mat in transformations] for i in range(4) for j in range(4)])

    # 각 원소를 시간에 따라 시각화
    for i, element in enumerate(elements):
        plt.plot(element, label=f'Element {i+1}')

    plt.legend()
    plt.show()


def main():
    root = "D:/TestSW/Source/20230531/data/trend_analysis/20230601"

    x_plus_root = os.path.join(root, "x_plus")
    x_minus_root = os.path.join(root, "x_minus")

    y_plus_root = os.path.join(root, "y_plus")
    y_minus_root = os.path.join(root, "y_minus")

    z_plus_root = os.path.join(root, "z_plus")
    z_minus_root = os.path.join(root, "z_minus")

    target_dir = os.path.join(y_plus_root, "target")
    target_npy_files = load_npy_files(target_dir)
    target_color_images = load_jpg_files(target_dir)

    transformation_y_plus = load_txt_files(y_plus_root)
    transformation_z_plus = load_txt_files(z_plus_root)
    transformation_y_minus = load_txt_files(y_minus_root)
    analysis_transformations(transformation_y_plus)
    # visualize_transformations(transformation_y_plus)
    # visualize_transformations(transformation_y_minus)

    # plot_transformation_diff(transformation_y_minus)
    # plot_heatmap_translations(transformation_z_plus)
    # plot_3d_translations(transformation_y_minus)

    # view_continuous_depth_data_plot(target_npy_files)
    # compare_target_and_corrected(y_plus_root)
    # compare_target_and_corrected(y_minus_root)
    # compare_target_and_corrected(z_plus_root)
    # compare_target_and_corrected(z_minus_root)
    # compare_target_and_corrected(x_plus_root)
    # compare_target_and_corrected(x_minus_root)


if __name__ == '__main__':
    main()
