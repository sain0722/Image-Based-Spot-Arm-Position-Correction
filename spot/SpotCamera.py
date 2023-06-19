import os
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request


class SpotCamera:
    ImageData = namedtuple('ImageData',
                           ['image_size', 'image_pinhole_intrinsics', 'image_arm_wr1', 'image_hand_sensor'])

    def __init__(self, image_client, gripper_client):
        self.image_client = image_client
        self.ParameterManager = gripper_client
        self.video_mode = False

    def take_image(self):
        source_name = 'hand_color_image'
        image = self.image_client.get_image_from_sources([source_name])
        image_source = image[0].source
        image_size = image_source.cols, image_source.rows
        image_pinhole_intrinsics = image_source.pinhole.intrinsics
        image_arm_wr1 = image[0].shot.transforms_snapshot.child_to_parent_edge_map['arm0.link_wr1'].parent_tform_child
        image_hand_color_sensor = image[0].shot.transforms_snapshot.child_to_parent_edge_map['hand_color_image_sensor'].parent_tform_child
        image_data = self.ImageData(image_size, image_pinhole_intrinsics, image_arm_wr1, image_hand_color_sensor)

        image, _ = image_to_opencv(image[0], auto_rotate=True)
        return image, image_data

    def take_image_from_source(self, camera_name):
        image_client = self.image_client
        source_name = camera_name
        image_sources = image_client.list_image_sources()
        source = [source for source in image_sources if source.name == source_name]
        pixel_format = source[0].pixel_formats[0]
        image_request = [
            build_image_request(source_name, pixel_format=pixel_format)
            # for source in image_sources if source.name == source_name
        ]
        image_responses = image_client.get_image(image_request)

        image_source = image_responses[0].source
        image_size = image_source.cols, image_source.rows
        image_pinhole_intrinsics = image_source.pinhole.intrinsics
        image_arm_wr1 = image_responses[0].shot.transforms_snapshot.child_to_parent_edge_map['arm0.link_wr1'].parent_tform_child
        image_hand_color_sensor = image_responses[0].shot.transforms_snapshot.child_to_parent_edge_map['hand_depth_image_sensor'].parent_tform_child
        image_data = self.ImageData(image_size, image_pinhole_intrinsics, image_arm_wr1, image_hand_color_sensor)

        if pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            image, _, cv_depth = image_to_opencv(image_responses[0], auto_rotate=True)
            return image, cv_depth, image_data
        else:
            image, _ = image_to_opencv(image_responses[0], auto_rotate=True)
            return image

    def get_depth(self):
        pixel_format = image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        image_format = image_pb2.Image.FORMAT_RAW
        request = build_image_request(image_source_name='hand_depth',
                                      quality_percent=100,
                                      image_format=image_format,
                                      pixel_format=pixel_format)
        response = self.image_client.get_image([request])[0]
        dtype = np.uint16
        img = np.frombuffer(response.shot.image.data, dtype=dtype)

        depth_data = img.reshape(response.shot.image.rows,
                                 response.shot.image.cols)
        depth_data = cv2.rotate(depth_data, cv2.ROTATE_90_CLOCKWISE)
        return depth_data

    def get_depth_image(self, iqr, outlier_removal=True):
        depth_data = self.get_depth()

        if outlier_removal:
            depth_data = remove_outlier(depth_data, iqr)

        return depth_data

    @staticmethod
    def depth_to_color(depth_data):
        min_val = np.min(depth_data)
        max_val = np.max(depth_data)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (depth_data - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
        return depth_color

    def get_depth_image_from_source(self, source_name, iqr, outlier_removal=True, is_color=False):
        image_sources = self.image_client.list_image_sources()
        source = [source for source in image_sources if source.name == source_name][0]
        pixel_format = source.pixel_formats[0]
        image_format = source.image_formats[0]
        request = build_image_request(image_source_name='hand_depth',
                                      quality_percent=100,
                                      image_format=image_format,
                                      pixel_format=pixel_format)
        response = self.image_client.get_image([request])[0]

        dtype = np.uint16
        img = np.frombuffer(response.shot.image.data, dtype=dtype)

        depth_data = img.reshape(response.shot.image.rows,
                                 response.shot.image.cols)
        if outlier_removal:
            depth_data = remove_outlier(depth_data, iqr)

        # top_10_positions, top_10_values = get_depth_info(depth_data)
        top_10_positions, top_10_values = get_depth_min_indices(depth_data)
        print(top_10_positions)
        print(top_10_values)
        if is_color:
            min_val = np.min(depth_data)
            max_val = np.max(depth_data)
            depth_range = max_val - min_val
            depth8 = (255.0 / depth_range * (depth_data - min_val)).astype('uint8')
            depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
            depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
            return depth_data, depth_color, top_10_positions, top_10_values

        return depth_data, top_10_positions, top_10_values


def get_depth_min_indices(depth_data):
    # 0을 제외한 값 추출
    nonzero_data = depth_data[depth_data != 0]

    # 가장 작은 10개의 값 추출
    min_values = np.sort(nonzero_data)[:10]

    # 가장 작은 10개의 값들의 인덱스 추출
    min_indices = np.where(np.isin(depth_data, min_values))

    top_10_positions = []
    top_10_values = []
    for i in range(10):
        row, col = min_indices[0][i], min_indices[1][i]
        value = depth_data[row, col]
        top_10_positions.append((row, col))
        top_10_values.append(value)
    return top_10_positions, top_10_values


def get_depth_info(depth_data):
    # depth_data 배열에서 상위 10개 값의 인덱스를 찾습니다.
    top_10_indices = np.argpartition(depth_data.flatten(), -10)[:]
    print(top_10_indices)
    # 상위 10개 값의 위치를 가져옵니다.
    top_10_positions = np.unravel_index(top_10_indices, depth_data.shape)

    # 상위 10개 값의 위치와 값을 저장합니다.
    sorted_top_10_positions = []
    sorted_top_10_values = []

    for i in range(10):
        row, col = top_10_positions[0][i], top_10_positions[1][i]
        value = depth_data[row, col]
        sorted_top_10_positions.append((row, col))
        sorted_top_10_values.append(value)

    # 상위 10개 값의 위치와 값을 내림차순으로 정렬합니다.
    sorted_top_10_positions = [pos for _, pos in
                               sorted(zip(sorted_top_10_values, sorted_top_10_positions), reverse=True)]
    sorted_top_10_values = sorted(sorted_top_10_values, reverse=True)

    return sorted_top_10_positions, sorted_top_10_values


def save_depth_data(name, depth):
    fname = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S") + "_" + name
    root = "depth_data"
    folder = datetime.now().strftime("%Y%m%d")
    saved_folder = os.path.join(root, folder)

    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    np.save(os.path.join(saved_folder, fname), depth)
    return saved_folder, fname


def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        cv_depth = img.reshape(image.shot.image.rows,
                               image.shot.image.cols)
        # saved_folder, fname = save_depth_data(image.source.name, cv_depth)
        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image.shot.image.data, dtype=np.uint8), -1)

        # Convert the visual image from a single channel to RGB, so we can add color
        # visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)

        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling

        # depth_data 배열을 복사합니다.
        depth_data_copy = np.copy(cv_depth)

        # depth_data 배열에서 outlier를 찾습니다.
        q1, q3 = np.percentile(depth_data_copy, [10, 90])
        iqr = q3 - q1
        outlier_threshold = q3 + (1.5 * iqr)
        outliers = np.where(depth_data_copy > outlier_threshold)

        # outlier를 0으로 치환합니다.
        depth_data_copy[outliers] = 0

        cv_depth = depth_data_copy

        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = None
        try:
            depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        except RuntimeWarning:
            print("image 없음")
            # os._exit(1)
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
        # Add the two images together.
        # out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

        if auto_rotate:
            # out = ndimage.rotate(depth_color, ROTATION_ANGLE[image.source.name])
            if image.source.name[0:5] == "front":
                depth_color = cv2.rotate(depth_color, cv2.ROTATE_90_CLOCKWISE)

            elif image.source.name[0:5] == "right":
                depth_color = cv2.rotate(depth_color, cv2.ROTATE_180)
        # pixel_format = image.shot.image.pixel_format

        # image_name = os.path.join(saved_folder, fname) + extension
        # cv2.imwrite(image_name, depth_color)
        return depth_color, extension, cv_depth

    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        # img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
        if image.source.name[0:5] == "front":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        elif image.source.name[0:5] == "right":
            img = cv2.rotate(img, cv2.ROTATE_180)

    # pixel_format = image.shot.image.pixel_format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # else:
    #     print("exception")
    #     print(img.shape)
    # saved_folder, fname = save_depth_data(image.source.name, img)
    # image_name = os.path.join(saved_folder, fname) + extension
    # cv2.imwrite(image_name, img)

    return img, extension


def remove_outlier(depth, iqr_range):
    # depth_data 배열을 복사합니다.
    depth_data_copy = np.copy(depth)

    # depth_data 배열에서 outlier를 찾습니다.
    q1, q3 = np.percentile(depth_data_copy, iqr_range)
    iqr = q3 - q1
    outlier_threshold = q3 + (1.5 * iqr)
    outliers = np.where(depth_data_copy > outlier_threshold)

    # outlier를 0으로 치환합니다.
    depth_data_copy[outliers] = 0

    return depth_data_copy
