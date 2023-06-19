import json


class CorrectedMetadata:
    def __init__(self):
        self.metadata = {
            "time": None,
            "hand_coord": None,
            "depth_median": None,
            "surf_pixel": None
        }
        self.image_path = None

    def set_data(self, time, hand_coord, median_diff, surf_pixel):
        self.metadata = {
            "time": time,
            "hand_coord": hand_coord,
            "median_diff": median_diff,
            "surf_pixel": surf_pixel
        }

    def set_time(self, time):
        self.metadata['time'] = time

    def set_hand_coord(self, hand_coord):
        self.metadata['hand_coord'] = hand_coord

    def set_depth_median(self, depth_median):
        self.metadata['depth_median'] = depth_median

    def set_surf_pixel(self, surf_pixel):
        self.metadata['surf_pixel'] = surf_pixel

    def set_image_path(self, image_path):
        self.image_path = image_path

    def save_to_json(self, file_path):
        self.metadata["image_path"] = self.image_path
        with open(file_path, "w") as file:
            json.dump(self.metadata, file, indent=4)


class CorrectedImageData:
    def __init__(self):
        self.data = {
            'hand_color': None,
            'depth_color': None,
            'hand_color_in_depth_frame': None,
            'depth_data': None,
            'pointcloud': None,
            'surf_source_target': None,
            'overlapped_source_surf': None,
            # 'icp_result_pcd': None,
            'correspondences': None
        }
        self._keys = list(self.data.keys())
        self._index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index < len(self._keys):
            key = self._keys[self._index]
            return key, self.data[key]
        else:
            self._index = -1
            raise StopIteration

    def set_data(self, hand_color, depth_color, hand_color_in_depth_frame, depth_data, pointcloud,
                 surf_source_corrected, overlapped_source_surf_corrected, correspondences):
        self.data = {
            'hand_color': hand_color,
            'depth_color': depth_color,
            'hand_color_in_depth_frame': hand_color_in_depth_frame,
            'depth_data': depth_data,
            'pointcloud': pointcloud,
            'surf_source_target': surf_source_corrected,
            'overlapped_source_surf': overlapped_source_surf_corrected,
            'correspondences': correspondences
        }

    def get_hand_color(self):
        return self.data['hand_color']

    def get_depth_color(self):
        return self.data['depth_color']

    def get_hand_color_in_depth_frame(self):
        return self.data['hand_color_in_depth_frame']

    def get_depth_data(self):
        return self.data['depth_data']

    def get_pointcloud(self):
        return self.data['pointcloud']
