import json


class SourceMetadata:
    def __init__(self):
        self.metadata = {
            "time": None,
            "hand_coord": None,
            "depth_median": None
        }
        self.image_path = None

    def set_data(self, time, hand_coord, depth_median):
        self.metadata = {
            "time": time,
            "hand_coord": hand_coord,
            "depth_median": depth_median
        }

    def set_time(self, time):
        if "time" in self.metadata.keys():
            self.metadata['time'] = time

    def set_hand_coord(self, hand_coord):
        if "hand_coord" in self.metadata.keys():
            self.metadata['hand_coord'] = hand_coord

    def set_depth_median(self, depth_median):
        self.metadata['depth_median'] = depth_median

    def set_image_path(self, image_path):
        self.image_path = image_path

    def save_to_json(self, file_path):
        self.metadata["image_path"] = self.image_path
        with open(file_path, "w") as file:
            json.dump(self.metadata, file, indent=4)


class SourceImageData:
    def __init__(self):
        self.data = {
            'hand_color': None,
            'depth_color': None,
            'hand_color_in_depth_frame': None,
            'depth_data': None,
            'pointcloud': None
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

    def set_data(self, hand_color, depth_color, hand_color_in_depth_frame, depth_data, pointcloud):
        self.data = {
            'hand_color': hand_color,
            'depth_color': depth_color,
            'hand_color_in_depth_frame': hand_color_in_depth_frame,
            'depth_data': depth_data,
            'pointcloud': pointcloud
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
