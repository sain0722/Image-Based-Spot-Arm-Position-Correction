import json


class ImagePath:
    def __init__(self, root, hand_color_file_name, depth_color_file_name,
                 hand_color_in_depth_frame_file_name, depth_data_file_name,
                 pointcloud_file_name, surf_image_file_name=None, overlapped_source_surf=None,
                 icp_result_pcd=None, icp_correspondences=None):
        self.image_path = {
            "root": root,
            "hand_color_file_name": hand_color_file_name,
            "depth_color_file_name": depth_color_file_name,
            "hand_color_in_depth_frame_file_name": hand_color_in_depth_frame_file_name,
            "depth_data_file_name": depth_data_file_name,
            "pointcloud_file_name": pointcloud_file_name,
            "surf_source_target_file_name": surf_image_file_name,
            "overlapped_source_surf_file_name": overlapped_source_surf,
            # "icp_result_pcd_file_name": icp_result_pcd,
            "correspondences_file_name": icp_correspondences
        }

    def save_to_json(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.image_path, file)
