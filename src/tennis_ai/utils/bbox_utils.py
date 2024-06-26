__all__ = ['get_center_of_bbox', 'measure_distance', 'get_foot_position', 'get_closest_keypoint_index',
           'get_height_of_bbox', 'measure_xy_distance']


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), y2


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_index = keypoint_indices[0]
    for key_point_idx in keypoint_indices:
        key_point = (keypoints[key_point_idx * 2], keypoints[key_point_idx * 2 + 1])
        distance = abs(point[1] - key_point[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_index = key_point_idx

    return key_point_index


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
