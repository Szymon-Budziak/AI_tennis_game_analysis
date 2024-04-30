from ultralytics import YOLO
import cv2
import pickle
import sys

sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

__all__ = ['PlayerTracker']


class PlayerTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detection_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detection_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items()
                                    if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players(self, court_keypoints, players_dict):
        distances = []
        for track_id, bbox in players_dict.items():
            player_center = get_center_of_bbox(bbox)
            # calculate distance between the player and each keypoint of the court
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])  # x, y positions
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id, min_distance))

        # sort the distances by the second element of the tuple in ascending order
        distances.sort(key=lambda x: x[1])
        # choose the first 2 players with the smallest distance
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_and_track_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_and_track_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        # persist tells the model should keep tracking the previous frame, we take 0 because it is only 1 image
        # persist is available only in the track method
        id_names_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            bbox_result = box.xyxy.tolist()[0]  # x_min, y_min, x_max, y_max of the bounding box
            obj_cls_id = box.cls.tolist()[0]  # class id of the object
            obj_cls_name = id_names_dict[obj_cls_id]  # map id to the name

            if obj_cls_name == 'person':
                player_dict[track_id] = bbox_result  # if the object is a person, we add it to the player_dict

        return player_dict

    def draw_bboxes(self, video_frames: list, player_detections: list) -> list:
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x_min, y_min, x_max, y_max = bbox
                frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                # 2 means that it is not going to be filled, but only the border
                cv2.putText(frame, f"Player ID:{track_id}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
