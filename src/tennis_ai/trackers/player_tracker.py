from ultralytics import YOLO
import cv2
import pickle

__all__ = ['PlayerTracker']


class PlayerTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

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
