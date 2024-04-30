import pandas as pd
from ultralytics import YOLO
import cv2
import pickle

__all__ = ['BallTracker']


class BallTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # list of bounding boxes which is going to be empty if there is no ball
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate(method='quadratic')
        # duplicate the first frames in order to not have empty first values which will crash the program
        # bfill - Fill NA/NaN values by using the next valid observation to fill the gap.
        df_ball_positions = df_ball_positions.bfill()
        # convert back into the same format that we got the detections from
        # list of dicts where 1 is id and x is the bounding box
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_and_track_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_and_track_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            bbox_result = box.xyxy.tolist()[0]  # x_min, y_min, x_max, y_max of the bounding box

            ball_dict[1] = bbox_result  # if the object is a ball, we add it to the ball_dict

        return ball_dict

    def draw_bboxes(self, video_frames: list, ball_detections: list) -> list:
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            # draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x_min, y_min, x_max, y_max = bbox
                frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)
                # 2 means that it is not going to be filled, but only the border
                cv2.putText(frame, f"Ball ID:{track_id}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
