from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2


def main():
    # read video
    input_video_path = "data/input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')

    # detect balls
    ball_tracker = BallTracker(model_path='models/yolo5_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path='tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # detect court lines
    court_line_detector = CourtLineDetector(model_path='models/keypoints_model.pth')
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # create mini court
    mini_court = MiniCourt(video_frames[0])

    # detect ball hits
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)

    # Draw output
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    # Draw court lines
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw frame number in the top left corner of the video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # save video
    save_video(output_video_frames, "data/output_videos/output_video.avi")


if __name__ == "__main__":
    main()
