from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('models/yolo5_best.pt')
    model = YOLO('models/yolov8x.pt')

    # predict on given image/video
    # result = model.predict('data/input_video.mp4', conf=0.2, save=True)

    # track on given video
    result = model.track('data/input_videos/input_video.mp4', conf=0.2, save=True)

    print(result)
    for box in result[0].boxes:
        print(box)
