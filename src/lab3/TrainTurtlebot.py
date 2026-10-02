from ultralytics import YOLO


def main():
    # Load pretrained YOLO model
    model = YOLO("yolo26n.pt")

    # Train model on Turtlebot dataset
    model.train(
        data="Turtlebots.yolov8/data.yaml",
        epochs=20,
        imgsz=640
    )


if __name__ == "__main__":
    main()