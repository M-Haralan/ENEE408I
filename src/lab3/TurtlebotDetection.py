from ultralytics import YOLO


def main():
    # Replace this path with the best.pt path printed after training
    model = YOLO("runs/detect/train/weights/best.pt")

    # Run Turtlebot detection using webcam
    model.predict(
        source=0,
        show=True
    )


if __name__ == "__main__":
    main()