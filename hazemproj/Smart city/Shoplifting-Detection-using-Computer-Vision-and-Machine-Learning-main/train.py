from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov12s.yaml')
    results = model.train(data='FYP-Shoplift-1/data.yaml', epochs=10)
