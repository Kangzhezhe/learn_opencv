from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('runs/detect/train/weights/best.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='/home/kangzhehao/Documents/holiday/task2/data.yaml', epochs=10, imgsz=640)
