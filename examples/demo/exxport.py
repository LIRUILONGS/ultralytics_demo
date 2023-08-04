from ultralytics import YOLO

# Load a model
model = YOLO('C:\\Users\\liruilong\\.yolo_model\\yolov8n-cls.pt')  # load an official model

# Export the model
model.export(format='onnx')
