from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data.yaml", epochs=100, imgsz=640)
