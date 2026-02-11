from ultralytics import YOLO

model = YOLO("yolo26l-obb.pt")
model.train(data="timbervision.yaml", imgsz=1024, epochs=100, batch=16)

model = YOLO("yolo26l-seg.pt")
model.train(data="timbervision.yaml", imgsz=1024, epochs=100, batch=16)