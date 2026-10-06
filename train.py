from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu",
    plots=True
)