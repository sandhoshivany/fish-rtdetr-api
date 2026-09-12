from ultralytics import RTDETR

DATASET = "dataset/data.yaml"

model = RTDETR("rtdetr-l.pt")

model.train(
    data=DATASET,
    epochs=10,
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    project="runs",
    name="fish_rtdetr"
)