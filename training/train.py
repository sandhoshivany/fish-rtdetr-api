from ultralytics import RTDETR

DATASET = "dataset/data.yaml"
CHECKPOINT = "weights/last.pt"

model = RTDETR(CHECKPOINT)

model.train(
    data=DATASET,
    epochs=50,
    imgsz=640,
    batch=16,
    resume=True
)
