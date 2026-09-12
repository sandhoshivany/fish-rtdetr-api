from ultralytics import RTDETR

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "weights/bests.pt"
DATASET_YAML = "dataset/data.yaml"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = RTDETR(MODEL_PATH)

# --------------------------------------------------
# EVALUATE ON TEST SET
# --------------------------------------------------

metrics = model.val(
    data=DATASET_YAML,
    split="test",
    imgsz=640,
    plots=True,
    save_json=True
)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n========== TEST RESULTS ==========")

print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

print("==================================")