from ultralytics import RTDETR

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "../weights/bests.pt"
DATASET_YAML = "../dataset/data.yaml"

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
# RESULTS
# --------------------------------------------------

precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map50_95 = metrics.box.map

f1 = 2 * (precision * recall) / (precision + recall)

print("\n========== TEST RESULTS ==========")
print(f"Precision:       {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:          {recall:.4f} ({recall * 100:.2f}%)")
print(f"mAP@0.50:        {map50:.4f} ({map50 * 100:.2f}%)")
print(f"mAP@0.50:0.95:   {map50_95:.4f} ({map50_95 * 100:.2f}%)")
print(f"F1-score:        {f1:.4f} ({f1 * 100:.2f}%)")
print("==================================")