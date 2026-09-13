from pathlib import Path
import yaml

RUN_DIR = Path("runs/detect/train")

args_file = RUN_DIR / "args.yaml"

if not args_file.exists():
    print("args.yaml not found!")
    print("Check your runs/detect/ folder.")
    exit()

with open(args_file, "r") as f:
    args = yaml.safe_load(f)

print("\n========== TRAINING SETUP ==========\n")

print("Model         :", args.get("model"))
print("Image Size    :", args.get("imgsz"))
print("Epochs        :", args.get("epochs"))
print("Batch Size    :", args.get("batch"))
print("Optimizer     :", args.get("optimizer"))
print("Learning Rate :", args.get("lr0"))
print("Device        :", args.get("device"))