from ultralytics import RTDETR


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "weights/bests.pt"

# Strict minimum confidence for reasoning
CONFIDENCE_THRESHOLD = 0.50

# Smaller inference size reduces RAM usage
IMAGE_SIZE = 640


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = RTDETR(MODEL_PATH)


# ============================================================
# GET TRAINED CLASSES
# ============================================================

TRAINED_CLASSES = {
    str(name).lower()
    for name in model.names.values()
}


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_objects(image_path):

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        save=False,
        verbose=False,
        device="cpu"
    )

    detections = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            # ------------------------------------------------
            # Safety check: only allow trained classes
            # ------------------------------------------------

            if class_name.lower() not in TRAINED_CLASSES:
                continue

            # ------------------------------------------------
            # Confidence guardrail
            # ------------------------------------------------

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    return detections