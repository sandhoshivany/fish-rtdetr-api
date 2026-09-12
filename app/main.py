from fastapi import FastAPI, UploadFile, File, Form

from app.detector import (
    detect_objects,
    CONFIDENCE_THRESHOLD
)

from app.reasoning import (
    needs_detector,
    answer_question
)

import shutil
import os


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fish RT-DETR Detection API",
    description=(
        "RT-DETR fish species detection with "
        "confidence-aware reasoning"
    ),
    version="1.0.0"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fish RT-DETR API is running",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


# ============================================================
# DETECTION ENDPOINT
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    os.makedirs("temp", exist_ok=True)

    image_path = "temp/input.jpg"

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    with open(image_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------------
    # Run detector
    # --------------------------------------------------------

    detections = detect_objects(image_path)

    # --------------------------------------------------------
    # Determine information status
    # --------------------------------------------------------

    if len(detections) == 0:

        information_status = "insufficient_information"

        message = (
            "No sufficiently confident detections "
            "from the trained classes were found."
        )

    else:

        information_status = "sufficient_information"

        message = (
            "Confident detections from the trained "
            "classes were found."
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "confidence_threshold": CONFIDENCE_THRESHOLD,

        "information_status": information_status,

        "message": message,

        "detections": detections
    }


# ============================================================
# REASONING ENDPOINT
# ============================================================

@app.post("/reason")
async def reason(

    file: UploadFile = File(...),

    question: str = Form(...)
):

    os.makedirs("temp", exist_ok=True)

    image_path = "temp/input.jpg"

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    with open(image_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # ========================================================
    # INTENT ROUTING
    # ========================================================

    if not needs_detector(question):

        return {

            "question": question,

            "intent": "not_image_related",

            "answer": (
                "This question does not require "
                "image detection."
            )
        }

    # ========================================================
    # RUN DETECTOR
    # ========================================================

    detections = detect_objects(image_path)

    # ========================================================
    # CONFIDENCE GUARDRAIL
    # ========================================================

    if not detections:

        return {

            "question": question,

            "intent": "image_detection",

            "confidence_threshold": CONFIDENCE_THRESHOLD,

            "information_status": "insufficient_information",

            "detections": [],

            "answer": (
                "Insufficient information: "
                "no sufficiently confident detections "
                "from the trained classes were found "
                "in the image."
            )
        }

    # ========================================================
    # STRUCTURED REASONING
    # ========================================================

    answer = answer_question(
        question,
        detections
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "question": question,

        "intent": "image_detection",

        "confidence_threshold": CONFIDENCE_THRESHOLD,

        "information_status": "sufficient_information",

        "detections": detections,

        "answer": answer
    }