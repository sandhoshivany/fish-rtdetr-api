<<<<<<< HEAD
# Fish RT-DETR Object Detection System

An end-to-end object detection system using RT-DETR and FastAPI.

## Project Overview

This project detects fish species from images using a fine-tuned RT-DETR model.

The system provides:

- Object detection
- Bounding boxes
- Confidence scores
- Natural-language image questions
- Simple handwritten reasoning
- Confidence-based insufficient-information responses

## Model

RT-DETR-L

## Dataset

Fish Detection Dataset

The dataset contains 13 fish classes.

Dataset source and license will be documented here.

## API

### POST /detect

Accepts an image and returns detected fish.

### POST /reason

Accepts an image and a natural-language question.

The system decides whether object detection is required and then reasons over the detector output.

## Running the API

```bash
uvicorn app.main:app --reload
=======
# fish-rtdetr-api
>>>>>>> 7b7c52587fe0201a85296ac522a485222d7fc3c5
