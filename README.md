#  Constrained Object Detection & Reasoning API

### RT-DETR-based Fish Detection with a Minimal Reasoning Layer

An end-to-end Computer Vision + Applied ML Engineering project built for the **RAP Pre-Hackathon Screening — Round 1**.

The system fine-tunes **RT-DETR** on a custom fish object-detection dataset, exposes the trained model through a **FastAPI REST API**, and adds a lightweight hand-written reasoning layer that converts natural-language questions into answers based on structured detection results.

---

##  Project Overview

The objective is to build an object detection system that can:

1. Detect multiple fish species in an image.
2. Return detected classes, bounding boxes, and confidence scores.
3. Accept natural-language questions about an image.
4. Decide whether the detector is required.
5. Reason over detection results without using an agentic framework.
6. Explicitly return **"insufficient information"** when the detection output is not reliable enough to answer.

### System Pipeline

```text
                    Input Image
                         │
                         ▼
                 ┌───────────────┐
                 │   RT-DETR     │
                 │ Object Detector│
                 └───────┬───────┘
                         │
                         ▼
              Structured Detections
              ┌────────────────────┐
              │ Class              │
              │ Confidence         │
              │ Bounding Box       │
              └─────────┬──────────┘
                        │
                        ▼
                 Reasoning Layer
                        │
          ┌─────────────┴─────────────┐
          │                           │
    Detection needed?            Not needed
          │                           │
          ▼                           ▼
   Reason over results         Direct response
          │
          ▼
     Final Answer
```

---

# Problem Statement

Build and deploy an object detection system using **RT-DETR** on a domain-specific dataset, expose it through an API, and implement a minimal reasoning layer capable of answering natural-language questions about the detected objects.

The project uses **fish species detection** as the real-world application domain.

Unlike standard COCO object detection, the selected classes are domain-specific fish species, requiring domain-specific training rather than simply using an off-the-shelf COCO detector.

---

# Why Fish Detection?

Fish detection provides a challenging computer-vision problem because underwater images frequently contain:

* Small objects
* Occlusion
* Motion blur
* Similar-looking species
* Variable lighting
* Complex backgrounds
* Different orientations and poses
* Multiple fish appearing simultaneously

These characteristics make the domain useful for evaluating the practical limitations of an object detection model.

The project focuses on detecting **13 fish species**.

---

#  Dataset

The dataset contains images annotated with bounding boxes for 13 fish classes.

### Classes

```text
1. AngelFish
2. BlueTang
3. ButterflyFish
4. ClownFish
5. GoldFish
6. Gourami
7. MorishIdol
8. PlatyFish
9. RibbonedSweetlips
10. ThreeStripedDamselfish
11. YellowCichlid
12. YellowTang
13. ZebraFish
```

### Dataset Structure

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

The dataset is maintained in YOLO-compatible annotation format and converted/used for RT-DETR training through the selected implementation.

### Dataset Configuration

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 13

names:
  [
    'AngelFish',
    'BlueTang',
    'ButterflyFish',
    'ClownFish',
    'GoldFish',
    'Gourami',
    'MorishIdol',
    'PlatyFish',
    'RibbonedSweetlips',
    'ThreeStripedDamselfish',
    'YellowCichlid',
    'YellowTang',
    'ZebraFish'
  ]
```

---

# Dataset Split

The dataset is separated into:

* **Training set** — used to learn model parameters
* **Validation set** — used during training for model selection and monitoring
* **Test set** — kept separate for final evaluation

The test set is not used to optimize the model.

This separation helps reduce the risk of evaluating the model on images it has already seen during training.

> Exact dataset source, dataset counts, and split percentages are documented in the project report.

---

#  Model

## RT-DETR

The project uses **RT-DETR (Real-Time DEtection TRansformer)** for object detection.

RT-DETR combines transformer-based detection with real-time inference characteristics and provides an alternative to conventional CNN-based detectors.

### Why RT-DETR?

RT-DETR was selected because:

* It is specifically designed for real-time object detection.
* It provides end-to-end detection without traditional NMS-based post-processing in its architecture.
* It performs well on multi-object detection tasks.
* It satisfies the screening requirement to use RT-DETR.
* It provides a strong foundation for domain-specific fine-tuning.

The model was **fine-tuned on the selected fish dataset** rather than using an unchanged COCO detector.

---

#  Training

Training was performed using the **Ultralytics RT-DETR implementation**.

Example training configuration:

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=YOUR_BATCH_SIZE
)
```

### Training Configuration

| Parameter         | Value                             |
| ----------------- | --------------------------------- |
| Model             | RT-DETR                           |
| Implementation    | Ultralytics                       |
| Number of classes | 13                                |
| Image size        | 640 × 640                         |
| Epochs            | 50                                |
| Optimizer         | Default Ultralytics configuration |
| Dataset           | Custom fish dataset               |
| Task              | Object Detection                  |

> The exact training configuration used for the submitted checkpoint is available in the training script.

---

# Model Evaluation

The final model was evaluated on the held-out test set.

### Results

| Metric             |     Result |
| ------------------ | ---------: |
| **mAP@0.50**       | **87.85%** |
| **mAP@0.50:0.95**  | **69.22%** |
| **Precision**      | **82.08%** |
| **Recall**         | **85.44%** |
| **Test Images**    |    **700** |
| **Test Instances** |  **1,134** |
| **Classes**        |     **13** |

### What These Metrics Mean

**Precision**

Measures how many predicted detections are actually correct.

```text
Precision = TP / (TP + FP)
```

Higher precision means fewer false-positive detections.

---

**Recall**

Measures how many of the actual objects were successfully detected.

```text
Recall = TP / (TP + FN)
```

Higher recall means fewer missed objects.

---

**mAP@0.50**

Measures mean Average Precision using an IoU threshold of 0.50.

It evaluates whether predicted bounding boxes sufficiently overlap with the ground-truth boxes.

---

**mAP@0.50:0.95**

Computes mAP across multiple IoU thresholds from 0.50 to 0.95.

This is a stricter measure of both detection and localization quality.

---

##  Important Evaluation Note

These metrics represent performance on the available held-out test set.

They **do not guarantee equivalent performance on unseen images**.

The final system must therefore be evaluated on the hidden evaluation set used by the screening process.

---

#  Failure Case Analysis

A major goal of this project is to understand where the detector fails rather than assuming that a high aggregate metric means perfect performance.

### Failure Case 1 — Small / Occluded Fish

Small fish or partially occluded fish can be missed or detected with low confidence.

**Likely causes:**

* Very few pixels representing the object
* Reduced visual features
* Partial visibility
* Overlap with other fish

---

### Failure Case 2 — Similar-Looking Species

Some species have similar body shapes, colors, or patterns.

This can result in one fish species being incorrectly classified as another.

**Likely causes:**

* Similar visual features
* Limited distinguishing information
* Inter-class visual similarity

---

### Failure Case 3 — Blur and Motion

Images containing motion blur can produce inaccurate localization or missed detections.

**Likely causes:**

* Loss of fine-grained features
* Blurred boundaries
* Reduced confidence in object localization

---

### Failure Case 4 — Complex Background / Lighting

Underwater scenes can contain reflections, vegetation, rocks, shadows, and uneven illumination.

**Likely causes:**

* Background objects resembling fish features
* Low contrast
* Color distortion
* Uneven lighting

---

### Failure Case 5 — Crowded Scenes

When multiple fish overlap or appear very close together, individual fish can be difficult to separate.

**Likely causes:**

* Heavy occlusion
* Overlapping bounding boxes
* Similar appearance between neighboring fish
* Difficulty determining object boundaries

---

# 🤖 Part B — Minimal Reasoning Layer

The second API endpoint accepts a natural-language question together with an image.

Example:

```text
"How many ClownFish are in this image?"
```

The reasoning layer follows a simple hand-written decision process.

```text
Question
   │
   ▼
Intent Routing
   │
   ├── Image-related?
   │       │
   │       ├── No → Direct response
   │       │
   │       └── Yes
   │             │
   │             ▼
   │        Call RT-DETR
   │             │
   │             ▼
   │       Structured Results
   │             │
   │             ▼
   │        Apply Reasoning
   │             │
   │             ▼
   │      Confidence Guardrail
   │             │
   │       ┌─────┴─────┐
   │       ▼           ▼
   │    Enough      Insufficient
   │   information  information
   │       │           │
   │       ▼           ▼
   │     Answer     "Insufficient
   │                 information"
```

### No Agentic Frameworks

The reasoning layer does **not** use:

* LangChain
* LangGraph
* CrewAI
* AutoGen
* Agent frameworks

Instead, the routing and reasoning logic is implemented directly in Python.

---

#  Confidence Guardrail

The system is designed not to blindly answer when the detector does not provide sufficient evidence.

For example, if the question is:

```text
"Is the smallest fish in the image a YellowTang?"
```

but the detector produces uncertain or incomplete detections, the system should not invent an answer.

Instead:

```text
Insufficient information to answer confidently from the detected objects.
```

This behavior is intentional.

The goal is to make the reasoning layer **honest about model uncertainty** rather than generating unsupported answers.

---

#  API

The application is implemented using **FastAPI**.

## Endpoint 1 — Object Detection

```http
POST /detect
```

Accepts an image and returns detected objects.

### Request

```text
multipart/form-data
image=<image file>
```

### Example Response

```json
{
  "detections": [
    {
      "class": "ClownFish",
      "confidence": 0.94,
      "bbox": [120, 85, 310, 240]
    },
    {
      "class": "BlueTang",
      "confidence": 0.87,
      "bbox": [350, 120, 510, 290]
    }
  ]
}
```

---

# Endpoint 2 — Question Answering / Reasoning

```http
POST /ask
```

Accepts:

* Image
* Natural-language question

### Example Request

```text
image=<image file>
question=How many fish are in this image?
```

### Example Response

```json
{
  "question": "How many fish are in this image?",
  "answer": "There are 2 detected fish in the image.",
  "detections_used": 2,
  "confidence_status": "sufficient"
}
```

---

#  Example Questions

The reasoning endpoint can answer questions such as:

```text
How many fish are in this image?

How many ClownFish are present?

What is the most common fish species?

Is there a YellowTang?

Which fish species were detected?
```

For questions that cannot be reliably answered from the detector's structured output, the system returns an explicit insufficient-information response.

---

# Project Structure

```text
fish-rtdetr-rap/
│
├── app/
│   ├── main.py
│   ├── detector.py
│   └── reasoning.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── weights/
│   └── best.pt
│
├── data/
│   └── data.yaml
│
├── examples/
│   └── sample.jpg
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

#  Running Locally

## 1. Clone the Repository

```bash
git clone <https://github.com/sandhoshivany/fish-rtdetr-api>
cd fish-rtdetr-rap
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# Training From Scratch / Reproducing the Run

To reproduce the training process:

```bash
python training/train.py
```

The training script contains:

* Dataset configuration
* RT-DETR model initialization
* Image size
* Epoch count
* Batch configuration
* Output directory
* Model checkpoint configuration

After training, the best checkpoint is saved and used by the FastAPI application.

---

#  Evaluation

Run:

```bash
python training/evaluate.py
```

The evaluation script calculates detection metrics on the test dataset.

The evaluation output includes:

```text
Precision
Recall
mAP@0.50
mAP@0.50:0.95
```

---

#  Docker

The project also supports containerized deployment.

Build:

```bash
docker build -t fish-rtdetr-api .
```

Run:

```bash
docker run -p 8000:8000 fish-rtdetr-api
```

Then open:

```text
http://localhost:8000/docs
```

---

#  Deployment

The FastAPI application has been prepared for remote deployment.

The deployed API allows reviewers to test the trained RT-DETR model without requiring the complete training environment.

### API Base URL

```text
https://fish-rtdetr-api.onrender.com/
```

### Swagger

```text
https://fish-rtdetr-api.onrender.com/docs
```.


 Reproducibility

The project documents:

* Python version
* Required packages
* Dataset structure
* Dataset configuration
* Model architecture
* Training parameters
* Image size
* Number of epochs
* Hardware used
* Training duration
* Evaluation procedure
* API startup commands

This allows another developer to reproduce the training and inference environment.



Technology Stack

| Component        | Technology        |
| ---------------- | ----------------- |
| Detection Model  | RT-DETR           |
| ML Framework     | Ultralytics       |
| Programming      | Python            |
| API              | FastAPI           |
| API Server       | Uvicorn           |
| Computer Vision  | OpenCV            |
| Containerization | Docker            |
| Documentation    | Swagger / OpenAPI |
| Version Control  | Git + GitHub      |


  Constraints Compliance

| RAP Requirement                     | Implementation                               |
| ----------------------------------- | -------------------------------------------- |
| RT-DETR                             | ✅ RT-DETR fine-tuning                        |
| Custom domain                       | ✅ Fish species detection                     |
| Non-COCO classes                    | ✅ Domain-specific fish classes               |
| Own dataset selection               | ✅ Custom fish dataset                        |
| Training code                       | ✅ Included                                   |
| Evaluation code                     | ✅ Included                                   |
| FastAPI detection endpoint          | ✅ Implemented                                |
| Natural-language reasoning endpoint | ✅ Implemented                                |
| Intent routing                      | ✅ Hand-written logic                         |
| Structured reasoning                | ✅ Detection output → reasoning               |
| Confidence guardrail                | ✅ Explicit insufficient-information handling |
| Agentic frameworks                  | ❌ Not used                                   |
| AutoML                              | ❌ Not used                                   |
| Reproducibility                     | ✅ Documented                                 |
| Failure analysis                    | ✅ Included                                   |
| Docker                              | ✅ Supported                                  |

  Limitations

Despite strong performance on the held-out test set, the model has known limitations.

Performance can degrade when:

* Fish are extremely small.
* Objects are heavily occluded.
* Images contain significant blur.
* Lighting differs significantly from the training data.
* Species have highly similar visual characteristics.
* Multiple fish overlap heavily.
* Images differ substantially from the training distribution.

The model's self-reported metrics should therefore not be interpreted as guaranteed performance on unseen data.


  Future Improvements

Potential improvements include:

* Increasing dataset diversity
* Adding more difficult underwater scenes
* Improving class balance
* Hard-negative mining
* More targeted augmentation
* Per-class error analysis
* Confidence calibration
* Better uncertainty handling
* Model optimization for lower-latency inference
* More robust natural-language intent routing


 Author

**Sandhoshivany G N**

B.Tech — Artificial Intelligence & Data Science
Sri Venkateswara College of Engineering

Links

* GitHub: `https://github.com/sandhoshivany`
* Portfolio: `https://portfolio-53a9.vercel.app/`
* LinkedIn: `sandhoshivany-g-n-0381b2333`


  Key Takeaway

This project demonstrates an end-to-end applied ML workflow:


Dataset Selection
       ↓
Data Preparation
       ↓
RT-DETR Fine-Tuning
       ↓
Evaluation
       ↓
Failure Analysis
       ↓
Model Serialization
       ↓
FastAPI Deployment
       ↓
Structured Detection Output
       ↓
Natural-Language Reasoning
       ↓
Confidence-Aware Answer


The focus is not only on achieving a high detection score, but on understanding the model's behavior, limitations, reproducibility, and integration into a usable ML API.
