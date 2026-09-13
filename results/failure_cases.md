# Failure Case Analysis

## Failure Case 1 — Small / Distant Fish

Observation:
Small fish objects were missed or detected with low confidence.

Likely Cause:
Insufficient visual features due to small object size.

Impact:
Lower recall for small objects.

---

## Failure Case 2 — Similar Appearance Between Species

Observation:
Visually similar fish species were occasionally confused.

Likely Cause:
Similar color, shape and body patterns between classes.

Impact:
False positives / incorrect class predictions.

---

## Failure Case 3 — Occlusion and Overlapping Fish

Observation:
Fish partially hidden behind other fish were sometimes missed.

Likely Cause:
Limited visible features caused incomplete object representation.

Impact:
Missed detections and inaccurate bounding boxes.

---

## Failure Case 4 — Background / Lighting Variation

Observation:
Performance decreased in images with complex backgrounds or lighting.

Likely Cause:
The visual appearance of the fish differs from the training distribution.

Impact:
Reduced confidence and occasional incorrect predictions.

---

## Failure Case 5 — High-Confidence Misclassification

Observation:
A visually similar fish may receive a high confidence score for the wrong class.

Likely Cause:
Confidence measures how strongly the model supports its predicted class; it does not guarantee that the prediction is factually correct.

Impact:
High-confidence false positives can pass a simple confidence threshold.

Possible Improvement:
Use better training data, hard-negative examples, class-specific analysis and a verification/reasoning layer.
