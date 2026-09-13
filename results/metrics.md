
# Model Evaluation

## Overview

The trained RT-DETR model was evaluated on the test dataset to measure its object detection and localization performance.

The test set contains **700 images**, **1,134 fish instances**, and **13 fish classes**.

## Evaluation Results

| Metric         |     Result |
| -------------- | ---------: |
| Precision      | **82.08%** |
| Recall         | **85.44%** |
| mAP@0.50       | **87.85%** |
| mAP@0.50:0.95  | **69.22%** |
| Test Images    |    **700** |
| Test Instances |  **1,134** |
| Classes        |     **13** |

## Metric Explanation

### Precision — 82.08%

Precision represents the percentage of predicted detections that were correct.

A precision of **82.08%** indicates that the majority of fish detections produced by the model were correct, while some predictions were false positives.

### Recall — 85.44%

Recall represents the percentage of actual fish instances that were successfully detected.

A recall of **85.44%** indicates that the model successfully detected most of the fish present in the test dataset, although some fish were missed.

### mAP@0.50 — 87.85%

mAP@0.50 measures the model's detection performance using an Intersection over Union (IoU) threshold of 0.50.

The model achieved **87.85% mAP@0.50**, indicating strong overall object detection performance when a predicted bounding box has sufficient overlap with the ground-truth bounding box.

### mAP@0.50:0.95 — 69.22%

mAP@0.50:0.95 is a stricter evaluation metric that averages performance across IoU thresholds from **0.50 to 0.95**.

The model achieved **69.22%**, indicating that precise bounding-box localization becomes more challenging at stricter IoU thresholds.

## Test Dataset

* **Test Images:** 700
* **Test Instances:** 1,134
* **Number of Classes:** 13

The test dataset was used only for final evaluation and was not used to train the model.

## Summary

The results show that the RT-DETR model provides strong fish detection performance, achieving **87.85% mAP@0.50**, **82.08% precision**, and **85.44% recall**. The lower **69.22% mAP@0.50:0.95** indicates that improving precise localization and handling difficult visual conditions remain areas for future improvement.
