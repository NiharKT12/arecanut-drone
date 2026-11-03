# ArecaVision — Arecanut Yellow Leaf Disease Detection

ArecaVision is a computer-vision system designed to detect arecanut palms in aerial plantation images and classify them as **Healthy** or **YLD (Yellow Leaf Disease)** using YOLO26s.

The application uses Flask for the web interface and ONNX Runtime for model inference.

> **Important:** This is a screening prototype, not a laboratory diagnostic tool. Predictions should be verified through expert field inspection.

## Features

- Detects individual arecanut palms in aerial images.
- Classifies palms as Healthy or YLD.
- Displays bounding boxes, class labels and confidence scores.
- Calculates detected palm counts.
- Estimates the proportion of palms affected by YLD.
- Provides a Flask-based web interface.
- Uses ONNX Runtime for lightweight inference.
- Includes dataset augmentation and model-training utilities.

## Model Comparison

Four YOLO models were evaluated: YOLO11n, YOLO11s, YOLO12s and YOLO26s.

| Metric | YOLO11n | YOLO11s | YOLO12s | YOLO26s |
|---|---:|---:|---:|---:|
| Parameters (M) | 2.59 | 9.43 | 9.25 | 9.95 |
| GFLOPs | 6.5 | 21.7 | 23.5 | 22.7 |
| Precision | 0.744 | 0.566 | 0.454 | 0.729 |
| Recall | 0.312 | 0.562 | 0.581 | 0.631 |
| Macro F1 | 0.274 | 0.529 | 0.495 | **0.653** |
| mAP50 | 0.283 | 0.509 | 0.503 | **0.639** |
| mAP50–95 | 0.118 | 0.236 | 0.254 | **0.303** |
| Count Error / Image | 2.08 | 1.25 | 1.77 | **0.50** |
| YLD Percentage Error / Image | 45.8 | 35.1 | 34.8 | **30.1** |
| Pooled YLD Percentage Error | 33.3 | 19.0 | 16.9 | 18.3 |

YOLO26s achieved the best overall Macro F1, mAP50 and mAP50–95 among the evaluated models. YOLO12s achieved better YLD AP50 and pooled YLD percentage error.

The evaluation used 40 held-out test images, with one training run per model.

### YOLO26s Class-wise Performance

| Class | AP50 | F1-score |
|---|---:|---:|
| Healthy | 0.491 | 0.574 |
| YLD | 0.788 | 0.731 |

### Confidence Threshold

The selected YOLO26s operating confidence threshold was **0.148**, determined using validation data.

## System Workflow

1. User uploads an aerial plantation image.
2. The image is resized and converted to RGB.
3. The image is letterboxed to 640 × 640.
4. ONNX Runtime performs model inference.
5. Predicted bounding boxes are mapped to original image dimensions.
6. Confidence filtering and Non-Maximum Suppression are applied.
7. The interface displays detections and confidence scores.
8. The system estimates the YLD proportion.

## Technology Stack

- Python
- YOLO26s
- Flask
- ONNX Runtime
- OpenCV
- HTML
- CSS
- JavaScript

## Project Structure

```text
arecanut-drone/
├── api/
│   └── index.py
├── templates/
│   └── index.html
├── scripts/
│   ├── augment_dataset.py
│   └── build_colab_notebook.py
├── notebooks/
│   └── train_yld_colab.ipynb
├── best.onnx
├── requirements.txt
├── requirements-train.txt
└── vercel.json
