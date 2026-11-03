ArecaVision — Arecanut Yellow Leaf Disease Detection

ArecaVision is a computer-vision system for detecting arecanut palms in aerial plantation imagery and classifying each detected palm as healthy or yld (Yellow Leaf Disease). The application displays detections and estimates the proportion of detected palms associated with YLD.

YOLO26s is the selected final model based on the project's four-model comparison against YOLO11n, YOLO11s and YOLO12s.

Important: This is a screening prototype, not a laboratory diagnostic tool. A predicted yld label is not a confirmed diagnosis. The estimated YLD percentage should not be treated as an exact agronomic measurement.

Features

Detects individual arecanut palms in RGB aerial images.

Classifies detections as healthy or yld.

Displays bounding boxes, class labels and confidence scores.

Summarizes detections and estimates the YLD proportion.

Provides a Flask-based web interface.

Uses ONNX Runtime for lightweight inference without loading the full PyTorch training stack.

Includes dataset augmentation and model-training utilities.

Final model: YOLO26s

The supplied comparison evaluated YOLO11n, YOLO11s, YOLO12s and YOLO26s on the same held-out test split of 40 images. Each model was trained once. The operating confidence was selected on the validation split to maximize mean F1 across the two classes; final test metrics were measured at that confidence. mAP was computed at confidence 0.001.

Overall test results

Metric

YOLO11n

YOLO11s

YOLO12s

YOLO26s

Parameters (M)

2.59

9.43

9.25

9.95

GFLOPs

6.5

21.7

23.5

22.7

Operating confidence

0.065

0.216

0.096

0.148

Precision

0.744

0.566

0.454

0.729

Recall

0.312

0.562

0.581

0.631

Macro F1

0.274

0.529

0.495

0.653

mAP50

0.283

0.509

0.503

0.639

mAP50–95

0.118

0.236

0.254

0.303

Count error / image (trees)

2.08

1.25

1.77

0.50

YLD percentage error / image (percentage points)

45.8

35.1

34.8

30.1

Pooled YLD percentage error (percentage points)

33.3

19.0

16.9

18.3

Higher is better for precision, recall, F1 and mAP. Lower is better for count and YLD-percentage errors.

YOLO26s class-wise results

Class

AP50

F1

healthy

0.491

0.574

yld

0.788

0.731

The comparison summary provides per-class AP50 and F1. Per-class precision and recall are not included in that summary, so they are not inferred here.

Interpretation

YOLO26s achieved the highest macro F1 (0.653), mAP50 (0.639) and mAP50–95 (0.303) among the four evaluated models.

YOLO26s achieved the lowest mean count error: 0.50 trees per image.

YOLO12s achieved the highest YLD AP50 (0.838) and the lowest pooled YLD-percentage error (16.9 percentage points). YOLO26s is the strongest overall model in this comparison, but it is not best on every individual metric.

YOLO26s has 9.95 million parameters and 22.7 GFLOPs, similar in scale to YOLO11s and YOLO12s.

YOLO26s still has a mean per-image YLD-percentage error of 30.1 percentage points. Detection performance does not guarantee an accurate plot-level disease percentage.

Confidence threshold

The selected YOLO26s operating confidence in the comparison was 0.148, chosen using validation data. This is an experiment-specific operating point, not a universal threshold. Confirm it against the final exported ONNX model and the application's post-processing pipeline. Do not tune the threshold using the held-out test images.

How the system works

The user selects an aerial plantation image in the web interface.

The image is resized when required and converted to RGB.

The image is letterboxed to the model input size of 640 × 640.

ONNX Runtime performs model inference.

Predicted boxes are mapped back to the original image dimensions.

Confidence filtering and non-maximum suppression are applied.

The interface displays bounding boxes, labels, confidence scores and count summaries.

The application estimates the YLD proportion from the detections.

Detected counts and estimated composition are different quantities. The number of boxes drawn is the number of palms detected at the selected operating point. A percentage derived from detections can be biased when the model detects the two classes with different reliability. Any class-wise count correction must be validated on independent data before field use.

Project structure

arecanut-drone/
├── api/
│   └── index.py                       # Flask application and ONNX inference
├── templates/
│   └── index.html                     # Image upload interface
├── scripts/
│   ├── augment_dataset.py             # Dataset augmentation
│   └── build_colab_notebook.py        # Generates training notebook
├── notebooks/
│   └── train_yld_colab.ipynb          # Model training workflow
├── best.onnx                          # ONNX model used by the application
├── requirements.txt                   # Runtime dependencies
├── requirements-train.txt             # Training dependencies
└── vercel.json                        # Vercel routing configuration, if used

Model-artifact check: best.onnx must be the exported YOLO26s checkpoint for the deployed application to be described as running YOLO26s. If the file has not yet been replaced with the YOLO26s export, export and test the correct checkpoint before deployment.

Datasets, training runs and PyTorch checkpoints may be stored separately and excluded from Git because of their size. Retain the dataset version, checkpoint and export details for reproducibility.

Run locally

1. Clone the repository

git clone https://github.com/NiharKT12/arecanut-drone.git
cd arecanut-drone

2. Create and activate a virtual environment

python -m venv .venv

Windows:

.venv\\Scripts\\activate

macOS / Linux:

source .venv/bin/activate

3. Install runtime dependencies

pip install -r requirements.txt

4. Start the application

python api/index.py

Open the local address printed by Flask (typically http://127.0.0.1:5000).

The application requires a compatible ONNX model at the path configured in api/index.py. Verify that this is the YOLO26s export before the final demonstration.

Dataset and augmentation

The project uses two object-detection classes:

healthy

yld

The annotation workflow uses YOLO polygon-format labels:

class_id x1 y1 x2 y2 x3 y3 ...

The augmentation script reads original images and polygon annotations, applies image transformations, and transforms the corresponding annotations. Image and polygon geometry must remain aligned after every geometric transformation.

Example commands:

# Generate an augmented dataset
python scripts/augment_dataset.py --dst dataset_aug

# Generate class-balanced training augmentation
python scripts/augment_dataset.py --dst dataset_bal --balance

Balancing should be applied to training data only. Validation and test sets should retain their original distribution. Augmentation can increase exposure to minority-class examples, but it does not create new independent plantation scenes or replace collecting more real images.

Training YOLO26s

The project includes a Colab-oriented training workflow. Use a GPU runtime and the final dataset version intended for the experiment.

Prepare and validate image-label pairs.

Confirm the class mapping (healthy, yld) and dataset configuration.

Split data into training, validation and test sets. Keep related images from the same scene or flight in the same split wherever possible.

Apply augmentation to training images only.

Train YOLO26s and retain the best checkpoint.

Evaluate the checkpoint on the validation set.

Select the operating confidence using validation data.

Export the selected checkpoint to ONNX.

Compare ONNX inference with the original model on fixed images.

Evaluate the final exported model on the held-out test split.

For reproducibility, record the exact model identifier, Ultralytics/PyTorch versions, image size, batch size, epochs, optimizer settings, random seed, dataset version, checkpoint hash, ONNX opset and inference threshold. Do not state a training setting unless it matches the actual final run.

Deployment notes

The Flask application uses ONNX Runtime for inference. Before deployment, verify:

best.onnx is the YOLO26s export.

Input tensor dimensions and preprocessing match the exported model.

RGB/BGR channel order is correct.

Letterbox padding is removed correctly when mapping boxes to original image coordinates.

Confidence filtering and NMS settings match the intended evaluation configuration.

Invalid uploads and images with no detections are handled safely.

The hosted environment can load the ONNX model within its memory, package-size and request-size limits.

If deploying to Vercel, check the current platform limits and confirm that the model and dependencies fit the function bundle and runtime constraints. Large aerial images may need client-side resizing before upload.

Limitations

Small test set: the comparative evaluation uses 40 held-out images. Results may vary on additional plantation scenes.

Single training run per model: the comparison reports one training run for each model. Repeated seeds are needed to estimate training variability.

Class imbalance: healthy and YLD examples are not equally represented. Monitor performance separately for both classes.

Scene diversity: augmented images do not replace independent real images from different plantations, seasons, lighting conditions and camera heights.

YLD percentage error: YOLO26s has a reported per-image YLD percentage error of 30.1 percentage points. This is not sufficiently precise to be treated as a definitive measurement for an individual plot.

Model export: the final ONNX artifact must be checked against the selected YOLO26s checkpoint. A model-family name alone does not identify the exact weights.

Field validation: the model has not been established as a diagnostic or treatment recommendation system. Predictions should be reviewed by a knowledgeable person.

Future work

Collect more real aerial images from independent arecanut plantations.

Improve representation of healthy palms and difficult visual conditions.

Evaluate across plantation-level splits, seasons and capture heights.

Repeat training with multiple random seeds and report mean and standard deviation.

Measure end-to-end inference latency and memory on the target deployment device.

Validate plot-level YLD estimates against manually verified counts.

Add uncertainty indicators to the estimated percentage.

Conduct field testing with intended users before operational use.

Acknowledgement

This project uses open-source computer-vision and web technologies, including the Ultralytics YOLO framework, Flask, ONNX and ONNX Runtime, along with supporting Python image-processing libraries.

License

Add the license selected for this repository before distributing or reusing the project.
