from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import base64


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "best_model.pt"

TEMPLATE_DIR = BASE_DIR / "templates"


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not TEMPLATE_DIR.exists():

    raise FileNotFoundError(
        f"Templates folder not found:\n{TEMPLATE_DIR}"
    )


if not (TEMPLATE_DIR / "index.html").exists():

    raise FileNotFoundError(
        f"index.html not found:\n"
        f"{TEMPLATE_DIR / 'index.html'}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print()
print("Model loaded successfully.")
print("Classes:", model.names)

print("=" * 60)


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

# YOLO itself runs at a very low threshold so that we can
# inspect both Healthy and YLD predictions.
YOLO_CONFIDENCE = 0.01

# Class-specific thresholds.
#
# Healthy predictions currently appear to have lower
# confidence, so we start low and tune this later.
HEALTHY_THRESHOLD = 0.01

# YLD appears stronger, so we use a higher threshold.
YLD_THRESHOLD = 0.10

# NMS IoU
IOU_THRESHOLD = 0.50


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template(
            "index.html"
        )


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="No image uploaded."
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image_bytes = file.read()

    print()
    print("=" * 60)
    print("IMAGE INFORMATION")
    print("=" * 60)

    print("Filename:", file.filename)
    print("Size:", len(image_bytes), "bytes")

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        return render_template(
            "index.html",
            error="Could not read the uploaded image."
        )


    print(
        "Resolution:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    print("=" * 60)


    # ========================================================
    # RUN YOLO
    # ========================================================

    print()
    print("=" * 60)
    print("RUNNING YOLO INFERENCE")
    print("=" * 60)

    try:

        results = model.predict(

            source=image,

            # Keep this LOW so Healthy detections are not
            # removed before we can inspect them.
            conf=YOLO_CONFIDENCE,

            imgsz=640,

            device="cpu",

            iou=IOU_THRESHOLD,

            max_det=100,

            verbose=False
        )

    except Exception as e:

        print()
        print("YOLO ERROR:")
        print(e)

        return render_template(
            "index.html",
            error=f"Model inference failed: {e}"
        )


    result = results[0]


    # ========================================================
    # DEBUG RAW DETECTIONS
    # ========================================================

    print()
    print("=" * 60)
    print("RAW YOLO DETECTIONS")
    print("=" * 60)

    print("Model classes:", model.names)

    if result.boxes is None:

        print("No boxes returned.")

    else:

        print(
            "Raw boxes:",
            len(result.boxes)
        )

        if len(result.boxes) > 0:

            healthy_confidences = []
            yld_confidences = []

            for i, box in enumerate(result.boxes):

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = str(
                    model.names[class_id]
                )

                print(
                    f"{i + 1}. "
                    f"{class_name} "
                    f"{confidence:.4f}"
                )

                if class_name.lower().strip() == "healthy":

                    healthy_confidences.append(
                        confidence
                    )

                elif class_name.lower().strip() == "yld":

                    yld_confidences.append(
                        confidence
                    )


            print()
            print("-" * 60)

            if healthy_confidences:

                print(
                    "Healthy detections:",
                    len(healthy_confidences)
                )

                print(
                    "Healthy minimum:",
                    f"{min(healthy_confidences):.4f}"
                )

                print(
                    "Healthy maximum:",
                    f"{max(healthy_confidences):.4f}"
                )

                print(
                    "Healthy average:",
                    f"{np.mean(healthy_confidences):.4f}"
                )

            else:

                print(
                    "NO HEALTHY DETECTIONS "
                    "FROM MODEL"
                )


            print()

            if yld_confidences:

                print(
                    "YLD detections:",
                    len(yld_confidences)
                )

                print(
                    "YLD minimum:",
                    f"{min(yld_confidences):.4f}"
                )

                print(
                    "YLD maximum:",
                    f"{max(yld_confidences):.4f}"
                )

                print(
                    "YLD average:",
                    f"{np.mean(yld_confidences):.4f}"
                )

            else:

                print(
                    "NO YLD DETECTIONS "
                    "FROM MODEL"
                )


    print("=" * 60)


    # ========================================================
    # DETECTIONS AFTER CLASS-SPECIFIC FILTERING
    # ========================================================

    detections = []

    healthy_count = 0

    yld_count = 0


    # ========================================================
    # PROCESS BOXES
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )

            class_name = str(
                model.names[class_id]
            )

            class_name_lower = (
                class_name.lower().strip()
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0].item()
            )


            # ------------------------------------------------
            # CLASS-SPECIFIC THRESHOLD
            # ------------------------------------------------

            if class_name_lower == "healthy":

                if confidence < HEALTHY_THRESHOLD:

                    continue


            elif class_name_lower == "yld":

                if confidence < YLD_THRESHOLD:

                    continue


            else:

                # Ignore unknown classes
                continue


            # ------------------------------------------------
            # COORDINATES
            # ------------------------------------------------

            x1, y1, x2, y2 = (

                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)

            )


            # ------------------------------------------------
            # COUNT
            # ------------------------------------------------

            if class_name_lower == "healthy":

                healthy_count += 1

            elif class_name_lower == "yld":

                yld_count += 1


            # ------------------------------------------------
            # STORE DETECTION
            # ------------------------------------------------

            detections.append({

                "class": class_name,

                "confidence": round(
                    confidence * 100,
                    2
                )

            })


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                3

            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (

                f"{class_name} "
                f"{confidence:.2f}"

            )


            cv2.putText(

                image,

                label,

                (
                    x1,
                    max(y1 - 10, 25)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


    # ========================================================
    # TOTAL PLANTS
    # ========================================================

    total_plants = (

        healthy_count +
        yld_count

    )


    # ========================================================
    # YLD PERCENTAGE
    # ========================================================

    if total_plants > 0:

        yld_percentage = (

            yld_count /
            total_plants

        ) * 100

    else:

        yld_percentage = 0


    # ========================================================
    # ENCODE RESULT IMAGE
    # ========================================================

    success, encoded_image = cv2.imencode(

        ".jpg",

        image,

        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]

    )


    if not success:

        return render_template(
            "index.html",
            error="Could not encode result image."
        )


    # ========================================================
    # BASE64
    # ========================================================

    image_base64 = base64.b64encode(

        encoded_image.tobytes()

    ).decode("utf-8")


    result_image = (

        "data:image/jpeg;base64,"
        + image_base64

    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL DETECTION RESULTS")
    print("=" * 60)

    print(
        "Total plants :",
        total_plants
    )

    print(
        "Healthy      :",
        healthy_count
    )

    print(
        "YLD          :",
        yld_count
    )

    print(
        "YLD rate     :",
        round(yld_percentage, 2),
        "%"
    )

    print("=" * 60)


    # ========================================================
    # RETURN PAGE
    # ========================================================

    return render_template(

        "index.html",

        result_image=result_image,

        detections=detections,

        total_detections=total_plants,

        healthy_count=healthy_count,

        yld_count=yld_count,

        yld_percentage=round(
            yld_percentage,
            2
        )

    )


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return render_template(

        "index.html",

        error=(
            "Image is too large. "
            "Please upload an image smaller than 4 MB."
        )

    ), 413


# ============================================================
# RUN LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Starting Flask server...")
    print("=" * 60)

    print()
    print("Open:")
    print("http://127.0.0.1:5000")

    print()
    print("Press CTRL+C to stop.")
    print()

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )