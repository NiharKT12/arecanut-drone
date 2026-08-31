from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import base64
import os
import tempfile

# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

# Vercel request limit is 4.5 MB
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "best_model.pt"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Classes:", model.names)

print("=" * 60)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

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
    # READ IMAGE DIRECTLY
    # ========================================================

    image_bytes = file.read()

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


    # ========================================================
    # RUN YOLO
    # ========================================================

    try:

        results = model.predict(

            source=image,

            conf=0.25,

            imgsz=640,

            device="cpu",

            verbose=False
        )

    except Exception as e:

        print("YOLO ERROR:")
        print(e)

        return render_template(
            "index.html",
            error=f"Model inference failed: {e}"
        )


    result = results[0]


    # ========================================================
    # DETECTION COUNTS
    # ========================================================

    detections = []

    healthy_count = 0
    yld_count = 0


    # ========================================================
    # DRAW BOXES
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            # ------------------------------------------------
            # Coordinates
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )


            class_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf[0].item()
            )


            # ------------------------------------------------
            # Count
            # ------------------------------------------------

            if class_name.lower() == "healthy":

                healthy_count += 1

            elif class_name.lower() == "yld":

                yld_count += 1


            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            detections.append({

                "class": class_name,

                "confidence": round(
                    confidence * 100,
                    2
                )

            })


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                3

            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(

                image,

                label,

                (x1, max(y1 - 10, 25)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


    # ========================================================
    # YLD PERCENTAGE
    # ========================================================

    total_plants = (
        healthy_count +
        yld_count
    )


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

    # Compress to JPEG
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


    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


    result_image = (
        "data:image/jpeg;base64,"
        + image_base64
    )


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
