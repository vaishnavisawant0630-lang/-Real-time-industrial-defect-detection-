
from pathlib import Path

from ultralytics import YOLO
from preprocessing import preprocess_frame

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

EXPECTED_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load the trained model
model = YOLO(str(MODEL_PATH))

print(f"Loaded model: {MODEL_PATH}")
print(f"Model class names: {model.names}")


def detect_frame(frame, confidence=CONFIDENCE):
    """Detect defects in an OpenCV BGR image or video frame."""
    frame = preprocess_frame(frame)

    result = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        device="cpu",
        verbose=False,
    )[0]

    return result


def get_detections(result):
    """Return each detection's class, confidence, and coordinates."""
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = result.names[class_id]

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(score, 4),
            "box_xyxy": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            ],
        })

    return detections
