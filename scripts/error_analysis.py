
from pathlib import Path
from ultralytics import YOLO
import shutil

MODEL_PATH = Path("models/best.pt")
TEST_DIR = Path("data/processed/images/test")
OUTPUT_DIR = Path("runs/detect/error_analysis")

CONFIDENCE = 0.25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NO_DETECTION_DIR = OUTPUT_DIR / "no_detections"
NO_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

no_detection_images = []
low_confidence_detections = []
class_counts = {}

image_paths = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(TEST_DIR.glob(pattern))

for image_path in sorted(image_paths):
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        no_detection_images.append(image_path)
        shutil.copy2(image_path, NO_DETECTION_DIR / image_path.name)
        continue

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = model.names[class_id]

        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if confidence < 0.50:
            low_confidence_detections.append(
                (image_path.name, class_name, confidence)
            )

    # Save an image with predicted boxes and class labels
    annotated = result.plot()
    output_path = OUTPUT_DIR / image_path.name

    import cv2
    cv2.imwrite(str(output_path), annotated)

print("\n" + "=" * 45)
print("ERROR ANALYSIS")
print("=" * 45)

print("\nDetections by class:")
for name, count in sorted(class_counts.items()):
    print(f"{name}: {count}")

print("\nLow-confidence detections (below 0.50):")
for filename, class_name, confidence in low_confidence_detections:
    print(f"{filename} | {class_name} | {confidence:.2f}")

print(f"\nImages with no detections: {len(no_detection_images)}")
for image_path in no_detection_images:
    print(image_path)

print(f"\nNo-detection images copied to: {NO_DETECTION_DIR}")
print(f"Annotated predictions saved to: {OUTPUT_DIR}")
print("\nAnalysis complete.")

