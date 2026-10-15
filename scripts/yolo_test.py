
import csv
import time
from collections import Counter
from pathlib import Path

from detect import detect_frame, get_detections
from preprocessing import load_image
from visualization import save_annotated_image

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "processed" / "images" / "test"
OUTPUT_DIR = ROOT / "runs" / "detect" / "yolo_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

images = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in EXTENSIONS
)

if not images:
    raise FileNotFoundError(f"No test images found in: {IMAGE_DIR}")

print(f"YOLO model: {ROOT / 'models' / 'best.pt'}")
print(f"Model class names: {__import__('detect').model.names}")
print(f"Test images found: {len(images)}")

rows = []
inference_times = []
total_class_counts = Counter()
warmup_count = min(3, len(images))

for index, image_path in enumerate(images):
    frame = load_image(image_path)

    start = time.perf_counter()
    result = detect_frame(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if index >= warmup_count:
        inference_times.append(elapsed_ms)

    output_path = OUTPUT_DIR / f"{image_path.stem}_prediction.jpg"
    save_annotated_image(result, output_path)

    detections = get_detections(result)

    # Initialize every expected class to zero for every image.
    class_counts = {name: 0 for name in EXPECTED_CLASSES}

    for detection in detections:
        class_name = detection["class_name"].strip().lower()
        class_name = class_name.replace("-", "_").replace(" ", "_")

        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            print(f"WARNING: Unexpected class name: {class_name}")

        total_class_counts[class_name] += 1

    row = {
        "image": image_path.name,
        "inference_ms": round(elapsed_ms, 2),
        "detection_count": len(detections),
    }

    row.update(class_counts)

    row["predictions"] = str([
        {
            "class": d["class_name"],
            "confidence": d["confidence"],
            "box_xyxy": d["box_xyxy"],
        }
        for d in detections
    ])

    row["output_image"] = output_path.name
    rows.append(row)

    print(
        f"[{index + 1}/{len(images)}] {image_path.name} | "
        f"{len(detections)} detections | "
        + ", ".join(
            f"{name}={class_counts[name]}"
            for name in EXPECTED_CLASSES
        )
    )

csv_path = OUTPUT_DIR / "yolo_results.csv"

fieldnames = [
    "image",
    "inference_ms",
    "detection_count",
    *EXPECTED_CLASSES,
    "predictions",
    "output_image",
]

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n========== YOLO TEST RESULTS ==========")
print(f"Images processed: {len(images)}")
print(f"Prediction images saved: {OUTPUT_DIR}")
print(f"CSV report: {csv_path}")

print("\nTotal detections per class:")
for class_name in EXPECTED_CLASSES:
    print(f"{class_name}: {total_class_counts[class_name]}")

unexpected = {
    name: count
    for name, count in total_class_counts.items()
    if name not in EXPECTED_CLASSES
}
if unexpected:
    print("Unexpected class names:", unexpected)

if inference_times:
    average_ms = sum(inference_times) / len(inference_times)
    fps = 1000 / average_ms if average_ms > 0 else 0

    print(f"\nTimed images after warm-up: {len(inference_times)}")
    print(f"Average inference latency: {average_ms:.2f} ms/image")
    print(f"Approximate inference FPS: {fps:.2f}")
else:
    print("Not enough images for post-warm-up timing.")

print("=======================================")
