from pathlib import Path
from ultralytics import YOLO
import cv2
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"
IMAGE_DIR = ROOT / "data" / "processed" / "images" / "test"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
WARMUP_IMAGES = 3

model = YOLO(str(MODEL_PATH))

extensions = {".jpg", ".jpeg", ".png", ".bmo", ".webp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)


if not images:
    raise FileNotFoundError(f"No test images found: {IMAGE_DIR}")

print(f"Model: {MODEL_PATH}")
print(f"Images found: {len(images)}")


for image_path in images[:min(WARMUP_IMAGES, len(images))]:
    image = cv2.imread(str(image_path))
    if model is None:
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

latencies = []

for index, image_path in enumerate(images):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path.name}")
        continue

    start = time.perf_counter()

    model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if index >= WARMUP_IMAGES:
        latencies.append(elapsed_ms)

    print(
        f"[{index + 1}/{len(images)}] "
        f"{image_path.name}: {elapsed_ms:.2f} ms"
    )

if not latencies:
    raise RuntimeError("Not enough readable image to benchmark.")

average_ms = sum(latencies) / len(latencies)
fps = 1000 / average_ms

print("\n========== BENCHMARK RESULTS ==========")
print(f"Images found: {len(images)}")
print(f"Timed images: {len(latencies)}")
print(f"Average latency: {average_ms:.2f} ms/image")
print(f"Approximate inference FPS: {fps:.2f}")
print("=======================================")


    
