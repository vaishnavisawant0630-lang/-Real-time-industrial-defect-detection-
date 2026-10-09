from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/best.pt"
TEST_IMAGES = "data/processed/images/test"

model = YOLO(MODEL_PATH)

print("Loading model.....")
print(f"Model: {MODEL_PATH}")
print(f"Input: {TEST_IMAGES}")

results = model.predict(
    source=TEST_IMAGES,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    device="cpu",
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs/detect",
    name="test_predictions",
    exist_ok=True
)

print("\n========================================")
print("TEST IMAGE INFERENCE COMPLETE")
print("========================================")
print("Results saved to:")
print("runs/detect/test_predictions")