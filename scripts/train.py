from pathlib import Path
from ultralytics import YOLO

# CONFIGURATION

DATASET = "data/dataset.yaml"
MODEL = "yolov8n.pt"

IMAGE_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16

PROJECT = "runs/detect"
RUN_NAME = "neu_yolov8n_baseline"


# CHECK DATASET

if not Path(DATASET).exists():
    raise FileNotFoundError(
        f"Dataset configuration not found: {DATASET}"
    )

# LOAD MODEL

print("=" * 60)
print("YOLOv8 BASELINE TRAINING")
print("=" * 60)

print(f"Model       : {MODEL}")
print(f"Dataset     : {DATASET}")
print(f"Image size  : {IMAGE_SIZE}")
print(f"Epochs      : {EPOCHS}")
print(f"Batch size  : {BATCH_SIZE}")

model = YOLO(MODEL)

# TRAIN

results = model.train(
    data=DATASET,

    # Training configuration
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    # Reproducibility
    seed=42,

    # Validation
    val=True,

    # Output
    project=PROJECT,
    name=RUN_NAME,

    # Save checkpoints
    save=True,
    save_period=10,

    # Workers
    workers=2,

    # Hardware
    device="cpu",

    # Verbose output
    verbose=True,
)


# COMPLETE

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Results saved to: "
    f"{PROJECT}/{RUN_NAME}"
)