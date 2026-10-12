
from pathlib import Path
import json
from datetime import datetime

from ultralytics import YOLO

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best.pt"
DATA_YAML = ROOT_DIR / "data" / "dataset.yaml"
OUTPUT_DIR = ROOT_DIR / "reports"

# Evaluation configuration
IMAGE_SIZE = 640
BATCH_SIZE = 4
DEVICE = "cpu"
SPLIT = "test"  # Use "test" to evaluate the test split


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("INDUSTRIAL DEFECT DETECTION - MODEL METRICS")
    print("=" * 55)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Split: {SPLIT}")
    print(f"Device: {DEVICE}")
    print()

    model = YOLO(str(MODEL_PATH))

    # Ultralytics calculates detection metrics against
    # the ground-truth labels in the selected dataset split.
    results = model.val(
        data=str(DATA_YAML),
        split=SPLIT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        verbose=True,
        project=str(ROOT_DIR / "runs" / "detect"),
        name=f"metrics_{SPLIT}",
        exist_ok=True,
    )

    # Overall detection metrics
    box_metrics = results.box

    metrics = {
        "evaluation_date": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "dataset": str(DATA_YAML.relative_to(ROOT_DIR)),
        "split": SPLIT,
        "image_size": IMAGE_SIZE,
        "device": DEVICE,
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "mAP50": float(box_metrics.map50),
        "mAP50_95": float(box_metrics.map),
    }

    # Per-class AP values
    metrics["per_class_mAP50"] = {
        model.names[class_id]: float(ap)
        for class_id, ap in enumerate(box_metrics.ap50)
    }

    metrics["per_class_mAP50_95"] = {
        model.names[class_id]: float(ap)
        for class_id, ap in enumerate(box_metrics.ap)
    }

    # Save metrics as JSON
    output_file = OUTPUT_DIR / f"metrics_{SPLIT}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("OVERALL EVALUATION METRICS")
    print("=" * 55)
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"mAP@50    : {metrics['mAP50']:.4f}")
    print(f"mAP@50-95 : {metrics['mAP50_95']:.4f}")

    print("\nPER-CLASS mAP@50")
    for class_name, ap in metrics["per_class_mAP50"].items():
        print(f"{class_name:20s}: {ap:.4f}")

    print("\nPER-CLASS mAP@50-95")
    for class_name, ap in metrics["per_class_mAP50_95"].items():
        print(f"{class_name:20s}: {ap:.4f}")

    print(f"\nMetrics saved to: {output_file}")
    print(f"Evaluation plots: {results.save_dir}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
