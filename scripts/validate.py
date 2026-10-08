from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
DATASET_PATH = "data/dataset.yaml"

model = YOLO(MODEL_PATH)

results = model.val(
    data=DATASET_PATH,
    imgsz=640,
    batch=8,
    device="cpu",
    plots=True
)


print("\n========== VALIDATION RESULTS ==========")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")
print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")