# -Real-time-industrial-defect-detection-
# Real-Time Industrial Defect Detection System

> **Computer Vision · YOLOv8 · Industrial Quality Control · Edge AI**

A real-time computer vision system for detecting, classifying, and localizing surface defects on manufactured metal surfaces. The system processes images or live camera feeds, detects defects using YOLOv8, and is designed for optimized edge deployment using ONNX/TensorRT.

---

## 🚦 Current Status — Last Updated: 2026-09-23

### ⏳ Currently At: Project Setup & Dataset Preparation

> **Next Step → Prepare the NEU Metal Surface Defects dataset, create/verify bounding-box annotations, and establish the training pipeline.**

### Current Development Stage

| Stage | Status |
|---|---|
| Project Structure | ✅ Planned |
| Dataset Selection | ✅ Selected |
| Dataset Preparation | ⏳ In Progress |
| Data Augmentation | ⬜ Not Started |
| YOLOv8 Training | ⬜ Not Started |
| Model Evaluation | ⬜ Not Started |
| Edge Optimization | ⬜ Not Started |
| Real-Time Camera | ⬜ Not Started |
| FastAPI Backend | ⬜ Not Started |
| Monitoring | ⬜ Not Started |
| Docker Deployment | ⬜ Not Started |

---

# 📊 Progress Tracker

## Stage 1 — Dataset Preparation & Image Processing

| # | Task | Files / Module | Status |
|---|---|---|---|
| 1 | **NEU Dataset Collection** | `data/raw/` | ⏳ In Progress |
| 2 | **Dataset Organization** | `data/images/`, `data/labels/` | ⬜ Not Started |
| 3 | **Annotation Verification** | `annotation_checker.py` | ⬜ Not Started |
| 4 | **YOLO Annotation Conversion** | `convert_annotations.py` | ⬜ Not Started |
| 5 | **Train/Validation/Test Split** | `dataset_split.py` | ⬜ Not Started |
| 6 | **Image Preprocessing** | `preprocessing.py` | ⬜ Not Started |
| 7 | **Data Augmentation** | `augmentation.py` | ⬜ Not Started |
| 8 | **Dataset Statistics** | `dataset_stats.py` | ⬜ Not Started |

### Stage 1 Deliverables

- NEU Metal Surface Defects dataset organized
- Six defect classes verified
- YOLO-compatible annotations
- Train/validation/test datasets
- Image preprocessing pipeline
- Data augmentation pipeline
- Dataset statistics

---

# Stage 2 — YOLOv8 Model Training

| # | Task | Files / Module | Status |
|---|---|---|---|
| 1 | **YOLOv8 Environment Setup** | `requirements.txt` | ⬜ Not Started |
| 2 | **Dataset Configuration** | `data/dataset.yaml` | ⬜ Not Started |
| 3 | **Baseline Model Setup** | `models/yolov8n.pt` | ⬜ Not Started |
| 4 | **Model Training** | `train.py` | ⬜ Not Started |
| 5 | **Hyperparameter Tuning** | `configs/` | ⬜ Not Started |
| 6 | **Validation** | `validate.py` | ⬜ Not Started |
| 7 | **Error Analysis** | `error_analysis.py` | ⬜ Not Started |
| 8 | **Best Model Selection** | `models/best.pt` | ⬜ Not Started |

### Training Parameters

| Parameter | Planned Value |
|---|---|
| Model | YOLOv8 |
| Initial Model | YOLOv8n |
| Image Size | 640 × 640 |
| Epochs | 50+ |
| Batch Size | 16, hardware dependent |
| Optimizer | YOLO/Ultralytics default initially |
| Classes | 6 |
| Framework | PyTorch + Ultralytics |
| Evaluation | Precision, Recall, mAP |

---

# Stage 3 — Model Evaluation & Error Analysis

| # | Task | Files / Module | Status |
|---|---|---|---|
| 1 | **Validation Dataset Evaluation** | `validate.py` | ⬜ Not Started |
| 2 | **Precision Calculation** | `metrics.py` | ⬜ Not Started |
| 3 | **Recall Calculation** | `metrics.py` | ⬜ Not Started |
| 4 | **mAP@50 Calculation** | `metrics.py` | ⬜ Not Started |
| 5 | **mAP@50-95 Calculation** | `metrics.py` | ⬜ Not Started |
| 6 | **False Positive Analysis** | `error_analysis.py` | ⬜ Not Started |
| 7 | **False Negative Analysis** | `error_analysis.py` | ⬜ Not Started |
| 8 | **Confusion Matrix** | `evaluation/` | ⬜ Not Started |
| 9 | **Inference Speed Testing** | `benchmark.py` | ⬜ Not Started |

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures correctness of detected defects |
| Recall | Measures how many actual defects are detected |
| mAP@50 | Detection performance at IoU 0.50 |
| mAP@50-95 | Detection performance across multiple IoU thresholds |
| FPS | Frames processed per second |
| Latency | Time required for inference |
| False Positives | Incorrect defect detections |
| False Negatives | Missed defects |

### Expected Evaluation Output

| Output | Description |
|---|---|
| `precision` | Overall detection precision |
| `recall` | Overall detection recall |
| `mAP50` | Mean Average Precision at IoU 0.50 |
| `mAP50-95` | Mean Average Precision across IoU thresholds |
| `confusion_matrix` | Class-level error analysis |
| `inference_latency` | Model inference time |
| `FPS` | Real-time processing capability |

> Actual performance values will be added after model training and evaluation. No performance numbers are assumed before testing.

---

# Stage 4 — Real-Time Detection

| # | Task | Files / Module | Status |
|---|---|---|---|
| 1 | **OpenCV Camera Capture** | `camera.py` | ⬜ Not Started |
| 2 | **Frame Preprocessing** | `preprocessing.py` | ⬜ Not Started |
| 3 | **YOLOv8 Inference** | `detect.py` | ⬜ Not Started |
| 4 | **Bounding Box Rendering** | `visualization.py` | ⬜ Not Started |
| 5 | **Confidence Display** | `visualization.py` | ⬜ Not Started |
| 6 | **Defect Counter** | `counter.py` | ⬜ Not Started |
| 7 | **FPS Measurement** | `benchmark.py` | ⬜ Not Started |
| 8 | **Live Video Testing** | `camera_test.py` | ⬜ Not Started |

### Real-Time Pipeline

```text
Industrial Camera
       ↓
OpenCV Frame Capture
       ↓
Image Preprocessing
       ↓
YOLOv8 Model
       ↓
Defect Detection
       ↓
Bounding Boxes
       ↓
Confidence Scores
       ↓
Defect Classification
       ↓
Real-Time Display
