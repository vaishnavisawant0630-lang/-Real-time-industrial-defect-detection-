# 🚀 Real-Time Industrial Defect Detection System

> **Computer Vision · YOLOv8 · Industrial Quality Control · Edge AI · Real-Time Inspection**

A real-time computer vision system designed to **detect, classify, localize, and analyze surface defects on manufactured metal surfaces**.

The system uses **YOLOv8** for object detection and supports image-based and real-time camera inspection. It provides defect localization, confidence scores, quality assessment, inspection results, monitoring, API integration, and deployment support.

---

## 📌 Project Overview

Industrial manufacturing requires fast and reliable quality inspection to identify defects such as:

- 🔴 Crazing
- 🔴 Inclusion
- 🔴 Patches
- 🔴 Pitted Surface
- 🔴 Rolled-in Scale
- 🔴 Scratches

Manual inspection can be time-consuming, inconsistent, and difficult to scale.

This project provides an **AI-powered automated inspection system** that analyzes metal surfaces and identifies defects using computer vision and deep learning.

### 🎯 Main Objectives

- Detect surface defects automatically
- Localize defects using bounding boxes
- Classify different defect types
- Display detection confidence
- Perform real-time camera inspection
- Calculate inspection/quality results
- Maintain inspection records
- Provide API-based backend integration
- Support monitoring and analytics
- Prepare the system for edge deployment
- Provide Docker-based deployment

---

# 🚦 Current Project Status

**Last Updated:** `2026-10-15`

### 🟢 Project Status: COMPLETE

> **All major development stages have been completed, including dataset preparation, YOLOv8 training, evaluation, real-time detection, FastAPI backend, monitoring, and Docker deployment.**

### Overall Progress

| Stage | Status |
|---|---|
| 📁 Project Structure | ✅ Complete |
| 📊 Dataset Preparation | ✅ Complete |
| 🧹 Data Processing | ✅ Complete |
| 🔄 Data Augmentation | ✅ Complete |
| 🤖 YOLOv8 Training | ✅ Complete |
| 📈 Model Evaluation | ✅ Complete |
| 🔍 Error Analysis | ✅ Complete |
| ⚡ Edge Optimization | ✅ Complete |
| 📷 Real-Time Camera | ✅ Complete |
| 🌐 FastAPI Backend | ✅ Complete |
| 📊 Monitoring | ✅ Complete |
| 🐳 Docker Deployment | ✅ Complete |
| 🎨 Streamlit Dashboard | ✅ Complete |

---

# 📊 Complete Progress Tracker

## Stage 1 — Dataset Preparation & Image Processing

| # | Task | File / Directory | Status |
|---:|---|---|---|
| 1 | NEU Dataset Collection | `data/raw/` | ✅ Complete |
| 2 | Dataset Organization | `data/images/`, `data/labels/` | ✅ Complete |
| 3 | Annotation Verification | `annotation_checker.py` | ✅ Complete |
| 4 | YOLO Annotation Conversion | `convert_annotations.py` | ✅ Complete |
| 5 | Train/Validation/Test Split | `dataset_split.py` | ✅ Complete |
| 6 | Image Preprocessing | `preprocessing.py` | ✅ Complete |
| 7 | Data Augmentation | `augmentation.py` | ✅ Complete |
| 8 | Dataset Statistics | `dataset_stats.py` | ✅ Complete |

### Stage 1 Deliverables

- ✅ NEU Metal Surface Defects dataset organized
- ✅ Six defect classes prepared
- ✅ Bounding-box annotations verified
- ✅ YOLO-compatible annotations created
- ✅ Train/validation/test datasets created
- ✅ Image preprocessing pipeline completed
- ✅ Data augmentation pipeline completed
- ✅ Dataset statistics generated

---

# Stage 2 — YOLOv8 Model Training

| # | Task | File / Module | Status |
|---:|---|---|---|
| 1 | YOLOv8 Environment Setup | `requirements.txt` | ✅ Complete |
| 2 | Dataset Configuration | `data/dataset.yaml` | ✅ Complete |
| 3 | Baseline Model Setup | `models/yolov8n.pt` | ✅ Complete |
| 4 | Model Training | `train.py` | ✅ Complete |
| 5 | Hyperparameter Tuning | `configs/` | ✅ Complete |
| 6 | Validation | `validate.py` | ✅ Complete |
| 7 | Error Analysis | `error_analysis.py` | ✅ Complete |
| 8 | Best Model Selection | `models/best.pt` | ✅ Complete |

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | YOLOv8 |
| Initial Model | YOLOv8n |
| Image Size | 640 × 640 |
| Epochs | 50+ |
| Batch Size | Hardware dependent |
| Optimizer | Ultralytics default / tuned |
| Number of Classes | 6 |
| Framework | PyTorch + Ultralytics |
| Evaluation | Precision, Recall, mAP |

---

# Stage 3 — Model Evaluation & Error Analysis

| # | Task | File / Module | Status |
|---:|---|---|---|
| 1 | Validation Dataset Evaluation | `validate.py` | ✅ Complete |
| 2 | Precision Calculation | `metrics.py` | ✅ Complete |
| 3 | Recall Calculation | `metrics.py` | ✅ Complete |
| 4 | mAP@50 Calculation | `metrics.py` | ✅ Complete |
| 5 | mAP@50-95 Calculation | `metrics.py` | ✅ Complete |
| 6 | False Positive Analysis | `error_analysis.py` | ✅ Complete |
| 7 | False Negative Analysis | `error_analysis.py` | ✅ Complete |
| 8 | Confusion Matrix | `evaluation/` | ✅ Complete |
| 9 | Inference Speed Testing | `benchmark.py` | ✅ Complete |

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures correctness of detections |
| Recall | Measures detected actual defects |
| mAP@50 | Detection performance at IoU 0.50 |
| mAP@50-95 | Detection performance across IoU thresholds |
| FPS | Real-time processing performance |
| Latency | Inference processing time |
| False Positives | Incorrect defect detections |
| False Negatives | Missed defect detections |

> **Note:** Actual numerical performance values should be reported only from the final experiment outputs.

---

# Stage 4 — Real-Time Detection

| # | Task | File / Module | Status |
|---:|---|---|---|
| 1 | OpenCV Camera Capture | `camera.py` | ✅ Complete |
| 2 | Frame Preprocessing | `preprocessing.py` | ✅ Complete |
| 3 | YOLOv8 Inference | `detect.py` | ✅ Complete |
| 4 | Bounding Box Rendering | `visualization.py` | ✅ Complete |
| 5 | Confidence Display | `visualization.py` | ✅ Complete |
| 6 | Defect Counter | `counter.py` | ✅ Complete |
| 7 | FPS Measurement | `benchmark.py` | ✅ Complete |
| 8 | Live Video Testing | `camera_test.py` | ✅ Complete |

### Real-Time Detection Pipeline

```text
┌──────────────────────────┐
│    Industrial Camera     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   OpenCV Frame Capture   │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Image Preprocessing    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       YOLOv8 Model       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Defect Detection      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Bounding Boxes + Class   │
│ + Confidence Score       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Defect Counter + FPS     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Inspection Result      │
└──────────────────────────┘
```

---

# Stage 5 — Edge Optimization

| # | Task | Status |
|---:|---|---|
| 1 | Model Optimization | ✅ Complete |
| 2 | ONNX Export | ✅ Complete |
| 3 | Inference Optimization | ✅ Complete |
| 4 | Edge Performance Testing | ✅ Complete |
| 5 | Latency Benchmarking | ✅ Complete |

### Edge Deployment Pipeline

```text
YOLOv8 Model
     ↓
Model Optimization
     ↓
ONNX Export
     ↓
Runtime
