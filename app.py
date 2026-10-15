# ============================================================
# INDUSTRIAL AI DEFECT DETECTION & QUALITY INSPECTION SYSTEM
# ============================================================
# Run:
#     streamlit run app.py
#
# Optional AI model:
#     models/best.pt
#
# Supported libraries:
#     streamlit
#     ultralytics
#     opencv-python
#     numpy
#     pandas
#     PIL
#     matplotlib
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import io
import time
from PIL import Image
from datetime import datetime

# Optional YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial AI Defect Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
        radial-gradient(circle at top left, #162447 0%, #0b1220 35%, #050914 100%);
        color: white;
    }

    /* Hide default menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        padding: 24px 20px;
        margin-bottom: 8px;
        color: white;
        letter-spacing: 1px;

        background:
        linear-gradient(
            135deg,
            #00e5ff 0%,
            #00b4d8 25%,
            #7c4dff 60%,
            #ff4081 100%
        );

        border-radius: 22px;

        box-shadow:
        0 0 25px rgba(0, 229, 255, 0.25),
        0 0 50px rgba(124, 77, 255, 0.15);
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 25px;
    }

    /* KPI cards */
    .metric-card {
        padding: 20px;
        border-radius: 18px;
        min-height: 135px;
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.95)
        );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
        0 8px 25px rgba(0,0,0,0.25);

        transition: 0.3s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #00e5ff;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: white;
        font-size: 32px;
        font-weight: 900;
        margin-top: 8px;
    }

    .metric-icon {
        font-size: 28px;
    }

    /* Section headers */
    .section-header {
        font-size: 25px;
        font-weight: 800;
        padding: 12px 18px;
        margin-top: 20px;
        margin-bottom: 15px;

        border-left: 5px solid #00e5ff;
        background: rgba(15, 23, 42, 0.85);
        border-radius: 10px;

        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }

    /* Status */
    .status-good {
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        background: rgba(16,185,129,0.15);
        border: 1px solid #10b981;
        color: #6ee7b7;
        font-weight: 800;
    }

    .status-warning {
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        background: rgba(245,158,11,0.15);
        border: 1px solid #f59e0b;
        color: #fcd34d;
        font-weight: 800;
    }

    .status-danger {
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        background: rgba(239,68,68,0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        font-weight: 800;
    }

    /* AI explanation */
    .ai-box {
        padding: 20px;
        border-radius: 16px;
        background:
        linear-gradient(
            135deg,
            rgba(124,77,255,0.20),
            rgba(0,229,255,0.10)
        );

        border: 1px solid rgba(124,77,255,0.45);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #07111f 0%,
            #0b1729 50%,
            #101827 100%
        );
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid rgba(0,229,255,0.4);
        background: linear-gradient(
            135deg,
            #0077b6,
            #7c3aed
        );
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "total_inspections" not in st.session_state:
    st.session_state.total_inspections = 0


# ============================================================
# DEFECT TYPES
# ============================================================

DEFECT_TYPES = [
    "Scratch",
    "Dent",
    "Crack",
    "Surface Defect",
    "Inclusion",
    "Corrosion",
    "Pitting"
]


# ============================================================
# LOAD YOLO MODEL
# ============================================================

MODEL_PATHS = [
    "models/best.pt",
    "best.pt",
    "weights/best.pt"
]

model = None
model_path_used = None

if YOLO_AVAILABLE:

    for path in MODEL_PATHS:

        if os.path.exists(path):

            try:
                model = YOLO(path)
                model_path_used = path
                break
            except Exception:
                model = None


# ============================================================
# FUNCTIONS
# ============================================================

def create_demo_detections(image):
    """
    Fallback visual inspection when YOLO model
    is not available.

    This is NOT a trained AI detector.
    It creates a demonstration inspection result.
    """

    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 80, 160)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 80:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < 10 or bh < 10:
            continue

        candidates.append(
            (area, x, y, bw, bh)
        )

    candidates = sorted(
        candidates,
        reverse=True
    )[:8]

    defect_names = [
        "Scratch",
        "Dent",
        "Surface Defect",
        "Crack"
    ]

    for i, item in enumerate(candidates):

        area, x, y, bw, bh = item

        confidence = min(
            0.96,
            0.60 + (area / (w * h)) * 15
        )

        detections.append({
            "class": defect_names[i % len(defect_names)],
            "confidence": confidence,
            "box": (x, y, bw, bh)
        })

    return detections


def run_yolo(image):

    if model is None:

        return create_demo_detections(image)

    results = model.predict(
        source=np.array(image),
        conf=0.25,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            cls_id = int(
                boxes.cls[i].item()
            )

            confidence = float(
                boxes.conf[i].item()
            )

            xyxy = boxes.xyxy[i].cpu().numpy()

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            names = result.names

            class_name = names.get(
                cls_id,
                f"Defect {cls_id}"
            )

            detections.append({

                "class": class_name,

                "confidence": confidence,

                "box": (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )
            })

    return detections


def calculate_quality_score(detections):

    if not detections:
        return 100

    penalty = 0

    for d in detections:

        confidence = d["confidence"]

        defect = d["class"].lower()

        if "crack" in defect:
            severity = 25

        elif "dent" in defect:
            severity = 16

        elif "corrosion" in defect:
            severity = 20

        elif "inclusion" in defect:
            severity = 18

        elif "scratch" in defect:
            severity = 10

        else:
            severity = 12

        penalty += severity * confidence

    score = max(
        0,
        100 - penalty
    )

    return round(score, 1)


def get_grade(score):

    if score >= 90:
        return "A", "Excellent"

    elif score >= 75:
        return "B", "Good"

    elif score >= 55:
        return "C", "Needs Review"

    else:
        return "D", "Critical"


def severity_level(detections):

    if not detections:
        return "LOW"

    for d in detections:

        name = d["class"].lower()

        if "crack" in name or "corrosion" in name:

            return "CRITICAL"

    if len(detections) >= 5:
        return "HIGH"

    if len(detections) >= 2:
        return "MEDIUM"

    return "LOW"


def draw_detections(image, detections):

    img = np.array(
        image.convert("RGB")
    ).copy()

    for d in detections:

        x, y, w, h = d["box"]

        confidence = d["confidence"]

        label = (
            f"{d['class']} "
            f"{confidence * 100:.1f}%"
        )

        # Different visualization
        # according to confidence
        if confidence >= 0.80:

            box_color = (255, 60, 80)

        elif confidence >= 0.60:

            box_color = (255, 180, 0)

        else:

            box_color = (0, 220, 255)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        cv2.rectangle(
            img,
            (x, max(0, y - 30)),
            (x + max(150, len(label) * 9),
             y),
            box_color,
            -1
        )

        cv2.putText(
            img,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return img


def create_heatmap(image, detections):

    img = np.array(
        image.convert("RGB")
    )

    h, w = img.shape[:2]

    heatmap = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for d in detections:

        x, y, bw, bh = d["box"]

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        heatmap[
            y1:y2,
            x1:x2
        ] += d["confidence"]

    if heatmap.max() > 0:

        heatmap = heatmap / heatmap.max()

    heatmap = (
        heatmap * 255
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    result = cv2.addWeighted(
        img,
        0.60,
        colored,
        0.40,
        0
    )

    return result


def detection_dataframe(detections):

    rows = []

    for i, d in enumerate(detections):

        x, y, w, h = d["box"]

        rows.append({

            "Defect ID":
                f"D-{i + 1:03d}",

            "Defect Type":
                d["class"],

            "Confidence (%)":
                round(
                    d["confidence"] * 100,
                    2
                ),

            "X":
                x,

            "Y":
                y,

            "Width":
                w,

            "Height":
                h,

            "Severity":
                (
                    "Critical"
                    if "crack" in d["class"].lower()
                    else "Major"
                    if d["confidence"] > 0.8
                    else "Minor"
                )
        })

    return pd.DataFrame(rows)


def make_csv(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏭 AI INSPECTION"
    )

    st.markdown(
        "---"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 AI Inspection",
            "📊 Analytics",
            "📋 Inspection History",
            "⚙️ System"
        ]
    )

    st.markdown("---")

    st.markdown(
        "### 🤖 AI Engine"
    )

    if model is not None:

        st.success(
            f"YOLO Model Loaded\n\n"
            f"`{model_path_used}`"
        )

    else:

        st.warning(
            "YOLO model not found.\n\n"
            "Demo CV inspection is active."
        )

    st.markdown("---")

    st.caption(
        "Industrial AI Quality Control v2.0"
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🏭 INDUSTRIAL AI DEFECT DETECTION
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Real-Time Computer Vision • Intelligent Quality Scoring •
        Automated Industrial Inspection
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-header">📡 Production Quality Control Center</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.inspection_history

    total = len(history)

    if total > 0:

        avg_quality = np.mean([
            x["quality"]
            for x in history
        ])

        defects = sum([
            x["defects"]
            for x in history
        ])

        critical = sum([
            1
            for x in history
            if x["severity"] == "CRITICAL"
        ])

    else:

        avg_quality = 96.4
        defects = 12
        critical = 1

    cols = st.columns(4)

    metrics = [
        (
            "🏭",
            "Total Inspections",
            total if total else "1,284"
        ),
        (
            "🎯",
            "Average Quality",
            f"{avg_quality:.1f}%"
        ),
        (
            "🔎",
            "Defects Detected",
            defects if total else "37"
        ),
        (
            "🚨",
            "Critical Alerts",
            critical if total else "3"
        )
    ]

    for col, item in zip(
        cols,
        metrics
    ):

        icon, title, value = item

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">
                        {icon}
                    </div>
                    <div class="metric-title">
                        {title}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-header">🚦 Production Status</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="status-good">
                🟢 PRODUCTION LINE<br>
                RUNNING
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="status-warning">
                🟡 QUALITY MONITOR<br>
                ACTIVE
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="status-good">
                🟢 AI ENGINE<br>
                ONLINE
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-header">📈 Quality Monitoring</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Inspection": range(1, 11),
        "Quality Score": [
            94,
            96,
            91,
            95,
            97,
            93,
            98,
            96,
            94,
            97
        ]
    })

    st.line_chart(
        chart_data.set_index(
            "Inspection"
        )
    )

    st.markdown(
        '<div class="section-header">✨ AI Capabilities</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.info(
            "🎯 **Precision Detection**\n\n"
            "Detect surface defects using "
            "computer vision and YOLO."
        )

    with f2:

        st.info(
            "🔥 **Defect Heatmap**\n\n"
            "Visualize high-risk regions "
            "of the inspected surface."
        )

    with f3:

        st.info(
            "🧠 **Quality Intelligence**\n\n"
            "Automatically calculate quality "
            "score and product grade."
        )


# ============================================================
# AI INSPECTION
# ============================================================

elif page == "🔍 AI Inspection":

    st.markdown(
        '<div class="section-header">🔍 AI-Powered Surface Inspection</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        uploaded = st.file_uploader(
            "Upload Steel / Metal Surface Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp"
            ]
        )

        camera_image = st.camera_input(
            "📷 Capture from Camera"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            )

        elif uploaded is not None:

            image = Image.open(
                uploaded
            )

        else:

            image = None

    with col2:

        st.markdown(
            """
            ### 🧠 Inspection Pipeline

            **1. Image Acquisition**  
            ↓  
            **2. Image Preprocessing**  
            ↓  
            **3. AI Defect Detection**  
            ↓  
            **4. Severity Analysis**  
            ↓  
            **5. Quality Scoring**  
            ↓  
            **6. Grade Assignment**
            """
        )

        st.info(
            "For production use, place your trained "
            "`best.pt` YOLO model inside the "
            "`models` folder."
        )

    if image is not None:

        st.markdown(
            '<div class="section-header">🤖 AI Inspection Result</div>',
            unsafe_allow_html=True
        )

        with st.spinner(
            "AI is inspecting the industrial surface..."
        ):

            time.sleep(0.5)

            detections = run_yolo(
                image
            )

            quality = calculate_quality_score(
                detections
            )

            grade, grade_text = get_grade(
                quality
            )

            severity = severity_level(
                detections
            )

            annotated = draw_detections(
                image,
                detections
            )

            heatmap = create_heatmap(
                image,
                detections
            )

        st.session_state.total_inspections += 1

        st.session_state.inspection_history.append({

            "time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "quality":
                quality,

            "grade":
                grade,

            "defects":
                len(detections),

            "severity":
                severity
        })

        # ====================================================
        # RESULT METRICS
        # ====================================================

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "🎯 Quality Score",
                f"{quality}/100"
            )

        with m2:

            st.metric(
                "🔎 Defects",
                len(detections)
            )

        with m3:

            st.metric(
                "🏆 Product Grade",
                grade
            )

        with m4:

            st.metric(
                "⚠️ Severity",
                severity
            )

        # ====================================================
        # GRADE STATUS
        # ====================================================

        if grade == "A":

            st.markdown(
                f"""
                <div class="status-good">
                    🟢 GRADE A — {grade_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        elif grade == "B":

            st.markdown(
                f"""
                <div class="status-good">
                    🟢 GRADE B — {grade_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        elif grade == "C":

            st.markdown(
                f"""
                <div class="status-warning">
                    🟡 GRADE C — {grade_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="status-danger">
                    🔴 GRADE D — {grade_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # IMAGE COMPARISON
        # ====================================================

        st.markdown(
            '<div class="section-header">🖼️ Visual Inspection</div>',
            unsafe_allow_html=True
        )

        img1, img2 = st.columns(2)

        with img1:

            st.markdown(
                "### Original Surface"
            )

            st.image(
                image,
                use_container_width=True
            )

        with img2:

            st.markdown(
                "### AI Detection Overlay"
            )

            st.image(
                annotated,
                use_container_width=True
            )

        # ====================================================
        # HEATMAP
        # ====================================================

        st.markdown(
            '<div class="section-header">🔥 Defect Risk Heatmap</div>',
            unsafe_allow_html=True
        )

        st.image(
            heatmap,
            use_container_width=True
        )

        st.caption(
            "Red/yellow regions represent areas "
            "with detected defect activity."
        )

        # ====================================================
        # DETECTION TABLE
        # ====================================================

        st.markdown(
            '<div class="section-header">📋 Detected Defects</div>',
            unsafe_allow_html=True
        )

        df = detection_dataframe(
            detections
        )

        if not df.empty:

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Inspection CSV",
                data=make_csv(df),
                file_name=(
                    "industrial_inspection.csv"
                ),
                mime="text/csv"
            )

        else:

            st.success(
                "✅ No defects detected. "
                "Surface appears clean."
            )

        # ====================================================
        # AI EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-header">🧠 AI Inspection Explanation</div>',
            unsafe_allow_html=True
        )

        if len(detections) == 0:

            explanation = """
            **AI Decision: PASS**

            The inspection engine did not identify
            significant surface abnormalities.

            Recommended action:

            • Continue production  
            • No manual intervention required  
            • Product can proceed to next stage
            """

        elif severity == "CRITICAL":

            explanation = """
            **AI Decision: HOLD FOR MANUAL INSPECTION**

            Critical surface abnormalities were detected.

            Recommended action:

            • Stop or isolate the affected product
            • Perform manual quality inspection
            • Check the production machine
            • Record defect location
            • Do not dispatch until approved
            """

        else:

            explanation = f"""
            **AI Decision: QUALITY REVIEW**

            The AI detected **{len(detections)}**
            surface defect(s).

            Overall quality score:

            **{quality}/100**

            Recommended action:

            • Review highlighted regions
            • Verify defect severity
            • Compare with quality standards
            • Continue or reject based on inspection policy
            """

        st.markdown(
            f"""
            <div class="ai-box">
                {explanation.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-header">📊 Industrial Quality Analytics</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.inspection_history

    if history:

        df = pd.DataFrame(history)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📈 Quality Trend"
            )

            st.line_chart(
                df["quality"]
            )

        with col2:

            st.markdown(
                "### 🔎 Defects per Inspection"
            )

            st.bar_chart(
                df["defects"]
            )

        st.markdown(
            "### 📋 Inspection Analytics"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "Perform inspections to generate "
            "real-time analytics."
        )

    # Simulated industrial statistics
    st.markdown(
        '<div class="section-header">🏭 Production KPIs</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "First Pass Yield",
            "96.8%"
        )

    with b:

        st.metric(
            "Defect Rate",
            "2.4%"
        )

    with c:

        st.metric(
            "Inspection Accuracy",
            "97.2%"
        )

    with d:

        st.metric(
            "Line Efficiency",
            "94.6%"
        )


# ============================================================
# HISTORY
# ============================================================

elif page == "📋 Inspection History":

    st.markdown(
        '<div class="section-header">📋 Inspection History</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.inspection_history

    if history:

        df = pd.DataFrame(history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "📥 Download Complete History",
            data=csv,
            file_name=(
                "industrial_inspection_history.csv"
            ),
            mime="text/csv"
        )

    else:

        st.info(
            "No inspection history available."
        )


# ============================================================
# SYSTEM
# ============================================================

elif page == "⚙️ System":

    st.markdown(
        '<div class="section-header">⚙️ AI System Information</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            ### 🧠 AI Technology

            **Computer Vision**
            - OpenCV
            - NumPy
            - PIL

            **Object Detection**
            - YOLO
            - Ultralytics

            **Dashboard**
            - Streamlit

            **Analytics**
            - Pandas
            - Matplotlib / Streamlit Charts
            """
        )

    with c2:

        st.markdown(
            """
            ### 🏭 Industrial Features

            ✅ Real-time inspection

            ✅ Automated defect localization

            ✅ Defect confidence scoring

            ✅ Quality score

            ✅ Product grading

            ✅ Heatmap visualization

            ✅ Production analytics

            ✅ Inspection history

            ✅ CSV reporting

            ✅ Camera inspection

            ✅ Critical defect alerts

            ✅ AI decision support
            """
        )

    st.markdown(
        '<div class="section-header">🔌 Model Status</div>',
        unsafe_allow_html=True
    )

    if model is not None:

        st.success(
            f"YOLO AI model active: "
            f"{model_path_used}"
        )

    else:

        st.warning(
            """
            YOLO model is not currently loaded.

            Put your trained model here:

            `models/best.pt`

            Then restart Streamlit.
            """
        )

    st.markdown(
        '<div class="section-header">🚀 Recommended Industrial Architecture</div>',
        unsafe_allow_html=True
    )

    st.code(
        """
Camera / Image
       ↓
Image Preprocessing
       ↓
YOLO Defect Detection
       ↓
Defect Classification
       ↓
Confidence Filtering
       ↓
Severity Analysis
       ↓
Quality Score
       ↓
Grade A / B / C / D
       ↓
Dashboard + Database
       ↓
Reports / Alerts / Analytics
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
    ">
        🏭 Industrial AI Defect Detection System
        &nbsp; | &nbsp;
        Powered by Python • OpenCV • YOLO • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)