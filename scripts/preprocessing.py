import cv2
import numpy as np


def  preprocess_frame(frame):
    """Validate a BGR image/frame before model inference."""

    if frame is None:
        raise ValueError("Input image/frame is empty")

    if not isinstance(frame, np.ndarray):
        raise TypeError("Input must be a Numpy image array.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    if frame.size == 0:
        raise ValueError("Input image has no pixels.")

    return np.ascontiguousarray(frame)

def load_image(image_path):
    """Load an image from disk and prepare it for inference."""
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(f"Could not read image:{image_path}")

    return preprocess_frame(frame)