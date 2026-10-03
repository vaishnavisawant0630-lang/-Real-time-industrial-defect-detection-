import cv2
import numpy as np

IMAGE_SIZE = 640

def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image:"
            f"{image_path}"
        )

    return image

def resize_image(image):
    return cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

def normalize_image(image):
    return image.astype(
        np.float32
    ) / 255.0

def preprocess_image(image):
    image = resize_image(image)
    image = normalize_image(image)

    return image

