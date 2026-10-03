from pathlib import Path
import random
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "augmented"
    / "images"
)


def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

def brightness_change(image):
    factor = random.uniform(0.8,1.2)

    result = (
        image.astype("float32")
        * factor
    )

    return result.clip(
        0,
        255
    ).astype("uint8")

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]:
        images.extend(SOURCE_DIR.glob(extension))

    print(f"Found {len(images)} training inages.")

    for image_path in images:
        image = cv2.imread(str(image_path))

        flipped = horizontal_flip(image)
        output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_flip"
            f"{image_path.suffix}"
        )

        cv2.write(
            str(output_path),
            flipped
        )

    print("Augmentation complete.")

if __name__ == "__main__":
    main()