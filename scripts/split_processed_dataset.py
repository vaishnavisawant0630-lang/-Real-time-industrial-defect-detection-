from pathlib import Path
import random
import shutil

IMAGE_DIR = Path("data/processed/images")
XML_DIR = Path("data/processed/annotations/xml")

OUTPUT = Path("data/processed")

random.seed(42)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".png",
    ".jpeg",
    ".bmp"
}

# FIND IMAGE/XML PAIRS

pairs = []

for image in IMAGE_DIR.iterdir():
    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    xml = XML_DIR / f"{image.stem}.xml"

    if xml.exists():
        pairs.append((image,xml))


print("="*70)
print("DATASET SPLITTING")
print("="*70)
print(f"Valid image/xml pairs: {len(pairs)}")

# SHUFFLE

random.shuffle(pairs)

# SPLIT

total = len(pairs)

train_end = int(total * 0.70)
val_end = int(total * 0.90)

train_data = pairs[:train_end]
val_data = pairs[train_end:val_end]
test_data = pairs[val_end:]

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

# CREATE DIRECTORIES

for split in splits:
    (
        OUTPUT
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT
        / "annotations"
        / "xml"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

# COPY FILES

for split, data in splits.items():
    print(
        f"\nCopying {split}: {len(data)}"
    )

    for image,xml in data:
        shutil.copy2(
            image,
            OUTPUT
            / "images"
            / split
            / image.name
        )

        shutil.copy2(
            xml,
            OUTPUT
            / "annotations"
            / "xml"
            / split
            / xml.name
        )

# SUMMARY

print("\n")
print("="*70)
print("SPLIT COMPLETE")
print("="*70)

print(f"Total : {total}")
print(f"Train : {len(train_data)}")
print(f"Val   : {len(val_data)}")
print(f"Test  : {len(test_data)}")

