from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images"
)

XML_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "xml"
)

SPLITS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    images = sorted(
        IMAGE_SOURCE.glob("*.*")
    )

    valid_pairs = []

    for image in images:
        xml = XML_SOURCE / f"{image.stem}.xml"
        if xml.exists():
            valid_pairs.append(
                (image,xml)
            )
    random.shuffle(valid_pairs)
    total = len(valid_pairs)

    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_data = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:],
    }

    for split, pairs in split_data.items():
        image_dir = IMAGE_SOURCE / split
        xml_dir = XML_SOURCE / split

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        xml_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image,xml in pairs:
            shutil.copy2(
                image,
                image_dir / image.name
            )
            shutil.copy2(
                xml,
                xml_dir / xml.name
            )

        print(f"{split}: {len(pairs)}")

    print()
    print("DATASET SPLIT COMPLETE")

if __name__ == "__main__":
    main()

    
        


