from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

IMAGE_DIR = Path("data/processed/images")
XML_DIR = Path("data/processed/annotations/xml")

CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
}

images = list(IMAGE_DIR.glob("*"))
xml_files = list(XML_DIR.glob("*"))

print("="*70)
print("PROCESSED DATSET VALIDATION")
print("="*70)

class_counts = Counter()
errors = []

for xml_file in xml_files:
    try:
        root = ET.parse(xml_file).getroot()
        filename = root.findtext("filename")
        if not filename:
            errors.append(
                f"{xml_file.name}: image missing: {filename}"
            )
        size = root.find("size")

        if size is None:
            errors.append(
                f"{xml_file.name}: missing size"
            )

            continue

        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        if width <= 0 or height <= 0:
            errors.append(
                f"{xml_file.name}: invalid image dimensions"
            )

        for obj in root.findall("object"):
            class_name = obj.findtext("name")

            if class_name not in CLASSES:
                errors.append(
                    f"{xml_file.name}: "f"Unkown class {class_name}"
                )
                continue

            class_counts[class_name] += 1

            bbox = obj.find("bndbox")

            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))

            if xmin < 0 or ymin < 0:
                errors.append(
                    f"{xml_file.name}: negative bbox"
                )

            if xmax > width or ymax > height:
                errors.append(
                    f"{xml_file.name}: bbox outside image"
                )

            if xmax <= xmin:
                errors.append(
                    f"{xml_file.name}: invalid bbox width"
                )

            if ymax <= ymin:
                errors.append(
                    f"{xml_file.name}: invalid bbox height"
                )

    except Exception as e:
        errors.append(
            f"{xml_file.name}: {e}"
        )

print("\nClass distribution:")

for class_name in sorted(CLASSES):
    print(
        f"{class_name:20s}:"
        f"{class_counts[class_name]}"
    )

print("\nValidation result:")

if errors:
    print(
        f"FAILED - {len(errors)} errors"
    )

    for error in errors[:30]:
        print(error)

else:
    print("PASSED - No annotation errors found.")

            
