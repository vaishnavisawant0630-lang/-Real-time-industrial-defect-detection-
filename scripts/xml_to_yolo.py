from pathlib import Path
import xml.etree.ElementTree as ET


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_MAP = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

XML_ROOT = PROJECT_ROOT / "data" / "processed" / "annotations" / "xml"
LABEL_ROOT = PROJECT_ROOT / "data" / "processed" / "labels"


def convert_xml(xml_path, output_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    image_width = float(size.find("width").text)
    image_height = float(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_MAP:
            print(f"Unknown class: {class_name}")
            continue

        class_id = CLASS_MAP[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp coordinates
        xmin = max(0, min(xmin, image_width))
        xmax = max(0, min(xmax, image_width))
        ymin = max(0, min(ymin, image_height))
        ymax = max(0, min(ymax, image_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


def main():

    total = 0

    for split in ["train", "val", "test"]:

        xml_dir = XML_ROOT / split
        label_dir = LABEL_ROOT / split

        xml_files = sorted(xml_dir.glob("*.xml"))

        print(f"{split.upper()}: {len(xml_files)} XML files")

        for xml_file in xml_files:

            output_file = label_dir / f"{xml_file.stem}.txt"

            try:
                convert_xml(xml_file, output_file)
                total += 1

            except Exception as e:
                print(f"ERROR: {xml_file.name}")
                print(e)

    print()
    print("XML → YOLO CONVERSION COMPLETE")
    print(f"Converted XML files: {total}")


if __name__ == "__main__":
    main()