from pathlib import Path
import cv2
import albumentations as A
from tqdm import tqdm
import xml.etree.ElementTree as ET


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" 

RAW_ROOT = DATA_ROOT / "raw"

PROCESSED_ROOT = DATA_ROOT / "processed"

PROCESSED_IMAGES = PROCESSED_ROOT / "images"

PROCESSED_XML = (
    PROCESSED_ROOT
    / "annotations"
    / "xml"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

PROCESSED_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_XML.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND XML FILES
# ============================================================

print("=" * 70)
print("SEARCHING FOR DATASET")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data root:    {DATA_ROOT}")
print()


xml_files = list(
    RAW_ROOT.rglob("*.xml")
)


image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"
]:

    image_files.extend(
        RAW_ROOT.rglob(extension)
    )


print(
    f"XML files found   : {len(xml_files)}"
)

print(
    f"Images found      : {len(image_files)}"
)


# ============================================================
# STOP IF XML DOES NOT EXIST
# ============================================================

if len(xml_files) == 0:

    print()
    print("=" * 70)
    print("ERROR: NO XML FILES FOUND")
    print("=" * 70)

    print(
        "\nThe script searched inside:"
    )

    print(RAW_ROOT)

    print(
        "\nMake sure your XML annotation files "
        "are somewhere inside:"
    )

    print(
        RAW_ROOT
    )

    print(
        "\nExample:"
    )

    print(
        "data/raw/NEU/annotations/xml/*.xml"
    )

    print(
        "\nRun this command to locate them:"
    )

    print(
        r'Get-ChildItem -Path . -Recurse -Filter *.xml | Select-Object -First 20 FullName'
    )

    raise SystemExit(1)


# ============================================================
# IMAGE LOOKUP
# ============================================================

image_lookup = {}

for image in image_files:

    image_lookup[
        image.stem.lower()
    ] = image


# ============================================================
# AUGMENTATION
# ============================================================

transform = A.Compose(
    [
        A.HorizontalFlip(
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),

        A.Rotate(
            limit=5,
            border_mode=cv2.BORDER_REFLECT,
            p=0.3
        ),

        A.GaussNoise(
            p=0.15
        )
    ],

    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)


# ============================================================
# READ XML
# ============================================================

def read_xml(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    filename = root.findtext(
        "filename"
    )

    size = root.find("size")

    width = int(
        size.findtext("width")
    )

    height = int(
        size.findtext("height")
    )

    boxes = []

    labels = []

    for obj in root.findall("object"):

        class_name = obj.findtext(
            "name"
        )

        if class_name not in CLASSES:

            print(
                f"Unknown class "
                f"{class_name} "
                f"in {xml_path.name}"
            )

            continue

        bbox = obj.find(
            "bndbox"
        )

        xmin = float(
            bbox.findtext("xmin")
        )

        ymin = float(
            bbox.findtext("ymin")
        )

        xmax = float(
            bbox.findtext("xmax")
        )

        ymax = float(
            bbox.findtext("ymax")
        )

        boxes.append(
            [
                xmin,
                ymin,
                xmax,
                ymax
            ]
        )

        labels.append(
            class_name
        )

    return (
        filename,
        width,
        height,
        boxes,
        labels
    )


# ============================================================
# CREATE XML
# ============================================================

def create_xml(
    output_path,
    image_name,
    width,
    height,
    boxes,
    labels
):

    annotation = ET.Element(
        "annotation"
    )

    folder = ET.SubElement(
        annotation,
        "folder"
    )

    folder.text = "processed"

    filename = ET.SubElement(
        annotation,
        "filename"
    )

    filename.text = image_name

    size = ET.SubElement(
        annotation,
        "size"
    )

    width_node = ET.SubElement(
        size,
        "width"
    )

    width_node.text = str(width)

    height_node = ET.SubElement(
        size,
        "height"
    )

    height_node.text = str(height)

    depth_node = ET.SubElement(
        size,
        "depth"
    )

    depth_node.text = "3"

    segmented = ET.SubElement(
        annotation,
        "segmented"
    )

    segmented.text = "0"

    for box, class_name in zip(
        boxes,
        labels
    ):

        xmin, ymin, xmax, ymax = box

        obj = ET.SubElement(
            annotation,
            "object"
        )

        name = ET.SubElement(
            obj,
            "name"
        )

        name.text = class_name

        pose = ET.SubElement(
            obj,
            "pose"
        )

        pose.text = "Unspecified"

        truncated = ET.SubElement(
            obj,
            "truncated"
        )

        truncated.text = "0"

        difficult = ET.SubElement(
            obj,
            "difficult"
        )

        difficult.text = "0"

        bndbox = ET.SubElement(
            obj,
            "bndbox"
        )

        for tag, value in [
            ("xmin", xmin),
            ("ymin", ymin),
            ("xmax", xmax),
            ("ymax", ymax)
        ]:

            node = ET.SubElement(
                bndbox,
                tag
            )

            node.text = str(
                int(round(value))
            )

    ET.indent(
        annotation,
        space="    "
    )

    tree = ET.ElementTree(
        annotation
    )

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )


# ============================================================
# PROCESS
# ============================================================

processed = 0
skipped = 0
failed = 0


print()
print("=" * 70)
print("PROCESSING DATASET")
print("=" * 70)


for xml_path in tqdm(
    xml_files,
    desc="Processing"
):

    try:

        (
            filename,
            original_width,
            original_height,
            boxes,
            labels
        ) = read_xml(xml_path)


        # ----------------------------------------------------
        # Find matching image
        # ----------------------------------------------------

        image_path = None

        if filename:

            filename_stem = Path(
                filename
            ).stem.lower()

            image_path = image_lookup.get(
                filename_stem
            )

        if image_path is None:

            image_path = image_lookup.get(
                xml_path.stem.lower()
            )


        if image_path is None:

            print(
                f"\nImage not found for:"
                f" {xml_path.name}"
            )

            skipped += 1

            continue


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"\nCannot read:"
                f" {image_path}"
            )

            skipped += 1

            continue


        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Apply transformation
        # ----------------------------------------------------

        result = transform(
            image=image_rgb,
            bboxes=boxes,
            class_labels=labels
        )


        processed_image = result[
            "image"
        ]

        processed_boxes = result[
            "bboxes"
        ]

        processed_labels = result[
            "class_labels"
        ]


        # ----------------------------------------------------
        # Convert back to OpenCV format
        # ----------------------------------------------------

        processed_image = cv2.cvtColor(
            processed_image,
            cv2.COLOR_RGB2BGR
        )


        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_image_name = (
            image_path.stem + ".jpg"
        )

        output_xml_name = (
            image_path.stem + ".xml"
        )


        output_image = (
            PROCESSED_IMAGES
            / output_image_name
        )

        output_xml = (
            PROCESSED_XML
            / output_xml_name
        )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_image),
            processed_image
        )


        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        height, width = (
            processed_image.shape[:2]
        )


        # ----------------------------------------------------
        # Save annotation
        # ----------------------------------------------------

        create_xml(
            output_xml,
            output_image_name,
            width,
            height,
            processed_boxes,
            processed_labels
        )


        processed += 1


    except Exception as e:

        print(
            f"\nFAILED:"
            f" {xml_path.name}"
        )

        print(e)

        failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    f"XML input       : {len(xml_files)}"
)

print(
    f"Images found    : {len(image_files)}"
)

print(
    f"Processed       : {processed}"
)

print(
    f"Skipped         : {skipped}"
)

print(
    f"Failed          : {failed}"
)

print()
print(
    "Processed images:"
)

print(
    PROCESSED_IMAGES
)

print()
print(
    "Processed XML:"
)

print(
    PROCESSED_XML
)