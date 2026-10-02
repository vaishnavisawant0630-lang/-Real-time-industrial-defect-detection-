from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_ROOT = PROJECT_ROOT / "data" / "processed" / "labels"

CLASS_COUNT = 6


def validate_label_file(path):

    errors = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{path}:{line_number}: Expected 5 values"
            )
            continue

        try:
            class_id = int(parts[0])

            values = [float(x) for x in parts[1:]]

        except ValueError:
            errors.append(
                f"{path}:{line_number}: Invalid numeric value"
            )
            continue

        if not 0 <= class_id < CLASS_COUNT:
            errors.append(
                f"{path}:{line_number}: Invalid class ID {class_id}"
            )

        for value in values:

            if not 0 <= value <= 1:
                errors.append(
                    f"{path}:{line_number}: "
                    f"Coordinate outside [0,1]: {value}"
                )

    return errors


def main():

    total_files = 0
    errors = []

    for split in ["train", "val", "test"]:

        label_dir = LABEL_ROOT / split

        files = sorted(label_dir.glob("*.txt"))

        print(f"{split}: {len(files)} labels")

        for label_file in files:

            total_files += 1

            errors.extend(
                validate_label_file(label_file)
            )

    print()
    print("YOLO LABEL VALIDATION")
    print("=" * 50)

    if errors:

        print(f"FAILED: {len(errors)} errors")

        for error in errors[:100]:
            print(error)

        return

    print("PASSED: All YOLO labels are valid.")
    print(f"Total label files: {total_files}")


if __name__ == "__main__":
    main()