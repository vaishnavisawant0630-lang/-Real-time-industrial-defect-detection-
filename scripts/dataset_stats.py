from pathlib import Path
from collections import Counter
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labels"
)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def main():

    statistics = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        label_dir = LABEL_ROOT / split

        counter = Counter()

        files = list(
            label_dir.glob("*.txt")
        )

        for label_file in files:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = (
                        line.strip()
                        .split()
                    )

                    if len(parts) != 5:
                        continue

                    class_id = int(
                        parts[0]
                    )

                    counter[class_id] += 1

        statistics[split] = {
            CLASS_NAMES[class_id]: counter[class_id]
            for class_id in range(
                len(CLASS_NAMES)
            )
        }

        print()
        print(split.upper())
        print("-" * 40)

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            print(
                f"{class_name:20s}: "
                f"{counter[class_id]}"
            )

    output = (
        PROJECT_ROOT
        / "evaluation"
        / "dataset_statistics.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4
        )

    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()