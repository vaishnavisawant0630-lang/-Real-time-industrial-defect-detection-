from collections import Counter

def count_defects(result):
    """Count predicted boxes by class for one frame."""
    counts = Counter()

    if result.boxes is None:
        return {}

    for class_id in result.boxes.cls.tolist():
        class_id = int(class_id)
        class_name = result.names[class_id]
        counts[class_name] += 1

    return dict(counts)

def format_counts(counts):
    """Formate counts for display."""
    if not counts:
        return "Defect detected: 0"

    return " | ".join(
        f"{name}: {count}"
        for name,count in sorted(counts.items())
    )

