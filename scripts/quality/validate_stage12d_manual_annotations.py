from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REJECTION_CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]


def validate(package_root: Path) -> dict[str, object]:
    view_path = package_root / "01_unresolved_view_review.csv"
    rejection_path = package_root / "02_input_rejection_review.csv"
    views = list(csv.DictReader(view_path.open("r", encoding="utf-8-sig", newline="")))
    rejections = list(csv.DictReader(rejection_path.open("r", encoding="utf-8-sig", newline="")))
    errors: list[str] = []
    if len(views) != 17:
        errors.append(f"Expected 17 view rows, found {len(views)}.")
    for number, row in enumerate(views, start=2):
        if row["split"] not in {"train", "validation"}:
            errors.append(f"View row {number}: split must be train or validation.")
        if row["view_label"] not in {"OTHER", "UNKNOWN"}:
            errors.append(f"View row {number}: view_label must be OTHER or UNKNOWN.")
        for field in ("reviewer", "evidence"):
            if not row[field].strip():
                errors.append(f"View row {number}: {field} is required.")
        if row["approval_status"] != "APPROVED":
            errors.append(f"View row {number}: approval_status must be APPROVED.")
        if row["protocol_version"] != "1.0.0":
            errors.append(f"View row {number}: protocol_version must be 1.0.0.")
        if not Path(row["image_path"]).is_file():
            errors.append(f"View row {number}: image_path does not exist.")

    expected_pairs = {
        (split, label) for split in ("train", "validation") for label in REJECTION_CLASSES
    }
    observed_pairs = {(row["split"], row["rejection_class"]) for row in rejections}
    if observed_pairs != expected_pairs or len(rejections) != len(expected_pairs):
        errors.append(
            "Rejection rows must contain each class exactly once in train and validation."
        )
    groups: dict[str, set[str]] = defaultdict(set)
    for number, row in enumerate(rejections, start=2):
        if row["split"] not in {"train", "validation"}:
            errors.append(f"Rejection row {number}: locked or unsupported split.")
        for field in ("image_path_or_identifier", "group_identifier", "reviewer", "evidence"):
            if not row[field].strip():
                errors.append(f"Rejection row {number}: {field} is required.")
        if row["approval_status"] != "APPROVED":
            errors.append(f"Rejection row {number}: approval_status must be APPROVED.")
        if row["protocol_version"] != "1.0.0":
            errors.append(f"Rejection row {number}: protocol_version must be 1.0.0.")
        if row["group_identifier"].strip():
            groups[row["group_identifier"]].add(row["split"])
    overlap = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if overlap:
        errors.append(f"Group identifiers cross train/validation: {len(overlap)}.")
    result = {
        "status": "PASSED" if not errors else "FAILED",
        "protocol_version": "1.0.0",
        "view_records": len(views),
        "rejection_records": len(rejections),
        "patient_or_group_split_violations": len(overlap),
        "locked_test_records": 0,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate completed Stage 12D manual annotations.")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"),
    )
    args = parser.parse_args()
    result = validate(args.package_root.resolve())
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
