from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]


def validate(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(split, label) for split in ("train", "validation") for label in CLASSES}
    errors: list[str] = []
    if len(rows) != 12 or {(row["split"], row["rejection_class"]) for row in rows} != expected:
        errors.append("The review must contain exactly the 12 required class/split slots.")
    approved = [row for row in rows if row["approval_status"] == "APPROVED"]
    incomplete = [
        row for row in rows if row["approval_status"] == "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE"
    ]
    approved_pairs = {(row["split"], row["rejection_class"]) for row in approved}
    required_approved = {
        ("validation", "INCOMPLETE_ANATOMY"),
        ("validation", "INADEQUATE_QUALITY"),
    }
    if approved_pairs != required_approved:
        errors.append("Only the two defensible validation slots may be approved.")
    if len(incomplete) != 10:
        errors.append("Exactly ten slots must remain explicitly incomplete.")
    for row in approved:
        if (
            not row["image_path_or_identifier"]
            or not row["group_identifier"]
            or not row["evidence"]
        ):
            errors.append(f"Approved slot lacks evidence: {row['split']}/{row['rejection_class']}.")
        if not Path(row["image_path_or_identifier"]).is_file():
            errors.append(f"Approved source file is missing: {row['rejection_class']}.")
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            errors.append("Locked or unsupported split detected.")
        if row["reviewer"] != "PROJECT_OWNER" or row["protocol_version"] != "1.0.0":
            errors.append(f"Review provenance mismatch: {row['split']}/{row['rejection_class']}.")
    result = {
        "status": "PASSED" if not errors else "FAILED",
        "approved_slots": len(approved),
        "incomplete_slots": len(incomplete),
        "locked_test_records": 0,
        "final_manifest_ready": False,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate partial Stage 12D rejection review.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    path = (
        args.project_root.resolve()
        / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
        / "02_input_rejection_review.csv"
    )
    return 0 if validate(path)["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
