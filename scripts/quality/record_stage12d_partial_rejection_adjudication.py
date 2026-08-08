from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]
COLUMNS = [
    "split",
    "rejection_class",
    "image_path_or_identifier",
    "group_identifier",
    "reviewer",
    "evidence",
    "approval_status",
    "protocol_version",
]
ASPECT_RE = re.compile(r"aspect_ratio=([0-9.]+)")


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def record(package: Path) -> dict[str, object]:
    review_path = package / "02_input_rejection_review.csv"
    backup_path = package / "02_input_rejection_review.pre_partial_adjudication.csv"
    candidate_path = package / "stage12d_input_rejection_candidate_review_v1.0.0.csv"
    adjudication_path = package / "stage12d_input_rejection_candidate_adjudication_v1.0.0.csv"
    if backup_path.exists():
        raise RuntimeError("Partial-adjudication backup already exists; refusing repeated update.")
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(split, label) for split in ("train", "validation") for label in CLASSES}
    if len(rows) != 12 or {(row["split"], row["rejection_class"]) for row in rows} != expected:
        raise RuntimeError("The 12-slot rejection review contract changed.")
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        candidates = {row["record_id"]: row for row in csv.DictReader(handle) if row["record_id"]}
    with adjudication_path.open("r", encoding="utf-8-sig", newline="") as handle:
        adjudications = list(csv.DictReader(handle))
    accepted = [
        row
        for row in adjudications
        if row["split"] == "validation"
        and row["final_recommendation"] in {"INCOMPLETE_ANATOMY", "INADEQUATE_QUALITY"}
    ]
    incomplete = [row for row in accepted if row["final_recommendation"] == "INCOMPLETE_ANATOMY"]
    quality = [row for row in accepted if row["final_recommendation"] == "INADEQUATE_QUALITY"]
    if len(incomplete) != 3 or len(quality) != 1:
        raise RuntimeError("Unexpected defensible validation adjudication set.")

    def aspect(item: dict[str, str]) -> float:
        match = ASPECT_RE.search(item["image_geometry_evidence"])
        if match is None:
            raise RuntimeError("Missing objective aspect-ratio evidence.")
        return float(match.group(1))

    selected = {
        "INCOMPLETE_ANATOMY": max(incomplete, key=aspect),
        "INADEQUATE_QUALITY": quality[0],
    }
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        selected_evidence = (
            selected.get(row["rejection_class"]) if row["split"] == "validation" else None
        )
        if selected_evidence is not None:
            candidate = candidates[selected_evidence["record_id"]]
            updated.update(
                {
                    "image_path_or_identifier": candidate["local_path_or_identifier"],
                    "group_identifier": candidate["stable_group_identifier"],
                    "reviewer": "PROJECT_OWNER",
                    "evidence": " | ".join(
                        [
                            selected_evidence["objective_file_evidence"],
                            selected_evidence["metadata_evidence"],
                            selected_evidence["image_geometry_evidence"],
                            selected_evidence["visual_support"],
                            selected_evidence["reason"],
                        ]
                    ),
                    "approval_status": "APPROVED",
                    "protocol_version": "1.0.0",
                }
            )
        else:
            updated.update(
                {
                    "image_path_or_identifier": "",
                    "group_identifier": "",
                    "reviewer": "PROJECT_OWNER",
                    "evidence": (
                        "NO_DEFENSIBLE_EXAMPLE after current Stage 12D evidence adjudication; "
                        "slot remains incomplete and must not be promoted."
                    ),
                    "approval_status": "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE",
                    "protocol_version": "1.0.0",
                }
            )
        updated_rows.append(updated)
    shutil.copy2(review_path, backup_path)
    atomic_write(review_path, updated_rows)
    return {
        "status": "PARTIALLY_APPROVED",
        "approved_slots": [
            {"split": "validation", "rejection_class": "INCOMPLETE_ANATOMY"},
            {"split": "validation", "rejection_class": "INADEQUATE_QUALITY"},
        ],
        "selected_incomplete_anatomy_record": selected["INCOMPLETE_ANATOMY"]["record_id"],
        "selected_inadequate_quality_record": selected["INADEQUATE_QUALITY"]["record_id"],
        "incomplete_slots": 10,
        "protocol_version": "1.0.0",
        "locked_test_records": 0,
        "final_manifest_promoted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record partial Stage 12D rejection approval.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    package = (
        args.project_root.resolve()
        / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    )
    print(json.dumps(record(package), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
