from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

if __package__:
    from scripts.quality.validate_stage12d_manual_annotations import validate
else:
    from validate_stage12d_manual_annotations import validate


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def apply(package: Path) -> dict[str, object]:
    validation = validate(package)
    if validation["status"] != "PASSED":
        raise RuntimeError("Manual annotations failed validation; input rejection was not applied.")
    review_path = package / "02_input_rejection_review.csv"
    manifest_path = package.parent / "input_rejection_annotations.csv"
    backup_path = package.parent / "input_rejection_annotations.pre_manual_review.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        existing = list(csv.DictReader(handle))
    if existing:
        raise RuntimeError("Input-rejection manifest is not empty; refusing to overwrite it.")
    if backup_path.exists():
        raise RuntimeError("Input-rejection backup already exists; refusing repeated application.")
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    shutil.copy2(manifest_path, backup_path)
    columns = [
        "record_key_hash",
        "patient_key_hash",
        "split",
        "primary_disposition",
        "reviewer_role",
        "source_evidence",
        "adjudication_status",
        "protocol_version",
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in reviews:
                writer.writerow(
                    {
                        "record_key_hash": stable_hash(row["image_path_or_identifier"], "record"),
                        "patient_key_hash": stable_hash(row["group_identifier"], "patient"),
                        "split": row["split"],
                        "primary_disposition": row["rejection_class"],
                        "reviewer_role": row["reviewer"],
                        "source_evidence": row["evidence"],
                        "adjudication_status": row["approval_status"],
                        "protocol_version": row["protocol_version"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "APPLIED",
        "records": len(reviews),
        "locked_test_records": 0,
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply approved Stage 12D rejection review.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    package = (
        args.project_root.resolve()
        / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    )
    print(json.dumps(apply(package), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
