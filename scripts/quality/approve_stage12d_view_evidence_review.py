from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

REVIEW_COLUMNS = [
    "split",
    "image_path",
    "image_identifier",
    "record_key_hash",
    "patient_key_hash",
    "source_view_metadata",
    "view_label",
    "reviewer",
    "evidence",
    "approval_status",
    "protocol_version",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_review_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_manifest_atomic(
    manifest: Path, approved_rows: list[dict[str, str]], evidence: dict[str, dict[str, str]]
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest.name}.", suffix=".tmp", dir=manifest.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            manifest.open("r", encoding="utf-8-sig", newline="") as source,
            os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as target,
        ):
            reader = csv.DictReader(source)
            columns = list(reader.fieldnames or [])
            writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            existing_ids: set[str] = set()
            for row in reader:
                existing_ids.add(row["record_key_hash"])
                writer.writerow(row)
            overlap = existing_ids & {row["record_key_hash"] for row in approved_rows}
            if overlap:
                raise RuntimeError(
                    f"Expanded-view manifest already contains {len(overlap)} reviewed records."
                )
            for row in approved_rows:
                review = evidence[row["record_key_hash"]]
                writer.writerow(
                    {
                        "record_key_hash": row["record_key_hash"],
                        "patient_key_hash": row["patient_key_hash"],
                        "split": row["split"],
                        "view_label": "UNKNOWN",
                        "reviewer_role": "PROJECT_OWNER",
                        "source_evidence": (review["metadata_evidence"] + " | " + review["reason"]),
                        "adjudication_status": "APPROVED",
                        "protocol_version": "1.0.0",
                    }
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary.exists():
            temporary.unlink()


def approve(package: Path) -> dict[str, object]:
    annotation_path = package / "01_unresolved_view_review.csv"
    evidence_path = package / "stage12d_view_evidence_review_v1.0.0.csv"
    manifest_path = package.parent / "expanded_view_annotations.csv"
    annotation_backup = package / "01_unresolved_view_review.pre_evidence_approval.csv"
    manifest_backup = package.parent / "expanded_view_annotations.pre_evidence_approval.csv"
    if annotation_backup.exists() or manifest_backup.exists():
        raise RuntimeError("Approval backups already exist; refusing a repeated update.")
    annotation_before = sha256(annotation_path)
    evidence_before = sha256(evidence_path)
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    evidence = {row["record_id"]: row for row in evidence_rows}
    annotation_ids = {row["record_key_hash"] for row in annotations}
    if len(annotations) != 17 or len(evidence_rows) != 17 or annotation_ids != set(evidence):
        raise RuntimeError("The 17-record annotation and evidence sets do not match.")
    if any(row["split"] not in {"train", "validation"} for row in annotations):
        raise RuntimeError("Approval refused: locked or unsupported split detected.")
    if any(row["recommended_label"] != "UNKNOWN" for row in evidence_rows):
        raise RuntimeError(
            "Approval refused: evidence contains a recommendation other than UNKNOWN."
        )
    if any("OTHER is not justified" not in row["reason"] for row in evidence_rows):
        raise RuntimeError("Approval refused: missing no-OTHER evidence rationale.")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        existing_ids = {row["record_key_hash"] for row in csv.DictReader(handle)}
    overlap = existing_ids & annotation_ids
    if overlap:
        raise RuntimeError(
            f"Expanded-view manifest already contains {len(overlap)} reviewed records."
        )

    shutil.copy2(annotation_path, annotation_backup)
    shutil.copy2(manifest_path, manifest_backup)
    approved_rows: list[dict[str, str]] = []
    for row in annotations:
        review = evidence[row["record_key_hash"]]
        updated = dict(row)
        updated.update(
            {
                "view_label": "UNKNOWN",
                "reviewer": "PROJECT_OWNER",
                "evidence": review["metadata_evidence"] + " | " + review["reason"],
                "approval_status": "APPROVED",
                "protocol_version": "1.0.0",
            }
        )
        approved_rows.append(updated)
    write_review_atomic(annotation_path, approved_rows)
    append_manifest_atomic(manifest_path, approved_rows, evidence)
    if sha256(evidence_path) != evidence_before:
        raise RuntimeError("Evidence review changed during approval.")
    return {
        "status": "APPROVED",
        "records": 17,
        "assigned_unknown": 17,
        "assigned_other": 0,
        "reviewer": "PROJECT_OWNER",
        "protocol_version": "1.0.0",
        "locked_test_records": 0,
        "annotation_sha256_before": annotation_before,
        "annotation_sha256_after": sha256(annotation_path),
        "evidence_sha256": evidence_before,
        "backups_created": [str(annotation_backup), str(manifest_backup)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve Stage 12D evidence recommendations.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    package = (
        args.project_root.resolve()
        / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    )
    print(json.dumps(approve(package), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
