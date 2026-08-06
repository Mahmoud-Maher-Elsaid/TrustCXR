from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import pydicom

LICENSE_TERMS = ("license", "licence", "terms", "readme", "citation", "credits")


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def metadata_identifiers(path: Path, column: str) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise RuntimeError(f"Identity column {column!r} is missing from {path}.")
        return {str(row[column]).strip() for row in reader if str(row[column]).strip()}


def license_evidence(root: Path, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_root = root / dataset["root"]
    rows = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or not any(term in path.name.lower() for term in LICENSE_TERMS):
            continue
        rows.append(
            {
                "dataset": dataset["name"],
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "evidence_status": "LOCAL_DOCUMENT_REQUIRES_HUMAN_REVIEW",
            }
        )
    return rows


def resolve_dicom_identity(
    root: Path,
    dataset: dict[str, Any],
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    image_root = root / dataset["image_root"]
    metadata_ids = metadata_identifiers(root / dataset["metadata"], dataset["metadata_id"])
    files = sorted(image_root.glob(f"*{dataset['image_suffix']}"))
    resolved_patients: set[str] = set()
    annotation_matches = 0
    patient_resolved = 0
    study_resolved = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in files:
            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["PatientID", "StudyInstanceUID", "SOPInstanceUID"],
            )
            raw_patient = str(getattr(header, "PatientID", "") or "").strip()
            raw_study = str(getattr(header, "StudyInstanceUID", "") or "").strip()
            raw_image = str(getattr(header, "SOPInstanceUID", "") or path.stem).strip()
            patient = (
                stable_hash(dataset["name"] + ":patient", raw_patient) if raw_patient else None
            )
            study = stable_hash(dataset["name"] + ":study", raw_study) if raw_study else None
            image = stable_hash(dataset["name"] + ":image", raw_image)
            annotation_match = path.stem in metadata_ids or raw_image in metadata_ids
            annotation_matches += int(annotation_match)
            patient_resolved += int(patient is not None)
            study_resolved += int(study is not None)
            if patient:
                resolved_patients.add(patient)
            connection.execute(
                "INSERT INTO identity_records VALUES (?, ?, ?, ?, ?, ?)",
                (dataset["name"], image, patient, study, int(annotation_match), "train_source"),
            )
    connection.commit()
    count = len(files)
    return {
        "image_count": count,
        "annotation_identifier_count": len(metadata_ids),
        "annotation_match_count": annotation_matches,
        "patient_identity_rate": patient_resolved / count if count else 0.0,
        "study_identity_rate": study_resolved / count if count else 0.0,
        "unique_patient_count": len(resolved_patients),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve Stage 10B governance and identity evidence."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        config.get("training_permitted") is not False
        or config.get("test_access_permitted") is not False
    ):
        raise RuntimeError("Stage 10B must prohibit training and test access.")
    index = root / config["identity_index"]
    index.parent.mkdir(parents=True, exist_ok=True)
    if index.exists():
        raise RuntimeError(
            f"Stage 10B identity index already exists; preserve or archive it first: {index}"
        )
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE identity_records (dataset TEXT, image_hash TEXT, patient_hash TEXT, "
        "study_hash TEXT, annotation_match INTEGER, source_split TEXT)"
    )
    results: list[dict[str, Any]] = []
    licenses: list[dict[str, Any]] = []
    try:
        for dataset in config["datasets"]:
            licenses.extend(license_evidence(root, dataset))
            identity = (
                resolve_dicom_identity(root, dataset, connection)
                if dataset.get("image_root")
                else {
                    "image_count": 0,
                    "annotation_identifier_count": len(
                        metadata_identifiers(root / dataset["metadata"], dataset["metadata_id"])
                    ),
                    "annotation_match_count": 0,
                    "patient_identity_rate": 0.0,
                    "study_identity_rate": 0.0,
                    "unique_patient_count": 0,
                }
            )
            license_status = (
                "HUMAN_APPROVED"
                if dataset["license_decision"] == "APPROVED"
                else "HUMAN_REVIEW_REQUIRED"
            )
            identity_status = (
                "PATIENT_TRACKING_RESOLVED"
                if identity["patient_identity_rate"] >= 0.99
                else "PATIENT_TRACKING_UNRESOLVED"
            )
            results.append(
                {
                    "dataset": dataset["name"],
                    **identity,
                    "identity_status": identity_status,
                    "license_status": license_status,
                    "training_ready": identity_status == "PATIENT_TRACKING_RESOLVED"
                    and license_status == "HUMAN_APPROVED",
                }
            )
    finally:
        connection.close()
    ready = sum(row["training_ready"] for row in results)
    summary = {
        "stage": "10B",
        "status": "PASSED_RESOLUTION_AUDIT",
        "gate": "GO_FOR_STAGE_10C_PATIENT_SAFE_SPLIT" if ready else "HOLD_FOR_LICENSE_OR_IDENTITY",
        "datasets_reviewed": len(results),
        "training_ready_datasets": ready,
        "license_approvals": sum(row["license_status"] == "HUMAN_APPROVED" for row in results),
        "patient_tracking_resolved": sum(
            row["identity_status"] == "PATIENT_TRACKING_RESOLVED" for row in results
        ),
        "training_permitted": False,
        "test_records_accessed": 0,
        "patient_level_rows_tracked": False,
        "identity_index_tracked": False,
    }
    reports = config["reports"]
    summary_path = root / reports["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_fields = list(results[0])
    with (root / reports["matrix"]).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fields)
        writer.writeheader()
        writer.writerows(results)
    license_fields = ["dataset", "relative_path", "sha256", "bytes", "evidence_status"]
    with (root / reports["license_evidence"]).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=license_fields)
        writer.writeheader()
        writer.writerows(licenses)
    report = "\n".join(
        [
            "# TrustCXR Stage 10B Governance and Patient Identity Resolution",
            "",
            f"- Status: `{summary['status']}`",
            f"- Gate: `{summary['gate']}`",
            f"- Training-ready datasets: `{ready}`",
            "- Training permitted by this stage: `false`",
            "- Test records accessed: `0`",
            "",
            "Local license files are evidence for human review, not automatic legal approval. "
            "Only hashed patient, study, and image identifiers are stored in the ignored "
            "SQLite index.",
            "",
        ]
    )
    (root / reports["report"]).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
