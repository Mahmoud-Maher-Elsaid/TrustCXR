from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)
OUTPUT_COLUMNS = {
    "expanded_view_annotations.csv": [
        "record_key_hash",
        "patient_key_hash",
        "split",
        "view_label",
        "reviewer_role",
        "source_evidence",
        "adjudication_status",
        "protocol_version",
    ],
    "input_rejection_annotations.csv": [
        "record_key_hash",
        "patient_key_hash",
        "split",
        "primary_disposition",
        "reviewer_role",
        "source_evidence",
        "adjudication_status",
        "protocol_version",
    ],
    "device_presence_annotations.csv": [
        "record_key_hash",
        "patient_key_hash",
        "split",
        "support_devices",
        "source_dataset",
        "source_label",
        "protocol_version",
    ],
    "manual_review_queue.csv": [
        "record_key_hash",
        "patient_key_hash",
        "split",
        "review_task",
        "source_view_metadata",
        "assigned_label",
        "reviewer_role",
        "source_evidence",
        "adjudication_status",
        "protocol_version",
        "review_note",
    ],
}


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def assign_patient_split(patient_id: str) -> str:
    digest = hashlib.sha256(f"trustcxr-stage5:{patient_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.80:
        return "train"
    if fraction < 0.90:
        return "validation"
    return "test"


def trusted_view(row: dict[str, str]) -> str | None:
    frontal_lateral = (row.get("Frontal/Lateral") or "").strip().upper()
    ap_pa = (row.get("AP/PA") or "").strip().upper()
    if frontal_lateral == "LATERAL":
        return "LATERAL"
    if frontal_lateral == "FRONTAL" and ap_pa in {"AP", "PA"}:
        return ap_pa
    return None


def write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing annotation work: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def construct(config: dict[str, Any], stage12d: dict[str, Any], root: Path) -> dict[str, Any]:
    if stage12d.get("gate") != "HOLD_FOR_STAGE_12D_ANNOTATION_COHORT_CONSTRUCTION":
        raise RuntimeError("Construction requires the Stage 12D annotation hold gate.")
    if stage12d.get("protocol_version") != config["protocol_version"]:
        raise RuntimeError("Protocol version mismatch.")
    if any(
        (
            config["training_permitted"],
            config["image_access_permitted"],
            config["locked_test_output_permitted"],
            config["frozen_stage5_stage9_stage10_stage11_results_may_be_modified"],
        )
    ):
        raise RuntimeError("Construction safety contract changed.")

    dataset_root = root / config["chexpert_root"]
    csv_paths = sorted(
        {path for pattern in config["chexpert_csv_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not csv_paths:
        raise RuntimeError("CheXpert development metadata CSV files were not found.")

    expanded_rows: list[dict[str, str]] = []
    device_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    excluded_test_records = 0
    seen_records: set[str] = set()
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Path", "Frontal/Lateral", "AP/PA", config["device_source_label"]}
            if not required.issubset(set(reader.fieldnames or [])):
                raise RuntimeError(f"Required CheXpert columns are missing from {csv_path}.")
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                match = PATIENT_RE.search(raw_path)
                if not raw_path or match is None:
                    continue
                patient_id = match.group(1).lower()
                split = assign_patient_split(patient_id)
                if split not in config["development_splits"]:
                    excluded_test_records += 1
                    continue
                record_hash = stable_hash(raw_path.lower(), "record")
                if record_hash in seen_records:
                    continue
                seen_records.add(record_hash)
                patient_hash = stable_hash(patient_id, "patient")
                view = trusted_view(row)
                if view is not None:
                    expanded_rows.append(
                        {
                            "record_key_hash": record_hash,
                            "patient_key_hash": patient_hash,
                            "split": split,
                            "view_label": view,
                            "reviewer_role": "TRUSTED_METADATA",
                            "source_evidence": "CHEXPERT_VIEW_METADATA",
                            "adjudication_status": "APPROVED",
                            "protocol_version": config["protocol_version"],
                        }
                    )
                else:
                    frontal_lateral = (row.get("Frontal/Lateral") or "").strip()
                    ap_pa = (row.get("AP/PA") or "").strip()
                    metadata = f"Frontal/Lateral={frontal_lateral};AP/PA={ap_pa}"
                    review_rows.append(
                        {
                            "record_key_hash": record_hash,
                            "patient_key_hash": patient_hash,
                            "split": split,
                            "review_task": "EXPANDED_VIEW",
                            "source_view_metadata": metadata,
                            "assigned_label": "",
                            "reviewer_role": "",
                            "source_evidence": "",
                            "adjudication_status": "PENDING",
                            "protocol_version": config["protocol_version"],
                            "review_note": "",
                        }
                    )
                raw_device = (row.get(config["device_source_label"]) or "").strip()
                mapped_device = config["device_value_mapping"].get(raw_device)
                if mapped_device is not None:
                    device_rows.append(
                        {
                            "record_key_hash": record_hash,
                            "patient_key_hash": patient_hash,
                            "split": split,
                            "support_devices": mapped_device,
                            "source_dataset": "chexpert_small",
                            "source_label": "Support Devices",
                            "protocol_version": config["protocol_version"],
                        }
                    )

    output_root = root / config["output_root"]
    rows_by_file = {
        "expanded_view_annotations.csv": expanded_rows,
        "input_rejection_annotations.csv": [],
        "device_presence_annotations.csv": device_rows,
        "manual_review_queue.csv": review_rows,
    }
    existing_outputs = [filename for filename in rows_by_file if (output_root / filename).exists()]
    if existing_outputs:
        raise RuntimeError(
            "Refusing to overwrite existing annotation work: " + ", ".join(existing_outputs)
        )
    for filename, rows in rows_by_file.items():
        write_csv_atomic(output_root / filename, OUTPUT_COLUMNS[filename], rows)
    summary = {
        "stage": "12D_ANNOTATION_COHORT_CONSTRUCTION",
        "protocol_version": config["protocol_version"],
        "expanded_view_trusted_records": len(expanded_rows),
        "device_presence_trusted_records": len(device_rows),
        "manual_view_review_records": len(review_rows),
        "input_rejection_records": 0,
        "excluded_locked_test_records": excluded_test_records,
        "locked_test_records_written": 0,
        "images_accessed": 0,
        "annotations_invented": False,
        "training_performed": False,
        "next_action": "COMPLETE_MANUAL_REVIEW_AND_INPUT_REJECTION_MANIFEST",
    }
    summary_path = output_root / "construction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct Stage 12D development annotation files."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage12d = json.loads((root / config["stage12d_evidence"]).read_text(encoding="utf-8"))
    summary = construct(config, stage12d, root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
