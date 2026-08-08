from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.quality.run_stage12d_annotation_cohort_readiness import audit, inspect_manifest

ROOT = Path(__file__).resolve().parents[2]


def inputs() -> tuple[dict, dict]:
    config = json.loads(
        (ROOT / "configs/quality/stage12d_annotation_cohort_readiness.json").read_text()
    )
    stage12c = json.loads(
        (
            ROOT / "reports/stage12/stage12c_annotation_device_scope_adjudication_summary.json"
        ).read_text()
    )
    return config, stage12c


def test_stage12d_holds_without_inventing_missing_annotations(tmp_path: Path) -> None:
    config, stage12c = inputs()
    result = audit(config, stage12c, tmp_path)
    assert result["cohort_ready"] is False
    assert set(result["missing_manifests"]) == {
        "expanded_view",
        "input_rejection",
        "device_presence",
    }
    assert result["annotations_invented"] is False
    assert result["locked_test_records_accessed"] == 0


def test_manifest_rejects_patient_overlap(tmp_path: Path) -> None:
    config, _ = inputs()
    spec = config["manifests"]["expanded_view"]
    path = tmp_path / spec["filename"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec["required_columns"])
        writer.writeheader()
        base = {
            "record_key_hash": "record-a",
            "patient_key_hash": "patient-a",
            "view_label": "OTHER",
            "reviewer_role": "reviewer",
            "source_evidence": "metadata",
            "adjudication_status": "APPROVED",
            "protocol_version": "1.0.0",
        }
        writer.writerow({**base, "split": "train"})
        writer.writerow({**base, "record_key_hash": "record-b", "split": "validation"})
    result = inspect_manifest(path, spec, "1.0.0", {"train", "validation"})
    assert result["ready"] is False
    assert result["patient_split_violations"] == 1


def test_manifest_rejects_locked_test_split(tmp_path: Path) -> None:
    config, _ = inputs()
    spec = config["manifests"]["device_presence"]
    path = tmp_path / spec["filename"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec["required_columns"])
        writer.writeheader()
        writer.writerow(
            {
                "record_key_hash": "record-a",
                "patient_key_hash": "patient-a",
                "split": "test",
                "support_devices": "1",
                "source_dataset": "chexpert_small",
                "source_label": "Support Devices",
                "protocol_version": "1.0.0",
            }
        )
    result = inspect_manifest(path, spec, "1.0.0", {"train", "validation"})
    assert result["ready"] is False
    assert result["invalid_counts"]["unsupported_or_locked_split"] == 1
