from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.multiview.run_stage13b_study_identity_resolution import resolve, stable


def test_stage13b_uses_explicit_study_structure(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/stage13a.json").write_text(
        json.dumps({"gate": "HOLD_FOR_STAGE_13B_STUDY_IDENTITY_RESOLUTION"})
    )
    raw = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    record = stable(raw.lower(), "record")
    patient = stable("patient00001", "patient")
    with (tmp_path / "views.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_key_hash", "patient_key_hash", "split", "view_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_key_hash": record,
                "patient_key_hash": patient,
                "split": "train",
                "view_label": "PA",
            }
        )
    metadata = tmp_path / "chexpert/archive"
    metadata.mkdir(parents=True)
    with (metadata / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Path"])
        writer.writeheader()
        writer.writerow({"Path": raw})
    config = {
        "stage13a_evidence": "reports/stage13a.json",
        "development_view_manifest": "views.csv",
        "chexpert_root": "chexpert",
        "chexpert_csv_patterns": ["**/train.csv"],
        "output_index": "output.sqlite",
        "allowed_splits": ["train", "validation"],
        "study_path_pattern": "(?i)(patient[0-9]+)/(study[0-9]+)",
        "patient_identity_as_study_identity_permitted": False,
        "heuristic_view_pairing_permitted": False,
        "image_access_permitted": False,
        "training_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_results_may_be_modified": False,
    }
    result = resolve(config, tmp_path)
    assert result["status"] == "PASSED_STUDY_IDENTITY_RESOLUTION"
    assert result["resolved_records"] == 1
    assert result["patient_identity_used_as_study_identity"] is False
    assert result["heuristic_pairs_created"] == 0
    assert result["locked_test_records_accessed"] == 0
