from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.multiview.run_stage13a_multiview_data_readiness import audit


def test_stage13a_holds_without_study_identity(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/stage12.json").write_text(
        json.dumps({"status": "PASSED_PARTIAL_SCOPE_CAPABILITY_FREEZE"})
    )
    with (tmp_path / "views.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_key_hash",
                "patient_key_hash",
                "split",
                "view_label",
                "protocol_version",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "record_key_hash": "r1",
                    "patient_key_hash": "p1",
                    "split": "train",
                    "view_label": "PA",
                    "protocol_version": "1.0.0",
                },
                {
                    "record_key_hash": "r2",
                    "patient_key_hash": "p1",
                    "split": "train",
                    "view_label": "LATERAL",
                    "protocol_version": "1.0.0",
                },
            ]
        )
    config = {
        "stage12_freeze": "reports/stage12.json",
        "development_view_manifest": "views.csv",
        "allowed_splits": ["train", "validation"],
        "supported_view_labels": ["AP", "PA", "LATERAL", "UNKNOWN"],
        "withheld_view_labels": ["OTHER"],
        "required_study_identity_fields": ["study_key_hash", "study_id_hash"],
        "patient_identity_field": "patient_key_hash",
        "record_identity_field": "record_key_hash",
        "study_grouping_from_patient_identity_alone_permitted": False,
        "training_permitted": False,
        "inference_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_stage5_stage9_stage10_stage11_stage12_results_may_be_modified": False,
    }
    result = audit(config, tmp_path)
    assert result["decision"] == "HOLD_FOR_STUDY_LEVEL_IDENTITY_AND_PAIRING"
    assert result["patient_identity_not_used_as_study_identity"] is True
    assert result["locked_test_records_accessed"] == 0
    assert result["training_performed"] is False
