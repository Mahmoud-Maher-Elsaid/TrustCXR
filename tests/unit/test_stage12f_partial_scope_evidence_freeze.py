from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.quality.run_stage12f_partial_scope_evidence_freeze import freeze


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_stage12f_preserves_partial_scope(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/stage12e.json").write_text(
        json.dumps({"status": "PASSED_PARTIAL_ANNOTATION_ACCEPTANCE"})
    )
    common = {"protocol_version": "1.0.0"}
    write_csv(
        tmp_path / "views.csv",
        ["split", "view_label", "protocol_version"],
        [
            {"split": "train", "view_label": "AP", **common},
            {"split": "validation", "view_label": "UNKNOWN", **common},
        ],
    )
    write_csv(
        tmp_path / "devices.csv",
        ["split", "support_devices", "protocol_version"],
        [{"split": "train", "support_devices": "1", **common}],
    )
    rejection_rows = []
    for split in ("train", "validation"):
        for label in ("CORRUPT_INPUT", "INCOMPLETE_ANATOMY"):
            approved = label == "INCOMPLETE_ANATOMY"
            rejection_rows.append(
                {
                    "split": split,
                    "rejection_class": label,
                    "approval_status": (
                        "APPROVED" if approved else "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE"
                    ),
                    **common,
                }
            )
    write_csv(
        tmp_path / "rejection.csv",
        ["split", "rejection_class", "approval_status", "protocol_version"],
        rejection_rows,
    )
    config = {
        "protocol_version": "1.0.0",
        "stage12e_evidence": "reports/stage12e.json",
        "expanded_view_manifest": "views.csv",
        "device_presence_manifest": "devices.csv",
        "input_rejection_review": "rejection.csv",
        "required_view_classes": ["AP", "PA", "LATERAL", "OTHER", "UNKNOWN"],
        "required_rejection_classes": ["CORRUPT_INPUT", "INCOMPLETE_ANATOMY"],
        "allowed_splits": ["train", "validation"],
        "unsupported_slots_may_be_promoted": False,
        "complete_model_training_permitted": False,
        "inference_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_stage5_stage9_stage10_stage11_results_may_be_modified": False,
    }
    result = freeze(config, tmp_path)
    assert result["unsupported_slot_count"] == 2
    assert result["complete_model_training_authorized"] is False
    assert result["locked_test_records_accessed"] == 0
    assert result["frozen_results_modified"] is False
