from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.quality.run_stage12e_partial_annotation_acceptance import evaluate


def test_stage12e_accepts_partial_state_without_forcing_completion(tmp_path: Path) -> None:
    (tmp_path / "reports/stage12").mkdir(parents=True)
    (tmp_path / "artifacts/review").mkdir(parents=True)
    (tmp_path / "reports/stage12/state.json").write_text(
        json.dumps({"status": "PARTIAL_ANNOTATION_STATE_VALIDATED"})
    )
    fields = [
        "split",
        "rejection_class",
        "image_path_or_identifier",
        "group_identifier",
        "reviewer",
        "evidence",
        "approval_status",
        "protocol_version",
    ]
    approved = {
        "train/INCOMPLETE_ANATOMY",
        "validation/INCOMPLETE_ANATOMY",
        "validation/INADEQUATE_QUALITY",
    }
    classes = [
        "CORRUPT_INPUT",
        "UNSUPPORTED_FORMAT",
        "NON_CHEST_INPUT",
        "INCOMPLETE_ANATOMY",
        "INADEQUATE_QUALITY",
        "UNSUPPORTED_VIEW",
    ]
    with (tmp_path / "artifacts/review/manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "validation"):
            for label in classes:
                key = f"{split}/{label}"
                writer.writerow(
                    {
                        "split": split,
                        "rejection_class": label,
                        "reviewer": "PROJECT_OWNER",
                        "evidence": "objective evidence",
                        "approval_status": (
                            "APPROVED" if key in approved else "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE"
                        ),
                        "protocol_version": "1.0.0",
                    }
                )
    config = {
        "protocol_version": "1.0.0",
        "upstream_state": "reports/stage12/state.json",
        "review_manifest": "artifacts/review/manifest.csv",
        "required_total_rows": 12,
        "required_approved_rows": 3,
        "required_incomplete_rows": 9,
        "approved_scope": sorted(approved),
        "unsupported_classes_may_be_forced_complete": False,
        "training_permitted": False,
        "inference_permitted": False,
        "locked_test_access_permitted": False,
    }
    result = evaluate(config, tmp_path)
    assert result["approved_rows"] == 3
    assert result["incomplete_no_defensible_example_rows"] == 9
    assert result["complete_stage12_training_authorized"] is False
    assert result["locked_test_records_accessed"] == 0
