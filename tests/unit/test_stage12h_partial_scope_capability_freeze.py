from __future__ import annotations

import json
from pathlib import Path

from scripts.quality.run_stage12h_partial_scope_capability_freeze import decide


def test_stage12h_freezes_only_supported_capabilities(tmp_path: Path) -> None:
    (tmp_path / "stage12f.json").write_text(
        json.dumps(
            {
                "status": "PASSED_PARTIAL_SCOPE_EVIDENCE_FREEZE",
                "unsupported_slot_count": 9,
                "unsupported_slots": [f"slot-{index}" for index in range(9)],
                "missing_view_classes": ["OTHER"],
                "view_counts": {"UNKNOWN": {"train": 16, "validation": 1}},
                "device_scope": "IMAGE_LEVEL_PRESENCE_ONLY_NO_LOCALIZATION",
            }
        )
    )
    (tmp_path / "stage12g.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED_ADDITIONAL_DEVELOPMENT_EVIDENCE_SEARCH",
                "candidate_count": 0,
                "labels_approved": 0,
            }
        )
    )
    config = {
        "protocol_version": "1.0.0",
        "stage12f_evidence": "stage12f.json",
        "stage12g_evidence": "stage12g.json",
        "required_unsupported_rejection_slots": 9,
        "required_missing_view_classes": ["OTHER"],
        "approved_unknown_view_records": 17,
        "repeat_search_without_new_governed_source_permitted": False,
        "complete_stage12_training_permitted": False,
        "device_localization_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_results_may_be_modified": False,
    }
    result = decide(config, tmp_path)
    assert result["unsupported_rejection_slot_count"] == 9
    assert result["approved_unknown_view_records"] == 17
    assert result["repeat_search_requires_new_governed_source"] is True
    assert result["complete_stage12_training_authorized"] is False
    assert result["locked_test_records_accessed"] == 0
