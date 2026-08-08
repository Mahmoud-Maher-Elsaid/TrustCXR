from __future__ import annotations

import json
from pathlib import Path


def test_remaining_discovery_plan_preserves_safety_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/quality/stage12d_remaining_candidate_discovery_plan.json").read_text()
    )
    assert len(config["missing_slots"]["train"]) == 6
    assert len(config["missing_slots"]["validation"]) == 4
    assert config["automatic_labeling_permitted"] is False
    assert config["synthetic_examples_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["training_permitted"] is False


def test_partial_annotation_state_is_not_promoted() -> None:
    root = Path(__file__).resolve().parents[2]
    state = json.loads(
        (root / "reports/stage12/stage12d_partial_annotation_state.json").read_text()
    )
    assert state["total_input_rejection_slots"] == 12
    assert state["approved_slots"] == 3
    assert state["incomplete_no_defensible_example_slots"] == 9
    assert state["protocol_version"] == "1.0.0"
    assert state["annotations_invented"] is False
    assert state["locked_test_records_accessed"] == 0
    assert state["training_performed"] is False
    assert state["final_manifest_ready"] is False
