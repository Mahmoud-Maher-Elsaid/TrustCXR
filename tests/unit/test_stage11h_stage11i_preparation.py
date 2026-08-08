from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.fusion.run_stage11i_fusion_coverage_decision import decide

ROOT = Path(__file__).resolve().parents[2]


def evidence(config: dict) -> dict:
    return {
        "gate": "HOLD_FOR_STAGE_11I_FUSION_COVERAGE_DECISION",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
        "inference_split": "validation",
        "shared_validation_records": 108,
        "evaluated_overlap_records": 19,
        "coverage_fraction": 19 / 108,
        "classifier_positive_records": 0,
        "localization_operating_point_accepted": False,
        "evidence_status_counts": {"UNCERTAIN": 19},
    }


def test_stage11i_holds_incomplete_uninformative_coverage() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11i_fusion_coverage_decision.json").read_text()
    )
    result = decide(config, evidence(config))
    assert result["coverage_adequate"] is False
    assert result["positive_support_assessment_possible"] is False
    assert result["decision"] == "HOLD_FOR_SHARED_VALIDATION_PREDICTION_COVERAGE"
    assert result["maximum_support_status"] == "PARTIALLY_SUPPORTED"
    assert result["new_stage9_predictions_generated"] is False


def test_stage11i_rejects_locked_test_access() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11i_fusion_coverage_decision.json").read_text()
    )
    invalid = evidence(config)
    invalid["locked_test_records_accessed"] = 1
    with pytest.raises(RuntimeError):
        decide(config, invalid)
