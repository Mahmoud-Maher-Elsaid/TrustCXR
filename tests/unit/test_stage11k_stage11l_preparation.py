from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.fusion.run_stage11l_fusion_acceptance_decision import decide

ROOT = Path(__file__).resolve().parents[2]


def evidence(config: dict) -> dict:
    return {
        "gate": "GO_FOR_STAGE_11L_FUSION_ACCEPTANCE_DECISION",
        "shared_validation_records": 108,
        "evaluated_records": 108,
        "coverage_fraction": 1.0,
        "evidence_status_counts": {"UNCERTAIN": 106, "UNLOCALIZED": 2},
        "localization_operating_point_accepted": False,
        "localization_reliable_for_contradiction": False,
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
    }


def test_stage11l_accepts_uncertainty_annotation_only() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11l_fusion_acceptance_decision.json").read_text()
    )
    result = decide(config, evidence(config))
    assert result["decision"] == "ACCEPT_RESEARCH_FUSION_AS_UNCERTAINTY_ANNOTATION_ONLY"
    assert result["reliable_positive_support_demonstrated"] is False
    assert result["localizer_may_contradict_classifier"] is False
    assert result["maximum_support_status"] == "PARTIALLY_SUPPORTED"


def test_stage11l_rejects_changed_status_counts() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11l_fusion_acceptance_decision.json").read_text()
    )
    invalid = evidence(config)
    invalid["evidence_status_counts"] = {"PARTIALLY_SUPPORTED": 1, "UNCERTAIN": 107}
    with pytest.raises(RuntimeError):
        decide(config, invalid)
