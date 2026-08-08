from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.fusion.run_stage11g_fusion_implementation import (
    evaluate_contract_cases,
    validate_upstream,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage11g_contract_matrix_preserves_disagreement() -> None:
    config = json.loads((ROOT / "configs/fusion/stage11g_fusion_implementation.json").read_text())
    results = evaluate_contract_cases(config["contract_cases"])
    statuses = {row["name"]: row["status"] for row in results}
    assert statuses["classifier_and_localizer_agree"] == "PARTIALLY_SUPPORTED"
    assert statuses["classifier_positive_without_reliable_localization"] == "UNLOCALIZED"
    assert statuses["localizer_positive_classifier_negative"] == "CONTRADICTED"
    assert statuses["outside_image_geometry"] == "OUTSIDE_EXPECTED_ANATOMY"
    assert "SUPPORTED" not in statuses.values()


def test_stage11g_requires_patient_safe_locked_test_contract() -> None:
    config = json.loads((ROOT / "configs/fusion/stage11g_fusion_implementation.json").read_text())
    stage11f = {
        "gate": "GO_FOR_STAGE_11G_FUSION_IMPLEMENTATION_PREPARATION",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "semantic_relation": config["semantic_relation"],
        "permitted_evidence_status": config["maximum_support_status"],
        "downstream_evidence_policy": config["downstream_evidence_policy"],
    }
    validate_upstream(config, stage11f)
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
    broken = dict(stage11f, patient_split_violations=1)
    with pytest.raises(RuntimeError):
        validate_upstream(config, broken)
