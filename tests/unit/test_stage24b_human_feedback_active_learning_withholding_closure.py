from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.feedback.run_stage24b_human_feedback_active_learning_withholding_closure import (
    decide_withholding,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (
        ROOT / "configs/feedback/stage24b_human_feedback_active_learning_withholding_closure.json"
    ).read_text(encoding="utf-8")
)


def test_stage24b_accepts_exact_stage24a_evidence() -> None:
    summary = ROOT / CONFIG["stage24a_summary"]
    assert sha256(summary) == CONFIG["stage24a_summary_sha256"]
    result = decide_withholding(CONFIG, ROOT)
    assert result["status"] == "SCIENTIFICALLY_WITHHELD_NO_GOVERNED_EXPERT_FEEDBACK_DATA"
    assert result["closure_classification"] == "WITHHELD_NOT_FAILED"
    assert result["stage24_closed"] is True


def test_stage24b_preserves_evidence_distinctions() -> None:
    distinctions = CONFIG["evidence_distinctions"]
    assert distinctions == {
        "dataset_ground_truth": "NOT_PROSPECTIVE_HUMAN_FEEDBACK",
        "stage12_manual_project_annotations": (
            "STAGE_SPECIFIC_ONLY_NOT_REUSABLE_AS_STAGE24_FEEDBACK"
        ),
        "semantic_governance_adjudication": "NOT_CLINICAL_EXPERT_FEEDBACK",
        "synthetic_fixtures": "NOT_HUMAN_FEEDBACK",
        "model_outputs": "NOT_HUMAN_FEEDBACK",
    }


def test_stage24b_future_feedback_schema_is_design_only() -> None:
    design = CONFIG["future_feedback_design"]
    assert design["status"] == "DESIGN_ONLY_NOT_ACTIVATED"
    assert "PSEUDONYMOUS_CASE_REFERENCE" in design["allowed_structured_fields"]
    assert "ORIGINAL_STRUCTURED_RESULT_IMMUTABLE_REFERENCE" in design["allowed_structured_fields"]
    assert {
        "REVIEWER_NAME",
        "PATIENT_IDENTIFIER",
        "RAW_PHI",
        "FREE_TEXT_CLINICAL_DIAGNOSIS",
    } <= set(design["prohibited_fields"])
    assert design["human_feedback_does_not_automatically_become_training_truth"] is True


def test_stage24b_future_training_requires_all_governance_gates() -> None:
    gates = set(CONFIG["future_training_feedback_gate"])
    assert gates == {
        "GOVERNED_REVIEWER_QUALIFICATION",
        "EXPLICIT_ANNOTATION_PROTOCOL",
        "ADJUDICATION_POLICY_WHERE_REQUIRED",
        "COMPLETE_PROVENANCE",
        "VERSIONED_REVIEW_MANIFEST",
        "PATIENT_SAFE_SPLIT_PROTECTION",
        "LOCKED_TEST_PROTECTION",
        "EXPLICIT_RETRAINING_MANIFEST_APPROVAL",
    }


def test_stage24b_preserves_locked_test_protection() -> None:
    assert set(CONFIG["test_set_policy"]) == {
        "NO_LOCKED_TEST_FEEDBACK_ACQUISITION",
        "NO_TEST_CORRECTIONS",
        "NO_TEST_RELABELING",
        "NO_ACTIVE_LEARNING_FROM_LOCKED_TEST",
        "NO_FEEDBACK_DRIVEN_TEST_TUNING",
    }


def test_stage24b_candidate_signals_remain_design_only() -> None:
    assert set(CONFIG["candidate_signals_design_only"]) == {
        "STAGE16_PREDICTIVE_UNCERTAINTY",
        "STAGE17_DEFER_REASON_CODES",
        "STAGE19_VERIFIER_STATUS",
        "STAGE20_DECISION",
        "GOVERNED_MODEL_DISAGREEMENT",
    }
    assert "STAGE13_SELECTIVE_PREDICTION" in CONFIG["prohibited_signals"]
    assert "OOD_CLAIM" in CONFIG["prohibited_signals"]


@pytest.mark.parametrize(
    "field",
    [
        "active_learning_queue_created",
        "feedback_collection_authorized",
        "feedback_driven_sampling_authorized",
        "active_learning_activation_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "dataset_split_modification_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
    ],
)
def test_stage24b_rejects_scope_expansion(field: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[field] = True
    with pytest.raises(RuntimeError, match=field):
        decide_withholding(changed, ROOT)


def test_stage24b_closure_has_no_operational_side_effects() -> None:
    result = decide_withholding(CONFIG, ROOT)
    assert result["active_learning_active"] is False
    assert result["active_learning_queue_created"] is False
    assert result["feedback_collected"] is False
    assert result["training_performed"] is False
    assert result["fine_tuning_performed"] is False
    assert result["checkpoints_modified"] is False
    assert result["dataset_splits_modified"] is False
    assert result["locked_test_records_accessed"] == 0


def test_stage24b_handoff_is_shortest_mandatory_path() -> None:
    result = decide_withholding(CONFIG, ROOT)
    assert result["next_canonical_stage"] == "25A_MLOPS_REPRODUCIBILITY_DATA_READINESS"
    assert result["remaining_major_mandatory_conceptual_stage_count"] == 3
    assert result["remaining_mandatory_conceptual_stages_after_stage24"] == [
        "ORIGINAL_STAGE22_MLOPS_AND_REPRODUCIBILITY",
        "ORIGINAL_STAGE23_EXTERNAL_VALIDATION_AND_ABLATIONS",
        "ORIGINAL_STAGE24_PAPER_AND_FINAL_RELEASE",
    ]
    assert result["optional_capability_expansion_required"] is False


def test_stage24b_no_llm_gate_exists() -> None:
    result = decide_withholding(CONFIG, ROOT)
    assert result["language_model_used"] is False
    assert result["language_model_endpoint_prepared"] is False
    assert result["currently_planned_llm_authorized_gate"] is None
    assert result["language_model_mandatory_for_project_completion"] is False
