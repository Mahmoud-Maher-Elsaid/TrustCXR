from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/feedback/stage24a_human_feedback_active_learning_data_readiness.json"
SCRIPT_PATH = (
    ROOT / "scripts/feedback/run_stage24a_human_feedback_active_learning_data_readiness.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("stage24a", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage23d_evidence_and_hash_are_frozen() -> None:
    config = load_config()
    summary = ROOT / config["stage23d_summary"]
    assert load_module().sha256(summary) == config["stage23d_summary_sha256"]
    evidence = json.loads(summary.read_text(encoding="utf-8"))
    assert evidence["status"] == "ACCEPTED_SYNTHETIC_DICOM_INTEROPERABILITY_RESEARCH_ONLY"
    assert evidence["gate"] == "GO_FOR_STAGE_24A_HUMAN_FEEDBACK_ACTIVE_LEARNING_DATA_READINESS"


def test_numbering_reconciliation_preserves_both_numbering_systems() -> None:
    numbering = load_config()["numbering_reconciliation"]
    assert numbering["actual_repository_stages_completed"] == [21, 22, 23]
    assert numbering["original_conceptual_stages_remaining"] == [21, 22, 23, 24]
    assert numbering["major_mandatory_stage_count_remaining"] == 4
    assert numbering["history_renumbered_or_replaced"] is False


def test_all_audit_evidence_paths_exist() -> None:
    audit = load_config()["evidence_audit"]
    for category in (
        "dataset_ground_truth",
        "manual_project_annotations",
        "semantic_governance_adjudication",
        "synthetic_fixtures",
        "model_outputs",
    ):
        assert audit[category]["evidence"]
        for path in audit[category]["evidence"]:
            assert (ROOT / path).is_file(), path


def test_evidence_categories_are_not_promoted_to_expert_feedback() -> None:
    audit = load_config()["evidence_audit"]
    assert audit["true_governed_expert_feedback"]["status"] == "ABSENT"
    assert audit["dataset_ground_truth"]["status"] == "AVAILABLE_NOT_PROSPECTIVE_FEEDBACK"
    assert audit["synthetic_fixtures"]["status"] == "NOT_HUMAN_FEEDBACK"
    assert audit["model_outputs"]["status"] == "NOT_HUMAN_FEEDBACK"


def test_manual_annotations_remain_stage_specific() -> None:
    manual = load_config()["evidence_audit"]["manual_project_annotations"]
    assert manual["status"] == "STAGE_SPECIFIC_ONLY_NOT_REUSABLE_AS_STAGE24_FEEDBACK"
    assert "clinical-expert qualification" in manual["reason"]


def test_feedback_record_contract_excludes_identity_and_free_diagnosis() -> None:
    readiness = load_config()["future_feedback_record_readiness"]
    assert readiness["status"] == "DESIGN_READY_NOT_ACTIVATED"
    assert "PSEUDONYMOUS_CASE_REFERENCE" in readiness["allowed_structured_fields"]
    assert {
        "REVIEWER_NAME",
        "PATIENT_IDENTIFIER",
        "RAW_PHI",
        "FREE_TEXT_CLINICAL_DIAGNOSIS",
    } <= set(readiness["prohibited_fields"])
    assert readiness["original_output_immutable"] is True


def test_allowed_feedback_purposes_are_research_only() -> None:
    purposes = set(load_config()["allowed_research_feedback_purposes"])
    assert "MODEL_SIGNAL_INCORRECT" in purposes
    assert "REPORT_WORDING_ISSUE" in purposes
    assert "DEFER_APPROPRIATE_OR_INAPPROPRIATE" in purposes
    assert "CLINICAL_DIAGNOSIS" not in purposes


def test_active_learning_is_withheld_and_uses_only_eligible_signals() -> None:
    readiness = load_config()["active_learning_readiness"]
    assert readiness["status"] == "SCIENTIFICALLY_WITHHELD_NO_GOVERNED_EXPERT_FEEDBACK_DATA"
    assert readiness["activation_authorized"] is False
    assert "STAGE16_PREDICTIVE_UNCERTAINTY" in readiness["candidate_signals_design_ready"]
    assert "STAGE13_SELECTIVE_PREDICTION" in readiness["prohibited_signals"]
    assert "OOD_CLAIM" in readiness["prohibited_signals"]


def test_feedback_never_automatically_becomes_training_truth() -> None:
    config = load_config()
    assert (
        config["future_feedback_record_readiness"]["feedback_automatically_becomes_training_truth"]
        is False
    )
    required = set(config["future_training_feedback_gate"])
    assert "GOVERNED_REVIEWER_QUALIFICATION" in required
    assert "PATIENT_SAFE_SPLIT_PROTECTION" in required
    assert "LOCKED_TEST_PROTECTION" in required


def test_locked_test_protections_are_explicit() -> None:
    policy = set(load_config()["test_set_policy"])
    assert policy == {
        "NO_LOCKED_TEST_FEEDBACK_ACQUISITION",
        "NO_TEST_CORRECTIONS",
        "NO_TEST_RELABELING",
        "NO_ACTIVE_LEARNING_FROM_LOCKED_TEST",
        "NO_FEEDBACK_DRIVEN_TEST_TUNING",
    }


@pytest.mark.parametrize(
    "field",
    [
        "governed_expert_feedback_exists",
        "existing_manual_annotations_reusable_for_stage24_feedback",
        "active_learning_activation_authorized",
        "training_authorized",
        "checkpoint_modification_authorized",
        "dataset_split_change_authorized",
        "new_patient_data_processing_authorized",
        "locked_test_access_authorized",
        "patient_identifier_exposure_authorized",
        "heuristic_patient_join_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
        "language_model_mandatory_for_project_completion",
    ],
)
def test_prohibited_authorization_fails_closed(field: str) -> None:
    config = copy.deepcopy(load_config())
    config[field] = True
    with pytest.raises(RuntimeError, match=field):
        load_module().audit_readiness(config, ROOT)


def test_audit_returns_withheld_readiness_without_side_effects(tmp_path: Path) -> None:
    config = load_config()
    result = load_module().audit_readiness(config, ROOT)
    assert result["status"] == config["expected_status"]
    assert result["gate"] == config["expected_gate"]
    assert result["active_learning_activation_authorized"] is False
    assert result["locked_test_records_accessed"] == 0
    assert result["training_performed"] is False
    assert not list(tmp_path.iterdir())


def test_shortest_closure_path_and_next_stage_are_frozen() -> None:
    config = load_config()
    assert (
        config["next_canonical_stage"] == "24B_HUMAN_FEEDBACK_ACTIVE_LEARNING_WITHHOLDING_DECISION"
    )
    assert config["stage24_may_close_with_scientifically_withheld_disposition"] is True
    assert len(config["shortest_mandatory_path"]) == 4
    assert config["optional_capability_expansion_may_be_omitted"] is True
    assert config["currently_planned_llm_authorized_gate"] is None
    assert config["language_model_mandatory_for_project_completion"] is False
