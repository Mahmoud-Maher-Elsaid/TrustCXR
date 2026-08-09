from __future__ import annotations

import json
from pathlib import Path

from scripts.external_validation.run_stage26b_external_validation_withholding_closure import (
    close_external_validation,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (
        ROOT / "configs/external_validation/stage26b_external_validation_withholding_closure.json"
    ).read_text(encoding="utf-8")
)


def test_stage26b_accepts_exact_stage26a_evidence() -> None:
    summary = ROOT / CONFIG["stage26a_summary"]
    assert sha256(summary) == CONFIG["stage26a_summary_sha256"]
    evidence = json.loads(summary.read_text(encoding="utf-8"))
    assert evidence["status"] == CONFIG["stage26a_status"]
    assert evidence["disposition"] == CONFIG["disposition"]
    assert evidence["closure_classification"] == "WITHHELD_NOT_FAILED"


def test_strict_external_validation_definition_is_unchanged() -> None:
    assert CONFIG["strict_external_validation_definition"] == [
        "INDEPENDENT_INSTITUTION_OR_SOURCE",
        "INDEPENDENT_DATASET_AND_COHORT_CONSTRUCTION",
        "PROVEN_NO_PATIENT_OVERLAP",
        "COMPATIBLE_GOVERNED_LABEL_SEMANTICS",
        "NO_TRAINING_VALIDATION_SELECTION_THRESHOLD_CALIBRATION_OR_POST_TEST_USE",
        "GOVERNED_LICENSE_IDENTITY_IMAGES_AND_LABELS",
    ]


def test_all_candidate_outcomes_remain_frozen_and_ineligible() -> None:
    evidence = json.loads((ROOT / CONFIG["stage26a_summary"]).read_text(encoding="utf-8"))
    assert CONFIG["candidate_outcomes_frozen_from_stage26a"] is True
    assert len(evidence["candidate_datasets"]) == CONFIG["candidate_dataset_count"] == 10
    assert evidence["eligible_candidate_datasets"] == []
    assert evidence["externally_validatable_target_components"] == []
    assert CONFIG["eligible_candidate_dataset_count"] == 0
    assert CONFIG["externally_validatable_target_component_count"] == 0


def test_all_documented_blocker_categories_are_preserved() -> None:
    assert set(CONFIG["preserved_blocker_categories"]) == {
        "PRIOR_MODEL_DEVELOPMENT_USE",
        "PATIENT_INDEPENDENCE_NOT_PROVEN",
        "INCOMPATIBLE_LABEL_SEMANTICS",
        "UNRESOLVED_PATIENT_IDENTITY",
        "UNRESOLVED_ANNOTATION_IMAGE_JOIN",
        "UNRESOLVED_LICENSE_WHERE_APPLICABLE",
    }


def test_capability_specific_withholding_is_exact() -> None:
    assert CONFIG["capability_withholding"] == {
        "stage5_view_technical_quality_proxy": "SCIENTIFICALLY_WITHHELD",
        "stage9_full_14_label_classifier": "SCIENTIFICALLY_WITHHELD",
        "stage9_partial_label_scope": "SCIENTIFICALLY_WITHHELD_IDENTITY_INDEPENDENCE_NOT_PROVEN",
        "stage10_localization_research_baseline": "SCIENTIFICALLY_WITHHELD",
        "stage13_frontal_classifier": "SCIENTIFICALLY_WITHHELD",
        "stage16_reliability_calibration": "SCIENTIFICALLY_WITHHELD",
        "stages17_to_20_deterministic_downstream": (
            "SCIENTIFICALLY_WITHHELD_UPSTREAM_EXTERNAL_EVIDENCE_ABSENT"
        ),
    }


def test_future_evidence_does_not_block_current_closure() -> None:
    assert CONFIG["future_external_dataset_required"] is True
    assert CONFIG["new_dataset_acquisition_required_for_current_closure"] is False
    assert (
        "GOVERNED_INDEPENDENT_PATIENT_IDENTITY_OR_PROVEN_NON_OVERLAP"
        in CONFIG["future_required_evidence"]
    )
    assert (
        "PROSPECTIVELY_FROZEN_COHORT_AND_METRICS_BEFORE_INFERENCE"
        in CONFIG["future_required_evidence"]
    )


def test_stage25_environment_lock_remains_frozen() -> None:
    lock = ROOT / CONFIG["canonical_environment_lock"]
    assert sha256(lock) == CONFIG["canonical_environment_lock_sha256"]
    assert CONFIG["canonical_environment_lock_sha256"] == (
        "cc63ac8bfb8dd6cc0f15469c4e7dfd6f620ec3747931ebd63c85fb11a8dc0786"
    )


def test_release_claim_is_explicit_and_prohibited_claims_are_complete() -> None:
    assert CONFIG["final_release_external_validation_statement"] == (
        "EXTERNAL_VALIDATION_NOT_PERFORMED"
    )
    assert set(CONFIG["prohibited_release_claims"]) == {
        "INDEPENDENT_CLINICAL_VALIDATION",
        "MULTI_INSTITUTION_VALIDATION",
        "DEPLOYMENT_VALIDATION",
        "PROSPECTIVE_VALIDATION",
        "CLINICAL_GENERALIZABILITY",
    }


def test_material_prior_limitations_remain_frozen() -> None:
    limitations = set(CONFIG["frozen_limitations"])
    assert {
        "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
        "NO_LOCALIZATION_ABSENCE_CONTRADICTION",
        "STAGE13_SELECTIVE_PREDICTION_NOT_ACCEPTED",
        "OOD_WITHHELD",
        "TEMPORAL_CHANGE_WITHHELD",
        "SEVERITY_WITHHELD",
        "STAGE17_DEFER_ONLY",
        "STAGE18_DETERMINISTIC_REPORTING_ONLY",
        "STAGE19_VERIFIER_RESTRICTIONS",
        "STAGE20_DEFER_HIGHEST_PRECEDENCE",
        "STAGE23_SYNTHETIC_ONLY_DICOM",
        "STAGE24_ACTIVE_LEARNING_WITHHELD",
        "STAGE25_WINDOWS_SCOPE_REPRODUCIBILITY_ONLY",
    } <= limitations


def test_stage26b_prohibits_execution_and_llm_scope() -> None:
    for field in (
        "new_dataset_acquisition_authorized",
        "new_patient_processing_authorized",
        "identity_matching_authorized",
        "model_inference_authorized",
        "prediction_generation_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "calibration_fitting_authorized",
        "threshold_tuning_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "retraining_required",
        "optional_capability_expansion_required",
    ):
        assert CONFIG[field] is False
    assert CONFIG["currently_planned_llm_authorized_gate"] is None


def test_stage26b_closure_result_is_withheld_not_failed() -> None:
    result = close_external_validation(CONFIG, ROOT)
    assert result["status"] == CONFIG["disposition"]
    assert result["closure_classification"] == "WITHHELD_NOT_FAILED"
    assert result["stage26_closed"] is True
    assert result["external_validation_performed"] is False
    assert result["model_inference_performed"] is False
    assert result["predictions_generated"] is False
    assert result["locked_test_records_accessed"] == 0


def test_stage26b_handoff_goes_directly_to_final_release() -> None:
    assert CONFIG["next_canonical_stage"] == "27A_PAPER_FINAL_RELEASE_AUDIT"
    assert CONFIG["remaining_mandatory_path"] == ["STAGE27A_PAPER_FINAL_RELEASE_AUDIT"]
    assert CONFIG["major_conceptual_stages_remaining_after_stage26"] == 1
