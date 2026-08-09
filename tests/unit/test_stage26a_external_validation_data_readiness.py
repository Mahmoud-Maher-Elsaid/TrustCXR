from __future__ import annotations

import json
from pathlib import Path

from scripts.external_validation.run_stage26a_external_validation_data_readiness import (
    ALLOWED_STATUSES,
    audit_readiness,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (
        ROOT / "configs/external_validation/stage26a_external_validation_data_readiness.json"
    ).read_text(encoding="utf-8")
)


def test_stage26a_accepts_exact_stage25b_closure() -> None:
    summary = ROOT / CONFIG["stage25b_summary"]
    assert sha256(summary) == CONFIG["stage25b_summary_sha256"]
    evidence = json.loads(summary.read_text(encoding="utf-8"))
    assert evidence["status"] == "ACCEPTED_REPRODUCIBLE_RESEARCH_PIPELINE_WINDOWS_SCOPE"
    assert evidence["stage25_closed"] is True
    assert evidence["all_accepted_checkpoints_integrity_valid"] is True


def test_stage26a_preserves_final_environment_contract() -> None:
    lock = ROOT / CONFIG["canonical_environment_lock"]
    assert sha256(lock) == CONFIG["canonical_environment_lock_sha256"]
    assert CONFIG["canonical_environment_lock_sha256"] == (
        "cc63ac8bfb8dd6cc0f15469c4e7dfd6f620ec3747931ebd63c85fb11a8dc0786"
    )


def test_all_ten_governed_dataset_roots_are_audited() -> None:
    catalog = json.loads((ROOT / "configs/data/dataset_catalog.json").read_text(encoding="utf-8"))
    assert len(catalog["datasets"]) == len(CONFIG["candidate_datasets"]) == 10
    assert {item["name"] for item in catalog["datasets"]} == {
        item["dataset"] for item in CONFIG["candidate_datasets"]
    }
    for dataset in catalog["datasets"]:
        assert (ROOT / "TrustCXR-Data" / dataset["folder"]).is_dir()


def test_candidate_statuses_use_frozen_vocabulary_and_none_are_eligible() -> None:
    for candidate in CONFIG["candidate_datasets"]:
        assert candidate["status"] in ALLOWED_STATUSES
        assert candidate["status"] not in {
            "ELIGIBLE_FOR_BOUNDED_EXTERNAL_VALIDATION",
            "PARTIALLY_ELIGIBLE_LIMITED_LABEL_SCOPE",
        }
        assert candidate["reasons"]
    assert CONFIG["eligible_candidate_datasets"] == []
    assert CONFIG["externally_validatable_target_components"] == []


def test_development_datasets_are_not_called_external() -> None:
    candidates = {item["dataset"]: item for item in CONFIG["candidate_datasets"]}
    assert candidates["NIH ChestXray14"]["status"] == "INELIGIBLE_NOT_INDEPENDENT"
    assert candidates["NIH CheXmask"]["status"] == "INELIGIBLE_NOT_INDEPENDENT"
    assert candidates["RSNA Pneumonia Detection Challenge"]["status"] == (
        "INELIGIBLE_NOT_INDEPENDENT"
    )
    assert candidates["CheXpert Small"]["status"] == "INELIGIBLE_NOT_INDEPENDENT"


def test_previously_withheld_identity_and_license_decisions_remain_active() -> None:
    candidates = {item["dataset"]: item for item in CONFIG["candidate_datasets"]}
    for dataset in (
        "VinBigData Chest X-ray Abnormalities Detection",
        "Indiana University Chest X-ray Reports",
        "SIIM-ACR Pneumothorax Segmentation",
        "TBX11K",
        "COVID-19 Radiography Database",
    ):
        assert candidates[dataset]["status"] == "INELIGIBLE_IDENTITY_NOT_GOVERNED"
    assert candidates["Chest Radiography Database Lung Masks"]["status"] == (
        "INELIGIBLE_LABEL_INCOMPATIBILITY"
    )
    assert (
        "INHERITED_LICENSE_AND_PATIENT_IDENTITY_REMAIN_UNRESOLVED"
        in candidates["Chest Radiography Database Lung Masks"]["reasons"]
    )


def test_stage9_full_and_partial_external_validation_are_withheld() -> None:
    assert CONFIG["stage9_full_14_label_external_validation_possible"] is False
    assert CONFIG["capability_readiness"]["stage9_full_14_label_classifier"] == (
        "SCIENTIFICALLY_WITHHELD"
    )
    assert CONFIG["capability_readiness"]["stage9_partial_label_scope"] == (
        "SCIENTIFICALLY_WITHHELD_IDENTITY_INDEPENDENCE_NOT_PROVEN"
    )


def test_stage13_stage10_and_reliability_external_validation_are_withheld() -> None:
    assert CONFIG["stage13_external_validation_possible"] is False
    assert CONFIG["stage10_localization_external_validation_possible"] is False
    assert CONFIG["capability_readiness"]["stage16_reliability_calibration"] == (
        "SCIENTIFICALLY_WITHHELD"
    )


def test_independent_patient_identity_is_not_claimed() -> None:
    assert CONFIG["governed_independent_patient_identity_sufficient"] is False
    assert (
        "GOVERNED_INDEPENDENT_PATIENT_IDENTITY_OR_PROVEN_NON_OVERLAP"
        in CONFIG["future_required_evidence"]
    )


def test_stage26a_performs_no_acquisition_execution_or_tuning() -> None:
    for field in (
        "new_dataset_acquisition_authorized",
        "new_identity_resolution_authorized",
        "new_label_harmonization_authorized",
        "new_manual_adjudication_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "prediction_generation_authorized",
        "threshold_tuning_authorized",
        "calibration_fitting_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
    ):
        assert CONFIG[field] is False
    assert CONFIG["currently_planned_llm_authorized_gate"] is None


def test_stage26a_readiness_result_is_withheld_not_failed() -> None:
    result = audit_readiness(CONFIG, ROOT)
    assert result["status"] == "PASSED_EXTERNAL_VALIDATION_DATA_READINESS_WITH_NO_ELIGIBLE_COHORT"
    assert result["disposition"] == (
        "SCIENTIFICALLY_WITHHELD_NO_GOVERNED_INDEPENDENT_EXTERNAL_VALIDATION_COHORT"
    )
    assert result["closure_classification"] == "WITHHELD_NOT_FAILED"
    assert result["model_inference_performed"] is False
    assert result["locked_test_records_accessed"] == 0


def test_stage26a_handoff_is_shortest_final_path() -> None:
    assert CONFIG["next_canonical_stage"] == "26B_EXTERNAL_VALIDATION_WITHHOLDING_CLOSURE"
    assert CONFIG["stage26_may_close_with_withheld_not_failed"] is True
    assert CONFIG["shortest_remaining_path"] == [
        "STAGE26B_EXTERNAL_VALIDATION_WITHHOLDING_CLOSURE",
        "STAGE27A_PAPER_FINAL_RELEASE_AUDIT",
    ]
    assert CONFIG["major_conceptual_stages_remaining_including_stage26"] == 2
