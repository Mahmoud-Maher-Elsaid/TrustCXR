from __future__ import annotations

import json
from pathlib import Path

from scripts.reproducibility.run_stage25a_mlops_reproducibility_data_readiness import (
    audit_readiness,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/reproducibility/stage25a_mlops_reproducibility_data_readiness.json").read_text(
        encoding="utf-8"
    )
)


def test_stage25a_accepts_exact_stage24b_closure_evidence() -> None:
    summary = ROOT / CONFIG["stage24b_summary"]
    assert sha256(summary) == CONFIG["stage24b_summary_sha256"]
    evidence = json.loads(summary.read_text(encoding="utf-8"))
    assert evidence["status"] == "SCIENTIFICALLY_WITHHELD_NO_GOVERNED_EXPERT_FEEDBACK_DATA"
    assert evidence["closure_classification"] == "WITHHELD_NOT_FAILED"
    assert evidence["stage24_closed"] is True


def test_stage25a_audits_all_accepted_checkpoint_integrity() -> None:
    result = audit_readiness(CONFIG, ROOT, verify_repository=False, verify_environment=False)
    assert result["all_accepted_model_checkpoints_have_integrity_evidence"] is True
    assert len(result["accepted_model_artifacts"]) == 7
    assert all(item["present_locally"] for item in result["accepted_model_artifacts"])
    assert all(item["integrity_verified"] for item in result["accepted_model_artifacts"])
    assert all(item["intentionally_git_ignored"] for item in result["accepted_model_artifacts"])


def test_stage25a_environment_is_partially_governed_with_one_blocker() -> None:
    dependencies = CONFIG["dependency_evidence"]
    assert dependencies["status"] == "PARTIALLY_GOVERNED_REQUIRES_UNIFIED_FINAL_LOCK"
    assert dependencies["conflicting_pins_found"] == []
    assert dependencies["ungoverned_runtime_imports_found"] == []
    assert (
        dependencies["blocking_gap"]
        == "NO_SINGLE_HASHED_FINAL_ENVIRONMENT_LOCK_AND_INSTALL_PROTOCOL"
    )
    assert CONFIG["gap_classification"]["BLOCKING"] == [dependencies["blocking_gap"]]


def test_stage25a_dataset_reconstruction_is_patient_safe_and_documented() -> None:
    assert len(CONFIG["dataset_reconstruction"]) == 4
    for dataset in CONFIG["dataset_reconstruction"]:
        assert dataset["patient_safe"] is True
        assert (ROOT / dataset["governance"]).is_file()
        assert (ROOT / dataset["layout"]).is_file()
        assert (ROOT / dataset["split_evidence"]).is_file()


def test_stage25a_canonical_evidence_sources_exist() -> None:
    for relative in CONFIG["canonical_evidence_sources"]:
        assert (ROOT / relative).is_file(), relative


def test_stage25a_preserves_determinism_and_cuda_limit() -> None:
    evidence = CONFIG["determinism_evidence"]
    assert evidence["patient_safe_splits"] is True
    assert evidence["recorded_training_seeds"] is True
    assert evidence["recorded_bootstrap_seeds"] is True
    assert evidence["synthetic_fixture_determinism"] is True
    assert evidence["exact_cuda_numerical_reproduction_guaranteed"] is False


def test_stage25a_preserves_all_material_scientific_limitations() -> None:
    limitations = set(CONFIG["frozen_limitations"])
    assert {
        "STAGE11_MAXIMUM_PARTIALLY_SUPPORTED",
        "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
        "STAGE13_SELECTIVE_PREDICTION_NOT_ACCEPTED",
        "OOD_WITHHELD",
        "STAGE14_TEMPORAL_WITHHELD",
        "STAGE15_SEVERITY_WITHHELD",
        "STAGE17_DEFER_ONLY",
        "STAGE18_DETERMINISTIC_REPORTING_ONLY",
        "STAGE20_DEFER_HIGHEST_PRECEDENCE",
        "STAGE22_REAL_IMAGE_AND_OVERLAY_LIMITS",
        "STAGE23_SYNTHETIC_ONLY_DICOM",
        "STAGE24_ACTIVE_LEARNING_WITHHELD",
    } <= limitations


def test_stage25a_gap_classification_does_not_promote_optional_work() -> None:
    gaps = CONFIG["gap_classification"]
    assert "NO_AUTOMATED_CI_WORKFLOW_CURRENTLY_TRACKED" in gaps["NON_BLOCKING_DOCUMENTATION_GAP"]
    assert "PRODUCTION_DEPLOYMENT_INFRASTRUCTURE" in gaps["OPTIONAL_IMPROVEMENT"]
    assert CONFIG["production_infrastructure_required"] is False
    assert CONFIG["optional_capability_expansion_required"] is False


def test_stage25a_prohibits_execution_and_llm_scope() -> None:
    for field in (
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "locked_test_access_authorized",
        "new_patient_processing_authorized",
        "active_learning_authorized",
        "gpu_residency_profiling_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
    ):
        assert CONFIG[field] is False
    assert CONFIG["currently_planned_llm_authorized_gate"] is None


def test_stage25a_handoff_is_minimal_closure_path() -> None:
    assert CONFIG["next_canonical_stage"] == "25B_MLOPS_REPRODUCIBILITY_CLOSURE"
    assert CONFIG["stage25_can_likely_close_without_retraining"] is True
    assert CONFIG["remaining_mandatory_path_after_stage25"] == [
        "ORIGINAL_STAGE23_EXTERNAL_VALIDATION_EVIDENCE_OR_FORMAL_BLOCKER_DISPOSITION",
        "ORIGINAL_STAGE24_PAPER_FINAL_RELEASE_AUDIT",
    ]
    assert CONFIG["major_conceptual_stages_remaining_including_stage25"] == 3
