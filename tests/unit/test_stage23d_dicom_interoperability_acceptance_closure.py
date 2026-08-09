from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.interoperability.run_stage23d_dicom_interoperability_acceptance_closure import (
    decide_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (
        ROOT / "configs/interoperability/stage23d_dicom_interoperability_acceptance_closure.json"
    ).read_text(encoding="utf-8")
)


def test_stage23d_accepts_exact_stage23c_evidence_only() -> None:
    result = decide_acceptance(CONFIG, ROOT)
    assert result["status"] == "ACCEPTED_SYNTHETIC_DICOM_INTEROPERABILITY_RESEARCH_ONLY"
    assert result["stage23b_contract_fingerprint"] == (
        "4b57e03cc8e2372afdfcca74147635d24bca4f08f52b4ae09415008e8906b354"
    )
    assert result["stage23_closed"]
    assert result["closure_scope"] == "SYNTHETIC_NON_PATIENT_ONLY"
    assert result["packages_installed"] == []


def test_stage23d_accepted_scope_is_exactly_demonstrated_scope() -> None:
    assert set(CONFIG["accepted_capabilities"]) == {
        "SYNTHETIC_NON_PATIENT_DICOM",
        "SINGLE_FRAME_GRAYSCALE",
        "EXPLICIT_VR_LITTLE_ENDIAN",
        "IMPLICIT_VR_LITTLE_ENDIAN",
        "MONOCHROME1",
        "MONOCHROME2",
        "DETERMINISTIC_RAW_MODALITY_DISPLAY_REPRESENTATION_SEPARATION",
        "DETERMINISTIC_GOVERNED_SYNTHETIC_DISPLAY_NORMALIZATION",
        "PRIVACY_FAIL_CLOSED",
        "BOUNDED_RESOURCE_LIMITS",
    }
    assert CONFIG["operational_limits"] == {
        "maximum_rows": 4096,
        "maximum_columns": 4096,
        "maximum_frames": 1,
        "maximum_uncompressed_pixel_bytes": 67108864,
    }


def test_stage23d_withholds_all_unsupported_dicom_and_clinical_scope() -> None:
    assert set(CONFIG["withheld_capabilities"]) == {
        "COMPRESSED_DICOM",
        "MULTI_FRAME_DICOM",
        "GENERIC_REAL_PATIENT_DICOM_USE",
        "REAL_DICOM_UI_RENDERING",
        "STAGE8_OVERLAY",
        "STAGE10_OVERLAY",
        "RELIABLE_LESION_LOCALIZATION",
        "FINDING_LATERALITY",
        "CLINICAL_DIAGNOSIS",
        "SEVERITY",
        "TEMPORAL_CHANGE",
        "PATIENT_PROCESSING",
    }


def test_stage23d_preserves_previous_stage_limitations() -> None:
    limitations = set(CONFIG["frozen_previous_limitations"])
    assert {
        "STAGE11_MAXIMUM_PARTIALLY_SUPPORTED",
        "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
        "NO_LOCALIZATION_ABSENCE_CONTRADICTION",
        "STAGE13_SELECTIVE_PREDICTION_NOT_ACCEPTED",
        "OOD_WITHHELD",
        "STAGE17_DEFER_ONLY",
        "STAGE18_DETERMINISTIC_REPORTING",
        "STAGE19_VERIFIER_RESTRICTIONS",
        "STAGE20_DEFER_HIGHEST_PRECEDENCE",
        "STAGE22_OVERLAY_HOLD",
        "NO_CLINICAL_APPROVAL",
        "NO_AUTONOMOUS_RELEASE",
    } <= limitations


@pytest.mark.parametrize(
    "key",
    [
        "package_installation_authorized",
        "real_dicom_authorized",
        "real_image_authorized",
        "patient_processing_authorized",
        "dicom_ui_rendering_authorized",
        "stage8_overlay_authorized",
        "stage10_overlay_authorized",
        "model_loading_authorized",
        "model_inference_authorized",
        "gpu_profiling_authorized",
        "locked_test_access_authorized",
        "training_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage23d_rejects_scope_expansion(key: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        decide_acceptance(changed, ROOT)


def test_stage23d_validation_does_not_modify_finalized_closure_output() -> None:
    output = ROOT / "reports/stage23/stage23d_dicom_interoperability_acceptance_closure.json"
    assert output.is_file()
    before = output.read_bytes()
    decide_acceptance(CONFIG, ROOT)
    assert output.read_bytes() == before


def test_stage23d_handoff_preserves_all_remaining_original_stages() -> None:
    assert CONFIG["next_canonical_stage"] == ("24A_HUMAN_FEEDBACK_ACTIVE_LEARNING_DATA_READINESS")
    assert CONFIG["mandatory_original_stages_remaining_after_stage23d"] == [21, 22, 23, 24]
    assert CONFIG["mandatory_original_stage_count_remaining_after_stage23d"] == 4
    assert not CONFIG["exact_future_repository_substage_count_determinable"]
    assert not CONFIG["direct_final_release_audit_authorized"]
    assert CONFIG["optional_capability_expansions_may_be_omitted"]
    assert not CONFIG["original_roadmap_stages_may_be_omitted"]


def test_stage23d_no_llm_is_mandatory_or_scheduled() -> None:
    assert not CONFIG["language_model_mandatory_for_project_completion"]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None
