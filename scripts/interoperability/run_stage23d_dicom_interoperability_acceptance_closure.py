from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_acceptance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage23c_summary"]
    if sha256(summary_path) != config["stage23c_summary_sha256"]:
        raise RuntimeError("Stage 23C summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASSED_SYNTHETIC_DICOM_IMPLEMENTATION_VALIDATION",
        "gate": "GO_FOR_STAGE_23D_DICOM_INTEROPERABILITY_ACCEPTANCE_AND_CLOSURE",
        "stage23b_contract_fingerprint": config["stage23b_contract_fingerprint"],
        "governed_dependency": "pydicom==3.0.2",
        "fixtures_created": 22,
        "fixtures_passed": 22,
        "fixtures_failed": 0,
        "transfer_syntaxes_validated": [
            "EXPLICIT_VR_LITTLE_ENDIAN",
            "IMPLICIT_VR_LITTLE_ENDIAN",
        ],
        "photometric_interpretations_validated": ["MONOCHROME1", "MONOCHROME2"],
        "deterministic_pixeldata_decoding": "PASSED",
        "deterministic_repeat": "PASSED",
        "temporary_artifact_cleanup": "COMPLETE",
        "real_dicom_files_used": 0,
        "patient_metadata_used": False,
        "dicom_ui_rendering_performed": False,
        "real_image_displayed": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "gpu_profiling_performed": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 23C acceptance evidence mismatch: {key}")
    required_accepted = {
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
    if set(config["accepted_capabilities"]) != required_accepted:
        raise RuntimeError("Stage 23D accepted capability scope changed.")
    required_withheld = {
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
    if set(config["withheld_capabilities"]) != required_withheld:
        raise RuntimeError("Stage 23D withholding scope changed.")
    for key in (
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
    ):
        if config[key]:
            raise RuntimeError(f"Stage 23D prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")
    if config["mandatory_original_stages_remaining_after_stage23d"] != [21, 22, 23, 24]:
        raise RuntimeError("Remaining original-roadmap stages changed.")
    if config["mandatory_original_stage_count_remaining_after_stage23d"] != 4:
        raise RuntimeError("Remaining mandatory stage count changed.")
    if config["direct_final_release_audit_authorized"]:
        raise RuntimeError("Mandatory original stages cannot be bypassed.")

    return {
        "stage": "23D",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage23b_contract_fingerprint": config["stage23b_contract_fingerprint"],
        "stage23c_summary_sha256": config["stage23c_summary_sha256"],
        "accepted_capabilities": config["accepted_capabilities"],
        "withheld_capabilities": config["withheld_capabilities"],
        "operational_limits": config["operational_limits"],
        "frozen_previous_limitations": config["frozen_previous_limitations"],
        "stage23_closed": True,
        "closure_scope": "SYNTHETIC_NON_PATIENT_ONLY",
        "packages_installed": [],
        "real_dicom_files_used": 0,
        "patient_data_used": False,
        "locked_test_records_accessed": 0,
        "model_inference_performed": False,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "mandatory_original_stages_remaining_after_stage23d": config[
            "mandatory_original_stages_remaining_after_stage23d"
        ],
        "mandatory_original_stage_count_remaining_after_stage23d": 4,
        "exact_future_repository_substage_count_determinable": False,
        "direct_final_release_audit_authorized": False,
        "optional_capability_expansions_may_be_omitted": True,
        "original_roadmap_stages_may_be_omitted": False,
        "language_model_mandatory_for_project_completion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = decide_acceptance(config, root)
    output = root / "reports/stage23/stage23d_dicom_interoperability_acceptance_closure.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
