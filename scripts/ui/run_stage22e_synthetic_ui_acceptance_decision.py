from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.ui.run_stage22c_synthetic_research_ui_implementation_validation import sha256

EXPECTED_RUNTIME_EVIDENCE = {
    "status": "PASSED_BOUNDED_SYNTHETIC_UI_RUNTIME_BROWSER_VALIDATION",
    "runtime_cases_passed": 12,
    "runtime_cases_failed": 0,
    "browser_validation_performed": True,
    "browser_tooling_used": "msedge.exe",
    "browser_automation_packages_added": False,
    "synthetic_png_rendered_count": 1,
    "synthetic_jpeg_rendered_count": 1,
    "real_images_displayed": 0,
    "dicom_support_activated": False,
    "stage8_overlay_activated": False,
    "stage10_overlay_activated": False,
    "external_requests_observed": 0,
    "browser_persistence_observed": False,
    "injection_safety_result": "PASSED",
    "accessibility_result": "PASSED",
    "deterministic_repeat_result": "PASSED",
    "persistent_server_started": False,
    "bounded_server_started": True,
    "server_cleanup_status": "TERMINATED",
    "temporary_artifact_cleanup_status": "COMPLETE",
    "real_model_loaded": False,
    "real_inference_performed": False,
    "gpu_residency_profiling_performed": False,
    "real_patient_records_used": 0,
    "locked_test_records_accessed": 0,
    "language_model_used": False,
    "language_model_endpoint_prepared": False,
    "currently_planned_llm_authorized_gate": None,
}

EXPECTED_TECHNOLOGY = {
    "STATIC_HTML",
    "CSS",
    "VANILLA_JAVASCRIPT",
    "EXISTING_FASTAPI_STARLETTE",
    "EXISTING_V1_API",
    "NO_FRONTEND_FRAMEWORK",
    "NO_NPM_NODE",
    "NO_CDN",
    "NO_CLOUD_ASSETS",
    "NO_EXTERNAL_ANALYTICS",
}


def decide_acceptance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage22d_summary"]
    if sha256(summary_path) != config["stage22d_summary_sha256"]:
        raise RuntimeError("Stage 22D summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["stage22b_contract_fingerprint"] != config["stage22b_contract_fingerprint"]:
        raise RuntimeError("Stage 22B contract fingerprint mismatch.")
    if summary["stage22c_summary_sha256"] != config["stage22c_summary_sha256"]:
        raise RuntimeError("Stage 22C evidence hash mismatch.")
    for key, value in EXPECTED_RUNTIME_EVIDENCE.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 22D acceptance evidence mismatch: {key}")
    if set(config["technology"]) != EXPECTED_TECHNOLOGY:
        raise RuntimeError("Stage 22E technology freeze changed.")
    if config["image_formats"] != {
        "PNG": "ACCEPTED_GOVERNED_LOCAL_RESEARCH_VIEWER_CONTRACT",
        "JPEG": "ACCEPTED_GOVERNED_LOCAL_RESEARCH_VIEWER_CONTRACT",
        "DICOM": "WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT",
        "TENSOR_NPZ": "INTERNAL_ONLY_NOT_PUBLIC_BROWSER_DISPLAY",
    }:
        raise RuntimeError("Stage 22E image-format freeze changed.")
    if config["overlays"] != {
        "STAGE8": "WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION",
        "STAGE10": "WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION",
    }:
        raise RuntimeError("Stage 22E overlay hold changed.")
    if config["stage16_contract"]["allowed_term"] != "PREDICTIVE UNCERTAINTY":
        raise RuntimeError("Stage 22E predictive uncertainty terminology changed.")
    if config["stage20_precedence"] != [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]:
        raise RuntimeError("Stage 20 decision precedence changed.")
    if config["stage20_accept_meaning"] != "NOT_CLINICAL_APPROVAL":
        raise RuntimeError("Stage 20 ACCEPT meaning changed.")
    if config["stage20_revision_meaning"] != "DETERMINISTIC_CANONICAL_TEMPLATE_REPAIR_ONLY":
        raise RuntimeError("Stage 20 REVISE meaning changed.")
    for key in (
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "patient_processing_permitted",
        "locked_test_access_permitted",
        "stage8_overlay_activation_permitted",
        "stage10_overlay_activation_permitted",
        "dicom_support_activation_permitted",
        "persistent_service_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 22E prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "22E",
        "status": config["acceptance_designation"],
        "gate": config["expected_gate"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "stage22c_summary_sha256": config["stage22c_summary_sha256"],
        "stage22d_summary_sha256": config["stage22d_summary_sha256"],
        "research_banner": config["research_banner"],
        "technology": config["technology"],
        "accepted_ui_scope": config["accepted_ui_scope"],
        "image_formats": config["image_formats"],
        "overlays": config["overlays"],
        "stage9_restrictions": config["stage9_restrictions"],
        "stage16_contract": config["stage16_contract"],
        "stage19_statuses": config["stage19_statuses"],
        "stage20_decisions": config["stage20_decisions"],
        "stage20_precedence": config["stage20_precedence"],
        "stage20_accept_meaning": config["stage20_accept_meaning"],
        "stage20_revision_meaning": config["stage20_revision_meaning"],
        "failure_semantics": config["failure_semantics"],
        "browser_privacy_guarantees": config["browser_privacy_guarantees"],
        "security_accessibility_evidence": config["security_accessibility_evidence"],
        "prohibited_overlay_inferences": config["prohibited_overlay_inferences"],
        "accepted_evidence_scope": "SYNTHETIC_NON_PATIENT_ONLY",
        "real_images_displayed": 0,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_real_image_display": False,
        "next_stage_authorizes_real_model_loading": False,
        "next_stage_authorizes_bounded_real_inference": False,
        "next_stage_authorizes_gpu_residency_profiling": False,
        "next_stage_authorizes_patient_processing": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = decide_acceptance(config, root)
    output = root / "reports/stage22/stage22e_synthetic_ui_acceptance_decision_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
