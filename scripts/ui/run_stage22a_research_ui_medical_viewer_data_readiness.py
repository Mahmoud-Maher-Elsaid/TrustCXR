from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_UI_ELEMENTS = {
    "VIEW_ASSESSMENT",
    "TECHNICAL_QUALITY_PROXY",
    "CLASSIFIER_SCORES",
    "LIMITED_FUSION",
    "RELIABILITY",
    "TRIAGE_DEFER",
    "REPORT_DRAFT",
    "VERIFIER",
    "DECISION",
    "JOB_AND_FAILURE",
}

REQUIRED_FORMATS = {
    "PNG": "READY_EXISTING_PILLOW_DECODE_PATHS",
    "JPEG": "READY_EXISTING_PILLOW_DECODE_PATHS",
    "DICOM": "WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT",
    "TENSOR_INTERNAL": "INTERNAL_ONLY_NOT_A_BROWSER_DISPLAY_CONTRACT",
}

REQUIRED_SAFETY_LIMITATIONS = {
    "STAGE11_MAXIMUM_SUPPORT_PARTIALLY_SUPPORTED",
    "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
    "NO_LOCALIZATION_ABSENCE_CONTRADICTION",
    "STAGE13_SELECTIVE_PREDICTION_NOT_ACCEPTED",
    "OOD_WITHHELD",
    "STAGE17_DEFER_ONLY",
    "STAGE18_DETERMINISTIC_GROUNDED_REPORTING",
    "STAGE19_VERIFIER_RESTRICTIONS",
    "STAGE20_DEFER_HIGHEST_PRECEDENCE",
    "NO_SEVERITY",
    "NO_TEMPORAL_CHANGE",
    "NO_DEVICE_LOCALIZATION",
    "NO_CLINICAL_DIAGNOSIS",
    "NO_TREATMENT_RECOMMENDATION",
    "NO_CLINICAL_APPROVAL",
    "NO_AUTONOMOUS_RELEASE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_readiness(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage21h_summary"]
    if sha256(summary_path) != config["stage21h_summary_sha256"]:
        raise RuntimeError("Stage 21H summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "research_designation": config["stage21_research_designation"],
        "accepted_scope": config["accepted_serving_scope"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21f_summary_sha256": config["stage21f_summary_sha256"],
        "stage21g_summary_sha256": config["stage21g_summary_sha256"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 21 frozen evidence mismatch: {key}")

    elements = config["candidate_ui_elements"]
    if {item["id"] for item in elements} != REQUIRED_UI_ELEMENTS:
        raise RuntimeError("Stage 22A UI capability inventory is incomplete.")
    required_fields = {
        "source_stage",
        "structured_field",
        "wording",
        "numerical_values",
        "qualifier",
        "prohibited",
    }
    if any(not required_fields <= item.keys() for item in elements):
        raise RuntimeError("A Stage 22A UI element lacks its display governance contract.")
    if config["image_format_readiness"] != REQUIRED_FORMATS:
        raise RuntimeError("Stage 22A image-format readiness changed.")
    if set(config["safety_limitations"]) != REQUIRED_SAFETY_LIMITATIONS:
        raise RuntimeError("Stage 22A safety limitations changed.")
    if config["new_ui_dependency_required"]:
        raise RuntimeError("Stage 22A must not add a UI dependency.")
    if config["ui_technology_recommendation"] != (
        "LIGHTWEIGHT_STATIC_HTML_CSS_JAVASCRIPT_SERVED_BY_EXISTING_FASTAPI"
    ):
        raise RuntimeError("Stage 22A technology recommendation changed.")
    if config["viewer_readiness"]["positive_lesion_overlay"] != ("WITHHELD_INSUFFICIENT_EVIDENCE"):
        raise RuntimeError("Unsupported localization overlay was enabled.")
    for key in (
        "ui_implementation_permitted",
        "real_image_display_permitted",
        "persistent_server_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 22A prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "22A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage21_research_designation": config["stage21_research_designation"],
        "accepted_serving_scope": config["accepted_serving_scope"],
        "stage21h_summary_sha256": config["stage21h_summary_sha256"],
        "eligible_ui_elements": elements,
        "viewer_readiness": config["viewer_readiness"],
        "overlay_policy": {
            "stage8": "QUALITY_FILTERED_PSEUDO_ANATOMY_ONLY_WITHHELD_PENDING_AUTHORIZATION",
            "stage10": "GEOMETRY_THORACIC_LOCATION_PROXY_ONLY_WITHHELD_PENDING_AUTHORIZATION",
            "stage11_maximum_support": "PARTIALLY_SUPPORTED",
            "positive_lesion_localization": "WITHHELD_INSUFFICIENT_EVIDENCE",
            "laterality_from_localization": "PROHIBITED",
            "negation_from_localization_absence": "PROHIBITED",
            "box_as_mask": "PROHIBITED",
            "heuristic_anatomical_join": "PROHIBITED",
        },
        "image_format_readiness": config["image_format_readiness"],
        "ui_technology_recommendation": config["ui_technology_recommendation"],
        "new_ui_dependency_required": False,
        "privacy_rules": config["privacy_rules"],
        "accessibility_requirements": config["accessibility_requirements"],
        "safety_limitations": config["safety_limitations"],
        "ui_implemented": False,
        "browser_ui_started": False,
        "persistent_server_started": False,
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
        "next_stage_authorizes_ui_implementation": False,
        "next_stage_authorizes_real_image_display": False,
        "next_stage_authorizes_real_model_inference": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_readiness(config, root)
    output = (
        root / "reports/stage22/stage22a_research_ui_medical_viewer_data_readiness_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
