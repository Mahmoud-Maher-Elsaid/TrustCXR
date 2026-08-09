from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_REGIONS = {
    "GLOBAL_RESEARCH_ONLY_BANNER",
    "JOB_STATUS_PANEL",
    "MEDICAL_IMAGE_VIEWER_PANEL",
    "VIEW_TECHNICAL_QUALITY_PANEL",
    "CLASSIFIER_SCORE_PANEL",
    "RELIABILITY_UNCERTAINTY_PANEL",
    "LIMITED_FUSION_EVIDENCE_PANEL",
    "RESEARCH_REPORT_DRAFT_PANEL",
    "VERIFIER_PANEL",
    "ACCEPT_REVISE_DEFER_DECISION_PANEL",
    "PROVENANCE_PANEL",
    "SANITIZED_TECHNICAL_ERROR_PANEL",
}

EXPECTED_LABEL_ORDER = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

EXPECTED_VERIFIER_STATUSES = {
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNVERIFIED",
    "CONTRADICTED",
    "NOT_APPLICABLE",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fingerprint(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage22a_summary"]
    if sha256(summary_path) != config["stage22a_summary_sha256"]:
        raise RuntimeError("Stage 22A summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["stage21h_summary_sha256"] != config["stage21h_summary_sha256"]:
        raise RuntimeError("Stage 21H evidence changed.")
    if summary["ui_technology_recommendation"] != (
        "LIGHTWEIGHT_STATIC_HTML_CSS_JAVASCRIPT_SERVED_BY_EXISTING_FASTAPI"
    ):
        raise RuntimeError("Stage 22A UI technology changed.")
    if summary["new_ui_dependency_required"]:
        raise RuntimeError("Stage 22A did not authorize a new UI dependency.")
    expected_formats = {
        "PNG": "READY_EXISTING_PILLOW_DECODE_PATHS",
        "JPEG": "READY_EXISTING_PILLOW_DECODE_PATHS",
        "DICOM": "WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT",
        "TENSOR_INTERNAL": "INTERNAL_ONLY_NOT_A_BROWSER_DISPLAY_CONTRACT",
    }
    if summary["image_format_readiness"] != expected_formats:
        raise RuntimeError("Stage 22A image format evidence changed.")

    contract = config["contract"]
    observed_fingerprint = contract_fingerprint(contract)
    if observed_fingerprint != config["contract_fingerprint"]:
        raise RuntimeError("Stage 22B contract fingerprint mismatch.")
    if set(contract["layout_regions"]) != REQUIRED_REGIONS:
        raise RuntimeError("Stage 22B layout contract is incomplete.")
    if contract["stage9"]["label_order"] != EXPECTED_LABEL_ORDER:
        raise RuntimeError("Stage 9 frozen label order changed.")
    if set(contract["stage19"]["status_wording"]) != EXPECTED_VERIFIER_STATUSES:
        raise RuntimeError("Stage 19 verifier status display is incomplete.")
    if contract["stage20"]["precedence"] != [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]:
        raise RuntimeError("Stage 20 decision precedence changed.")
    if contract["image_viewer"]["allowed_formats"] != ["PNG", "JPEG"]:
        raise RuntimeError("Only PNG and JPEG may enter the prospective viewer contract.")
    if "DICOM" not in contract["image_viewer"]["withheld_or_prohibited"]:
        raise RuntimeError("DICOM viewer hold was removed.")
    if contract["stage10_11"]["maximum_support"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11 evidence support was upgraded.")
    if contract["stage16"]["uncertainty_term"] != "PREDICTIVE UNCERTAINTY":
        raise RuntimeError("Predictive uncertainty terminology changed.")
    if not contract["visual_semantics"]["required_non_color_distinction"]:
        raise RuntimeError("Non-color status distinctions are required.")
    if "NO_NPM_NODE" not in contract["technology"] or "NO_CDN" not in contract["technology"]:
        raise RuntimeError("Frozen minimal UI technology changed.")
    if len(contract["synthetic_fixture_specs"]) != 16:
        raise RuntimeError("Stage 22B synthetic fixture preparation is incomplete.")

    for key in (
        "ui_implementation_permitted",
        "synthetic_image_rendering_permitted",
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "patient_processing_permitted",
        "locked_test_access_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 22B prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "22B",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "contract_fingerprint": config["contract_fingerprint"],
        "stage22a_summary_sha256": config["stage22a_summary_sha256"],
        "stage21h_summary_sha256": config["stage21h_summary_sha256"],
        "research_banner": contract["designation"]["research_banner"],
        "layout_regions": contract["layout_regions"],
        "image_viewer_contract": contract["image_viewer"],
        "overlay_contract": contract["overlays"],
        "stage9_label_order": contract["stage9"]["label_order"],
        "verifier_status_wording": contract["stage19"]["status_wording"],
        "decision_precedence": contract["stage20"]["precedence"],
        "failure_display_contract": contract["stage21"],
        "provenance_contract": contract["provenance"],
        "visual_semantics": contract["visual_semantics"],
        "accessibility": contract["accessibility"],
        "privacy": contract["privacy"],
        "technology": contract["technology"],
        "synthetic_fixture_specs": contract["synthetic_fixture_specs"],
        "ui_implemented": False,
        "browser_ui_started": False,
        "synthetic_images_rendered": 0,
        "real_images_displayed": 0,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_ui_implementation": True,
        "next_stage_authorizes_synthetic_image_viewer_rendering": True,
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
    result = validate_contract(config, root)
    output = root / "reports/stage22/stage22b_research_ui_display_contract_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
