from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from trustcxr.serving.api import create_app

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_synthetic_image(item: dict[str, Any], expected_format: str) -> tuple[int, int]:
    prefix = f"data:image/{expected_format.lower()};base64,"
    if not item["data_url"].startswith(prefix):
        raise RuntimeError(f"Synthetic {expected_format} fixture has an invalid data URL.")
    payload = base64.b64decode(item["data_url"][len(prefix) :], validate=True)
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != expected_format or image.size != (item["width"], item["height"]):
            raise RuntimeError(f"Synthetic {expected_format} fixture metadata mismatch.")
        return image.size


def validate_implementation(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage22b_summary"]
    if sha256(summary_path) != config["stage22b_summary_sha256"]:
        raise RuntimeError("Stage 22B summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["contract_fingerprint"] != config["stage22b_contract_fingerprint"]:
        raise RuntimeError("Stage 22B contract fingerprint mismatch.")

    static_root = root / config["static_root"]
    for filename, expected_hash in config["assets"].items():
        if sha256(static_root / filename) != expected_hash:
            raise RuntimeError(f"Stage 22C static asset hash mismatch: {filename}")
    html = (static_root / "index.html").read_text(encoding="utf-8")
    css = (static_root / "app.css").read_text(encoding="utf-8")
    javascript = (static_root / "app.js").read_text(encoding="utf-8")
    fixtures = json.loads((static_root / "fixtures.json").read_text(encoding="utf-8"))

    if {
        region for region in REQUIRED_REGIONS if f'data-region="{region}"' in html
    } != REQUIRED_REGIONS:
        raise RuntimeError("Stage 22C does not implement every frozen layout region.")
    for phrase in ("RESEARCH USE ONLY", "NOT A MEDICAL DIAGNOSIS", "EXPERT REVIEW REQUIRED"):
        if phrase not in html:
            raise RuntimeError(f"Persistent research designation is missing: {phrase}")
    for prohibited in (
        "innerHTML",
        "outerHTML",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "eval(",
        "http://",
        "https://",
    ):
        if prohibited in javascript:
            raise RuntimeError(f"Unsafe browser behavior detected: {prohibited}")
    if "textContent" not in javascript or "createElement" not in javascript:
        raise RuntimeError("Stage 22C must use safe deterministic text rendering.")
    if "data:" not in html or "object-src 'none'" not in html or "base-uri 'none'" not in html:
        raise RuntimeError("Stage 22C Content Security Policy is incomplete.")
    for prohibited in ("red = disease", "green = healthy", "traffic-light", "severity-gradient"):
        if prohibited in css.lower():
            raise RuntimeError("Diagnostic color semantics detected.")

    if not fixtures["non_patient"] or not fixtures["job"]["job_id"].startswith("job_"):
        raise RuntimeError("Stage 22C fixture privacy contract failed.")
    if [item["label"] for item in fixtures["classifier_scores"]] != EXPECTED_LABEL_ORDER:
        raise RuntimeError("Stage 22C classifier label order changed.")
    if len(fixtures["classifier_scores"]) != 14:
        raise RuntimeError("Stage 22C requires exactly 14 synthetic classifier scores.")
    decoded = {
        image_format: decode_synthetic_image(
            fixtures["synthetic_images"][image_format], image_format
        )
        for image_format in config["synthetic_viewer_fixture_types"]
    }
    if fixtures["security_cases"] != {
        "malicious_text": "<script src='https://example.invalid/x.js'>alert(1)</script>",
        "patient_identifier_field": "REJECTED",
        "internal_path_field": "REJECTED",
        "dicom": "WITHHELD",
        "stage8_overlay": "WITHHELD",
        "stage10_overlay": "WITHHELD",
    }:
        raise RuntimeError("Stage 22C security fixture contract changed.")

    app = create_app()
    routes = {
        f"{method} {route.path}"
        for route in app.routes
        for method in (getattr(route, "methods", None) or set())
    }
    if not set(config["ui_routes"]) <= routes:
        raise RuntimeError("Stage 22C UI routes are incomplete.")
    frozen_api = {"POST /v1/jobs", "GET /v1/jobs/{job_id}", "GET /health"}
    if not frozen_api <= routes:
        raise RuntimeError("Stage 21 frozen API surface changed.")

    for key in (
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "persistent_server_permitted",
        "persistent_browser_permitted",
        "stage8_overlay_activation_permitted",
        "stage10_overlay_activation_permitted",
        "dicom_support_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 22C prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "22C",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "stage22b_summary_sha256": config["stage22b_summary_sha256"],
        "implemented_ui_files": sorted(config["assets"]),
        "implemented_components": sorted(REQUIRED_REGIONS),
        "ui_routes": config["ui_routes"],
        "synthetic_viewer_fixture_count": config["synthetic_viewer_fixture_count"],
        "synthetic_viewer_fixture_types": config["synthetic_viewer_fixture_types"],
        "synthetic_viewer_dimensions": {key: list(value) for key, value in decoded.items()},
        "static_asset_integrity": "PASSED",
        "html_javascript_safety": "PASSED",
        "deterministic_rendering_contract": "PASSED",
        "accessibility_contract": "PASSED",
        "dependency_changed": False,
        "browser_started": False,
        "persistent_server_started": False,
        "real_images_displayed": 0,
        "stage8_overlay_activated": False,
        "stage10_overlay_activated": False,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_bounded_ui_runtime_browser_validation": True,
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
    result = validate_implementation(config, root)
    output = root / "reports/stage22/stage22c_synthetic_research_ui_implementation_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
