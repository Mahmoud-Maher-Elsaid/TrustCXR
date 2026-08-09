from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.ui.run_stage22a_research_ui_medical_viewer_data_readiness import (
    REQUIRED_FORMATS,
    REQUIRED_SAFETY_LIMITATIONS,
    REQUIRED_UI_ELEMENTS,
    audit_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ui/stage22a_research_ui_medical_viewer_data_readiness.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage22a_frozen_stage21_evidence_is_preserved() -> None:
    result = audit_readiness(config(), ROOT)
    assert result["stage21_research_designation"] == "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"
    assert result["accepted_serving_scope"] == (
        "MINIMAL_LOCAL_RESEARCH_SERVING_SYNTHETIC_RUNTIME_ONLY"
    )
    assert result["stage21h_summary_sha256"] == (
        "ff59a3710e618092ee1e5da6ca4ea136fac02c50ac0ee77b3974f1d1be1f0fb0"
    )


def test_stage22a_ui_inventory_is_complete_and_governed() -> None:
    result = audit_readiness(config(), ROOT)
    elements = result["eligible_ui_elements"]
    assert {item["id"] for item in elements} == REQUIRED_UI_ELEMENTS
    for item in elements:
        assert item["source_stage"]
        assert item["structured_field"]
        assert item["wording"]
        assert item["qualifier"]
        assert item["prohibited"]


def test_stage22a_scores_are_research_only_and_uncertainty_is_not_epistemic() -> None:
    elements = {
        item["id"]: item for item in audit_readiness(config(), ROOT)["eligible_ui_elements"]
    }
    scores = elements["CLASSIFIER_SCORES"]
    assert scores["numerical_values"]
    assert "not a diagnosis" in scores["wording"]
    reliability = elements["RELIABILITY"]
    assert reliability["numerical_values"]
    assert "PREDICTIVE_ONLY_NOT_EPISTEMIC" in reliability["qualifier"]


def test_stage22a_report_verifier_and_decision_display_limits() -> None:
    elements = {
        item["id"]: item for item in audit_readiness(config(), ROOT)["eligible_ui_elements"]
    }
    assert "AI-generated research report draft" in elements["REPORT_DRAFT"]["wording"]
    assert "UPGRADE_WITHHELD_OR_PARTIAL_EVIDENCE" == elements["VERIFIER"]["prohibited"]
    assert "DEFER_PRECEDENCE" in elements["DECISION"]["qualifier"]


def test_stage22a_format_readiness_is_evidence_limited() -> None:
    result = audit_readiness(config(), ROOT)
    assert result["image_format_readiness"] == REQUIRED_FORMATS
    assert result["image_format_readiness"]["PNG"].startswith("READY_")
    assert result["image_format_readiness"]["JPEG"].startswith("READY_")
    assert result["image_format_readiness"]["DICOM"].startswith("WITHHELD_")
    assert result["image_format_readiness"]["TENSOR_INTERNAL"].startswith("INTERNAL_ONLY_")


def test_stage22a_repository_has_png_jpeg_and_dataset_specific_dicom_evidence() -> None:
    stage13 = (ROOT / "src/trustcxr/multiview/stage13d_baseline.py").read_text(encoding="utf-8")
    stage10 = (ROOT / "src/trustcxr/detection/stage10e_rsna.py").read_text(encoding="utf-8")
    structure = (ROOT / "src/trustcxr/data/structure_validation.py").read_text(encoding="utf-8")
    assert "Image.open" in stage13
    assert '".jpg"' in structure and '".png"' in structure
    assert "pydicom.dcmread" in stage10
    assert "stage10e_rsna" in str(ROOT / "src/trustcxr/detection/stage10e_rsna.py")


def test_stage22a_overlay_policy_cannot_upgrade_proxy_evidence() -> None:
    result = audit_readiness(config(), ROOT)
    policy = result["overlay_policy"]
    assert policy["stage11_maximum_support"] == "PARTIALLY_SUPPORTED"
    assert policy["positive_lesion_localization"] == "WITHHELD_INSUFFICIENT_EVIDENCE"
    assert policy["laterality_from_localization"] == "PROHIBITED"
    assert policy["negation_from_localization_absence"] == "PROHIBITED"
    assert policy["box_as_mask"] == "PROHIBITED"


def test_stage22a_uses_existing_stack_without_new_dependency() -> None:
    result = audit_readiness(config(), ROOT)
    assert result["ui_technology_recommendation"] == (
        "LIGHTWEIGHT_STATIC_HTML_CSS_JAVASCRIPT_SERVED_BY_EXISTING_FASTAPI"
    )
    assert not result["new_ui_dependency_required"]
    serving_lock = (ROOT / "requirements/lock-serving-stage21.txt").read_text(encoding="utf-8")
    assert "fastapi==0.116.1" in serving_lock
    assert "streamlit" not in serving_lock.lower()
    assert "gradio" not in serving_lock.lower()


def test_stage22a_preserves_every_safety_limitation() -> None:
    result = audit_readiness(config(), ROOT)
    assert set(result["safety_limitations"]) == REQUIRED_SAFETY_LIMITATIONS


@pytest.mark.parametrize(
    "key",
    [
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
    ],
)
def test_stage22a_rejects_execution_scope_expansion(key: str) -> None:
    cfg = config()
    cfg[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        audit_readiness(cfg, ROOT)


def test_stage22a_privacy_and_accessibility_contracts_are_explicit() -> None:
    result = audit_readiness(config(), ROOT)
    assert "NO_EXTERNAL_IMAGE_UPLOAD" in result["privacy_rules"]
    assert "NO_TELEMETRY_OR_EXTERNAL_ANALYTICS_BY_DEFAULT" in result["privacy_rules"]
    assert "VISIBLE_RESEARCH_ONLY_BANNER" in result["accessibility_requirements"]
    assert "NO_DIAGNOSTIC_COLOR_SEMANTICS" in result["accessibility_requirements"]


def test_stage22a_next_stage_is_contract_only_without_llm_gate() -> None:
    result = audit_readiness(config(), ROOT)
    assert result["next_canonical_stage"] == "22B_RESEARCH_UI_DISPLAY_CONTRACT"
    assert not result["next_stage_authorizes_ui_implementation"]
    assert not result["next_stage_authorizes_real_image_display"]
    assert not result["next_stage_authorizes_real_model_inference"]
    assert not result["next_stage_authorizes_language_model_work"]
    assert result["currently_planned_llm_authorized_gate"] is None


def test_stage22a_readiness_audit_does_not_write_output() -> None:
    output = (
        ROOT / "reports/stage22/stage22a_research_ui_medical_viewer_data_readiness_summary.json"
    )
    before = output.read_bytes() if output.exists() else None
    audit_readiness(config(), ROOT)
    after = output.read_bytes() if output.exists() else None
    assert after == before
