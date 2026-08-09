from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.ui.run_stage22b_research_ui_display_contract import (
    EXPECTED_LABEL_ORDER,
    EXPECTED_VERIFIER_STATUSES,
    REQUIRED_REGIONS,
    contract_fingerprint,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ui/stage22b_research_ui_display_contract.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def result() -> dict:
    return validate_contract(config(), ROOT)


def test_stage22b_contract_fingerprint_is_exact_and_reproducible() -> None:
    cfg = config()
    assert cfg["contract_fingerprint"] == (
        "f413fbb969261cf9b9b51eb0870aba4de9555483acff42212c04639f293b3a2f"
    )
    assert contract_fingerprint(cfg["contract"]) == cfg["contract_fingerprint"]
    assert result()["contract_fingerprint"] == cfg["contract_fingerprint"]


def test_stage22b_research_banner_and_prohibited_implications() -> None:
    designation = config()["contract"]["designation"]
    assert designation["research_banner"] == [
        "RESEARCH USE ONLY",
        "NOT A MEDICAL DIAGNOSIS",
        "EXPERT REVIEW REQUIRED",
    ]
    assert set(designation["prohibited_implications"]) == {
        "CLINICAL_APPROVAL",
        "AUTONOMOUS_RELEASE",
        "DIAGNOSIS",
        "TREATMENT_RECOMMENDATION",
        "CLINICAL_CERTAINTY",
    }


def test_stage22b_all_layout_regions_are_frozen() -> None:
    assert set(result()["layout_regions"]) == REQUIRED_REGIONS


def test_stage22b_image_viewer_allows_only_png_jpeg_and_local_governed_source() -> None:
    viewer = result()["image_viewer_contract"]
    assert viewer["allowed_formats"] == ["PNG", "JPEG"]
    assert viewer["source_scope"] == "LOCAL_GOVERNED_IMAGE_ONLY"
    for withheld in (
        "DICOM",
        "EXTERNAL_IMAGE_URL",
        "REMOTE_IMAGE_LOADING",
        "BROWSER_PERSISTENCE",
        "UNSUPPORTED_OVERLAY",
        "ARBITRARY_FILESYSTEM_PATH",
    ):
        assert withheld in viewer["withheld_or_prohibited"]


def test_stage22b_overlays_remain_withheld_and_explicitly_proxy_only() -> None:
    overlays = result()["overlay_contract"]
    assert overlays["stage8"]["evidence_class"] == "QUALITY_FILTERED_PSEUDO_ANATOMY_ONLY"
    assert overlays["stage10"]["evidence_class"] == "GEOMETRY_THORACIC_LOCATION_PROXY_ONLY"
    assert overlays["stage8"]["authorization"].startswith("WITHHELD_")
    assert overlays["stage10"]["authorization"].startswith("WITHHELD_")
    assert "RELIABLE_POSITIVE_LESION_LOCALIZATION" in overlays["prohibited"]
    assert "NEGATION_FROM_LOCALIZATION_ABSENCE" in overlays["prohibited"]


def test_stage22b_stage5_display_does_not_invent_other_unknown() -> None:
    stage5 = config()["contract"]["stage5"]
    assert stage5["view_values"] == ["AP", "PA", "LATERAL"]
    assert stage5["other_unknown_model_capability"] == "WITHHELD_NOT_IMPLEMENTED"
    assert "not clinical" in stage5["view_label"].lower()
    assert "not a clinical image-quality assessment" in stage5["technical_quality_label"]


def test_stage22b_stage9_label_order_and_score_qualifier_are_frozen() -> None:
    stage9 = config()["contract"]["stage9"]
    assert stage9["label_order"] == EXPECTED_LABEL_ORDER
    assert stage9["score_fields"] == ["FINDING_LABEL", "MODEL_SCORE", "RESEARCH_MODEL_QUALIFIER"]
    assert "not a diagnosis" in stage9["qualifier"]
    assert "INVENTED_THRESHOLD" in stage9["prohibited"]


def test_stage22b_stage16_uses_predictive_uncertainty_only() -> None:
    stage16 = config()["contract"]["stage16"]
    assert stage16["uncertainty_term"] == "PREDICTIVE UNCERTAINTY"
    assert "EPISTEMIC UNCERTAINTY" in stage16["prohibited_terms"]
    assert "OOD DETECTION" in stage16["prohibited_terms"]
    assert stage16["ood_status"] == "OOD_WITHHELD"
    assert stage16["stage13_selective_prediction"] == "NOT_ACCEPTED"


def test_stage22b_fusion_and_triage_cannot_be_upgraded() -> None:
    fusion = config()["contract"]["stage10_11"]
    assert fusion["maximum_support"] == "PARTIALLY_SUPPORTED"
    assert fusion["required_identity"] == "EXACT_GOVERNED_IDENTITY"
    assert "FULL_SUPPORT" in fusion["prohibited"]
    triage = config()["contract"]["stage17"]
    assert triage["allowed_decisions"] == ["DEFER"]
    assert {"ROUTINE", "PRIORITY", "URGENT", "CRITICAL"} <= set(triage["prohibited"])


def test_stage22b_report_panel_is_deterministic_and_nonclinical() -> None:
    report = config()["contract"]["stage18"]
    assert report["title"] == "AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW"
    assert report["disclaimer"] == (
        "Research use only. Not a medical diagnosis. Expert review is required."
    )
    assert "LLM_GENERATION" in report["prohibited"]
    assert "FREE_FORM_AI_REWRITING" in report["prohibited"]


def test_stage22b_verifier_wording_covers_exact_statuses_without_equivalence() -> None:
    wording = result()["verifier_status_wording"]
    assert set(wording) == EXPECTED_VERIFIER_STATUSES
    assert "Not equivalent to verified" in wording["PARTIALLY_VERIFIED"]
    assert "Withheld" in wording["WITHHELD_INSUFFICIENT_EVIDENCE"]
    assert "explicit accepted structured conflict" in wording["CONTRADICTED"]
    assert not config()["contract"]["stage19"]["missing_evidence_is_contradiction"]


def test_stage22b_decision_precedence_and_labels_are_exact() -> None:
    stage20 = config()["contract"]["stage20"]
    assert stage20["precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert "NOT CLINICAL APPROVAL" in stage20["accept_label"]
    assert stage20["revise_label"] == "DETERMINISTIC CANONICAL TEMPLATE REPAIR ONLY."
    assert not stage20["clinical_urgency_language"]


def test_stage22b_defer_and_failed_sanitized_are_distinct() -> None:
    stage21 = result()["failure_display_contract"]
    assert stage21["safety_disposition"]["state"] == "DEFER"
    assert stage21["technical_disposition"]["state"] == "FAILED_SANITIZED"
    assert set(stage21["prohibited_output"]) == {
        "STACK_TRACE",
        "RAW_EXCEPTION",
        "INTERNAL_FILESYSTEM_PATH",
        "CHECKPOINT_PATH",
    }


def test_stage22b_provenance_is_safe_and_identity_free() -> None:
    provenance = result()["provenance_contract"]
    assert "SOURCE_STAGE" in provenance["safe_fields"]
    assert "EVIDENCE_CODE" in provenance["safe_fields"]
    assert "LOCAL_PATH" in provenance["prohibited_fields"]
    assert "PATIENT_IDENTITY" in provenance["prohibited_fields"]


def test_stage22b_visual_semantics_are_not_diagnostic_or_color_only() -> None:
    visual = result()["visual_semantics"]
    assert visual["colors_represent_ui_state_only"]
    assert visual["required_non_color_distinction"]
    assert "RED_EQUALS_DISEASE" in visual["prohibited"]
    assert "GREEN_EQUALS_HEALTHY" in visual["prohibited"]
    assert "TRAFFIC_LIGHT_DIAGNOSIS" in visual["prohibited"]


def test_stage22b_accessibility_privacy_and_technology_are_complete() -> None:
    output = result()
    assert "SEMANTIC_HEADINGS" in output["accessibility"]
    assert "KEYBOARD_ACCESSIBLE_CONTROLS" in output["accessibility"]
    assert "NO_PATIENT_IDENTIFYING_INFORMATION" in output["privacy"]
    assert "NO_TELEMETRY" in output["privacy"]
    assert "STATIC_HTML" in output["technology"]
    assert "VANILLA_JAVASCRIPT" in output["technology"]
    assert "NO_NPM_NODE" in output["technology"]
    assert "NO_CDN" in output["technology"]


def test_stage22b_prepares_all_synthetic_fixture_contracts() -> None:
    fixtures = result()["synthetic_fixture_specs"]
    assert len(fixtures) == 16
    assert "RESEARCH_ONLY_BANNER" in fixtures
    assert "DICOM_WITHHELD" in fixtures
    assert "OVERLAY_WITHHELD" in fixtures
    assert "DETERMINISTIC_RENDERING" in fixtures
    assert "ACCESSIBILITY_REQUIRED_LABELS" in fixtures


@pytest.mark.parametrize(
    "key",
    [
        "ui_implementation_permitted",
        "synthetic_image_rendering_permitted",
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "patient_processing_permitted",
        "locked_test_access_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage22b_rejects_execution_scope_expansion(key: str) -> None:
    cfg = config()
    cfg[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        validate_contract(cfg, ROOT)


def test_stage22b_next_stage_is_synthetic_ui_only() -> None:
    output = result()
    assert output["next_canonical_stage"] == ("22C_SYNTHETIC_RESEARCH_UI_IMPLEMENTATION_VALIDATION")
    assert output["next_stage_authorizes_ui_implementation"]
    assert output["next_stage_authorizes_synthetic_image_viewer_rendering"]
    assert not output["next_stage_authorizes_real_image_display"]
    assert not output["next_stage_authorizes_real_model_inference"]
    assert not output["next_stage_authorizes_language_model_work"]
    assert output["currently_planned_llm_authorized_gate"] is None


def test_stage22b_validation_does_not_write_summary() -> None:
    output = ROOT / "reports/stage22/stage22b_research_ui_display_contract_summary.json"
    before = output.read_bytes() if output.exists() else None
    validate_contract(config(), ROOT)
    after = output.read_bytes() if output.exists() else None
    assert after == before
