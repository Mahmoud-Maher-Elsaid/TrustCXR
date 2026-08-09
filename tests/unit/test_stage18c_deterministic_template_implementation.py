from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustcxr.reporting.grounded_contract import render_report, render_report_json

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "reports/stage18/stage18b_deterministic_grounded_report_contract_summary.json"
)


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def statement(evidence_type: str, template_id: str, parameters: dict) -> dict:
    uncertain = evidence_type in contract()["statement_policy"]["requires_explicit_uncertainty"]
    source_stage = {
        "model_identified_view_ap_pa_or_lateral": "5",
        "technical_quality_proxy_warning_with_nonclinical_qualifier": "5",
        "stage9_classifier_finding_signal": "9",
        "stage11_uncertain_or_unlocalized_fusion_status": "11",
        "research_system_defer_status_with_reason_code": "17",
    }.get(evidence_type, "9")
    evidence_code = {
        "model_identified_view_ap_pa_or_lateral": "VIEW_MODEL_OUTPUT",
        "technical_quality_proxy_warning_with_nonclinical_qualifier": (
            "TECHNICAL_QUALITY_PROXY_WARNING"
        ),
        "stage9_classifier_finding_signal": "CLASSIFIER_SIGNAL",
        "stage11_uncertain_or_unlocalized_fusion_status": "FUSION_UNLOCALIZED",
        "research_system_defer_status_with_reason_code": "RESEARCH_TRIAGE_DEFER",
    }.get(evidence_type, "CLASSIFIER_SIGNAL")
    return {
        "evidence_type": evidence_type,
        "grounding_status": "EXPLICIT_UNCERTAINTY" if uncertain else "DIRECT_STRUCTURED",
        "template_id": template_id,
        "template_parameters": parameters,
        "source_stage": source_stage,
        "source_version": "fixture-v1",
        "evidence_code": evidence_code,
        "structured_source_field": "fixture/value",
    }


def payload(statements: list[dict]) -> dict:
    return {
        "report_identity": contract()["report_identity"],
        "research_use_disclaimer": contract()["research_use_disclaimer"],
        "statements": statements,
        "omitted_capabilities": [
            {"capability": "severity", "reason_code": "WITHHELD_NO_VALID_EVIDENCE"}
        ],
    }


def test_valid_grounded_rendering_preserves_provenance_and_uncertainty() -> None:
    source = statement(
        "stage9_classifier_finding_signal",
        "CLASSIFIER_SIGNAL_UNCERTAIN",
        {"finding": "Atelectasis", "model_score": 0.42},
    )
    rendered = render_report(payload([source]), contract())["statements"][0]
    assert "uncertain Atelectasis signal" in rendered["text"]
    assert "not a clinical diagnosis" in rendered["text"]
    assert rendered["source_stage"] == "9"
    assert rendered["evidence_code"] == "CLASSIFIER_SIGNAL"


def test_fusion_defer_and_technical_proxy_templates_are_exact() -> None:
    statements = [
        statement(
            "stage11_uncertain_or_unlocalized_fusion_status",
            "FUSION_STATUS_UNCERTAIN",
            {"fusion_status": "UNLOCALIZED"},
        ),
        statement(
            "research_system_defer_status_with_reason_code",
            "RESEARCH_DEFER",
            {"reason_codes": ["REQUIRED_TRIAGE_EVIDENCE_MISSING"]},
        ),
        statement(
            "technical_quality_proxy_warning_with_nonclinical_qualifier",
            "TECHNICAL_PROXY_WARNING",
            {"warning": "TECHNICAL_PROXY_FAILED"},
        ),
    ]
    texts = [row["text"] for row in render_report(payload(statements), contract())["statements"]]
    assert "reliable localization support is not established" in texts[0]
    assert "REQUIRED_TRIAGE_EVIDENCE_MISSING" in texts[1]
    assert "not a clinical image-quality assessment" in texts[2]


def test_forbidden_claims_rejected_and_missing_grounding_omitted() -> None:
    forbidden = statement("severity", "CLASSIFIER_SIGNAL_UNCERTAIN", {})
    with pytest.raises(ValueError, match="Forbidden"):
        render_report(payload([forbidden]), contract())
    ungrounded = statement(
        "model_identified_view_ap_pa_or_lateral", "VIEW_IDENTIFIED", {"view": "PA"}
    )
    ungrounded.pop("evidence_code")
    assert render_report(payload([ungrounded]), contract())["statements"] == []


def test_output_is_reproducible_and_free_text_injection_is_blocked() -> None:
    source = statement(
        "stage9_classifier_finding_signal",
        "CLASSIFIER_SIGNAL_UNCERTAIN",
        {"finding": "Atelectasis", "model_score": 0.42},
    )
    assert render_report_json(payload([source]), contract()) == render_report_json(
        payload([source]), contract()
    )
    injected = source | {
        "template_parameters": {"finding": "Atelectasis; diagnose", "model_score": 0.42}
    }
    with pytest.raises(ValueError, match="finding"):
        render_report(payload([injected]), contract())


def test_patient_identifiers_and_schema_extras_are_rejected() -> None:
    source = statement("model_identified_view_ap_pa_or_lateral", "VIEW_IDENTIFIED", {"view": "AP"})
    with pytest.raises(ValueError, match="schema fields"):
        render_report(payload([source]) | {"extra": "not allowed"}, contract())
    with pytest.raises(ValueError, match="Patient-identifying"):
        render_report(payload([source | {"patient_id": "prohibited"}]), contract())
