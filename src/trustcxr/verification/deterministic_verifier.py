from __future__ import annotations

from typing import Any

from trustcxr.reporting.grounded_contract import render_report, validate_statement

WITHHELD_ANATOMICAL = {
    "RELIABLE_POSITIVE_LESION_LOCALIZATION",
    "FINDING_LATERALITY_FROM_LOCALIZATION",
    "NEGATION_FROM_LOCALIZATION_ABSENCE",
    "BOUNDING_BOX_AS_PIXEL_MASK",
    "UNGOVERNED_CROSS_SOURCE_ANATOMICAL_JOIN",
}
PROXY_ANATOMICAL = {
    "STAGE8_QUALITY_FILTERED_PSEUDO_LUNG_HEART_MASKS",
    "STAGE10_IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY",
}
UNSUPPORTED_DOMAINS = {
    "severity",
    "temporal_change",
    "ood",
    "device_localization",
    "treatment",
    "patient_history",
    "clinical_certainty",
    "clinical_diagnosis",
}


def verify_textual(
    statement: dict[str, Any],
    rendered_text: str,
    contract: dict[str, Any],
    *,
    evidence_available: bool,
    exact_identity: bool,
    explicit_accepted_conflict: bool = False,
) -> str:
    evidence_type = statement.get("evidence_type")
    if evidence_type in contract["statement_policy"]["must_omit"]:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    if not exact_identity:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    try:
        validate_statement(statement, contract)
    except ValueError:
        return "UNVERIFIED"
    if not evidence_available:
        return "UNVERIFIED"
    if explicit_accepted_conflict:
        return "CONTRADICTED"
    payload = {
        "report_identity": contract["report_identity"],
        "research_use_disclaimer": contract["research_use_disclaimer"],
        "statements": [statement],
        "omitted_capabilities": [],
    }
    expected = render_report(payload, contract)["statements"][0]["text"]
    if rendered_text != expected:
        return "UNVERIFIED"
    if statement["grounding_status"] == "EXPLICIT_UNCERTAINTY":
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def verify_anatomical(
    capability: str,
    *,
    exact_identity: bool,
    within_governed_scope: bool,
    evidence_available: bool,
    explicit_accepted_conflict: bool = False,
    absence_only: bool = False,
    applicable: bool = True,
) -> str:
    if not applicable:
        return "NOT_APPLICABLE"
    if capability in UNSUPPORTED_DOMAINS:
        return "NOT_APPLICABLE"
    if not exact_identity or not within_governed_scope:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    if capability in WITHHELD_ANATOMICAL:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    if not evidence_available:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    if absence_only:
        return "WITHHELD_INSUFFICIENT_EVIDENCE"
    if explicit_accepted_conflict:
        return "CONTRADICTED"
    if capability in PROXY_ANATOMICAL:
        return "PARTIALLY_VERIFIED"
    if capability == "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY":
        return "VERIFIED"
    return "WITHHELD_INSUFFICIENT_EVIDENCE"
