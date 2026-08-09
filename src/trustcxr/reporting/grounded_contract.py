from __future__ import annotations

from typing import Any

PROVENANCE_FIELDS = {
    "source_stage",
    "source_version",
    "evidence_code",
    "structured_source_field",
}
PATIENT_IDENTIFYING_KEYS = {
    "patient_id",
    "patient_name",
    "mrn",
    "date_of_birth",
    "dicom_path",
    "local_path",
}
TEMPLATE_FOR_EVIDENCE = {
    "model_identified_view_ap_pa_or_lateral": "VIEW_IDENTIFIED",
    "technical_quality_proxy_warning_with_nonclinical_qualifier": "TECHNICAL_PROXY_WARNING",
    "research_system_defer_status_with_reason_code": "RESEARCH_DEFER",
    "stage9_classifier_finding_signal": "CLASSIFIER_SIGNAL_UNCERTAIN",
    "predictive_probability": "CLASSIFIER_SIGNAL_UNCERTAIN",
    "stage11_uncertain_or_unlocalized_fusion_status": "FUSION_STATUS_UNCERTAIN",
}


def validate_statement(statement: dict[str, Any], contract: dict[str, Any]) -> None:
    missing = PROVENANCE_FIELDS - set(statement)
    if missing:
        raise ValueError(f"Grounding provenance missing: {sorted(missing)}")
    policy = contract["statement_policy"]
    evidence_type = statement.get("evidence_type")
    if evidence_type in policy["must_omit"]:
        raise ValueError(f"Forbidden report claim: {evidence_type}")
    if evidence_type not in policy["direct_structured"] + policy["requires_explicit_uncertainty"]:
        raise ValueError(f"Unsupported evidence type: {evidence_type}")
    if evidence_type in policy["requires_explicit_uncertainty"]:
        if statement.get("grounding_status") != "EXPLICIT_UNCERTAINTY":
            raise ValueError("Uncertain evidence requires explicit uncertainty status.")
    elif statement.get("grounding_status") != "DIRECT_STRUCTURED":
        raise ValueError("Direct evidence requires direct structured status.")
    if statement.get("template_id") != TEMPLATE_FOR_EVIDENCE[evidence_type]:
        raise ValueError("Statement template is not frozen in the contract.")
    if not isinstance(statement.get("template_parameters"), dict):
        raise ValueError("Statement requires structured template parameters.")
    if not all(statement[field] for field in PROVENANCE_FIELDS):
        raise ValueError("Grounding provenance fields must be non-empty.")


def validate_payload(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    if payload.get("report_identity") != contract["report_identity"]:
        raise ValueError("Research report identity changed.")
    if payload.get("research_use_disclaimer") != contract["research_use_disclaimer"]:
        raise ValueError("Research-use disclaimer changed.")
    keys = set(payload)
    if keys & PATIENT_IDENTIFYING_KEYS:
        raise ValueError("Patient-identifying fields are prohibited.")
    for statement in payload.get("statements", []):
        if set(statement) & PATIENT_IDENTIFYING_KEYS:
            raise ValueError("Patient-identifying statement fields are prohibited.")
        validate_statement(statement, contract)
    for omission in payload.get("omitted_capabilities", []):
        if omission.get("capability") not in contract["statement_policy"]["must_omit"]:
            raise ValueError("Omission metadata contains an unknown capability.")
        if not omission.get("reason_code"):
            raise ValueError("Omitted capability requires a reason code.")
