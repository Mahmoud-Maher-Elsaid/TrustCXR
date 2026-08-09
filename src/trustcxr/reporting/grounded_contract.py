from __future__ import annotations

import json
import re
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
STATEMENT_FIELDS = {
    "evidence_type",
    "grounding_status",
    "template_id",
    "template_parameters",
    *PROVENANCE_FIELDS,
}
PARAMETERS_FOR_TEMPLATE = {
    "VIEW_IDENTIFIED": {"view"},
    "TECHNICAL_PROXY_WARNING": {"warning"},
    "RESEARCH_DEFER": {"reason_codes"},
    "CLASSIFIER_SIGNAL_UNCERTAIN": {"finding", "model_score"},
    "FUSION_STATUS_UNCERTAIN": {"fusion_status"},
}
NIH_LABELS = {
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
}
DEFER_REASON_CODES = {
    "PREDICTIVE_UNCERTAINTY_ABOVE_FROZEN_LIMIT",
    "TECHNICAL_QUALITY_PROXY_FAILED_NOT_CLINICAL_QUALITY",
    "VIEW_OUTSIDE_FROZEN_STAGE5_CAPABILITY",
    "FUSION_EVIDENCE_NOT_RELIABLY_SUPPORTIVE",
    "REQUIRED_TRIAGE_EVIDENCE_MISSING",
}
ALLOWED_SOURCE_STAGES = {"5", "9", "11", "17"}
EVIDENCE_CODES_FOR_TYPE = {
    "model_identified_view_ap_pa_or_lateral": {"VIEW_MODEL_OUTPUT"},
    "technical_quality_proxy_warning_with_nonclinical_qualifier": {
        "TECHNICAL_QUALITY_PROXY_WARNING"
    },
    "research_system_defer_status_with_reason_code": {"RESEARCH_TRIAGE_DEFER"},
    "stage9_classifier_finding_signal": {"CLASSIFIER_SIGNAL"},
    "predictive_probability": {"PREDICTIVE_MODEL_SCORE"},
    "stage11_uncertain_or_unlocalized_fusion_status": {
        "FUSION_UNCERTAIN",
        "FUSION_UNLOCALIZED",
    },
}


def validate_statement(statement: dict[str, Any], contract: dict[str, Any]) -> None:
    missing = PROVENANCE_FIELDS - set(statement)
    if missing:
        raise ValueError(f"Grounding provenance missing: {sorted(missing)}")
    if set(statement) != STATEMENT_FIELDS:
        raise ValueError("Statement does not match the frozen schema fields.")
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
    template_id = statement["template_id"]
    parameters = statement["template_parameters"]
    if set(parameters) != PARAMETERS_FOR_TEMPLATE[template_id]:
        raise ValueError("Template parameters do not match the frozen template.")
    _validate_parameters(template_id, parameters)
    if not all(statement[field] for field in PROVENANCE_FIELDS):
        raise ValueError("Grounding provenance fields must be non-empty.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", statement["source_stage"]):
        raise ValueError("Invalid source stage provenance.")
    if statement["source_stage"] not in ALLOWED_SOURCE_STAGES:
        raise ValueError("Unsupported source stage provenance.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", statement["source_version"]):
        raise ValueError("Invalid source version provenance.")
    if not re.fullmatch(r"[A-Z0-9_]+", statement["evidence_code"]):
        raise ValueError("Invalid evidence code provenance.")
    if statement["evidence_code"] not in EVIDENCE_CODES_FOR_TYPE[evidence_type]:
        raise ValueError("Unsupported evidence code provenance.")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", statement["structured_source_field"]):
        raise ValueError("Invalid structured source field provenance.")


def _validate_parameters(template_id: str, parameters: dict[str, Any]) -> None:
    if template_id == "VIEW_IDENTIFIED" and parameters["view"] not in {"AP", "PA", "LATERAL"}:
        raise ValueError("Unsupported view template value.")
    if (
        template_id == "TECHNICAL_PROXY_WARNING"
        and parameters["warning"] != "TECHNICAL_PROXY_FAILED"
    ):
        raise ValueError("Unsupported technical-proxy warning value.")
    if template_id == "RESEARCH_DEFER":
        codes = parameters["reason_codes"]
        if not isinstance(codes, list) or not codes or not set(codes) <= DEFER_REASON_CODES:
            raise ValueError("Unsupported DEFER reason code.")
    if template_id == "CLASSIFIER_SIGNAL_UNCERTAIN":
        if parameters["finding"] not in NIH_LABELS:
            raise ValueError("Unsupported classifier finding.")
        score = parameters["model_score"]
        if not isinstance(score, (float, int)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError("Model score must be numeric and within [0, 1].")
    if template_id == "FUSION_STATUS_UNCERTAIN" and parameters["fusion_status"] not in {
        "UNCERTAIN",
        "UNLOCALIZED",
    }:
        raise ValueError("Unsupported Stage 11 fusion status.")


def validate_payload(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    if set(payload) != {
        "report_identity",
        "research_use_disclaimer",
        "statements",
        "omitted_capabilities",
    }:
        raise ValueError("Report payload does not match the frozen schema fields.")
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
        if not re.fullmatch(r"[A-Z0-9_]+", omission["reason_code"]):
            raise ValueError("Omission reason code must be a structured code.")


def render_report(payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(payload)
    filtered["statements"] = []
    for statement in payload.get("statements", []):
        try:
            validate_statement(statement, contract)
        except ValueError as error:
            if str(error).startswith("Grounding provenance missing") or str(error).startswith(
                "Grounding provenance fields must be non-empty"
            ):
                continue
            raise
        filtered["statements"].append(statement)
    validate_payload(filtered, contract)
    rendered = []
    for statement in filtered["statements"]:
        parameters = dict(statement["template_parameters"])
        if statement["template_id"] == "RESEARCH_DEFER":
            parameters["reason_codes"] = ", ".join(sorted(parameters["reason_codes"]))
        if statement["template_id"] == "CLASSIFIER_SIGNAL_UNCERTAIN":
            parameters["model_score"] = f"{parameters['model_score']:.6f}"
        text = contract["templates"][statement["template_id"]].format(**parameters)
        rendered.append(
            {
                "text": text,
                "grounding_status": statement["grounding_status"],
                "source_stage": statement["source_stage"],
                "source_version": statement["source_version"],
                "evidence_code": statement["evidence_code"],
                "structured_source_field": statement["structured_source_field"],
            }
        )
    rendered.sort(
        key=lambda row: (
            row["source_stage"],
            row["structured_source_field"],
            row["evidence_code"],
            row["text"],
        )
    )
    omissions = sorted(
        filtered["omitted_capabilities"],
        key=lambda row: (row["capability"], row["reason_code"]),
    )
    return {
        "report_identity": filtered["report_identity"],
        "research_use_disclaimer": filtered["research_use_disclaimer"],
        "statements": rendered,
        "omitted_capabilities": omissions,
    }


def render_report_json(payload: dict[str, Any], contract: dict[str, Any]) -> str:
    return json.dumps(render_report(payload, contract), indent=2, sort_keys=True) + "\n"
