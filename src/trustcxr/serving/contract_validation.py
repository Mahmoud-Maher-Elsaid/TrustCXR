from __future__ import annotations

from typing import Any

RESEARCH_DESIGNATION = "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"
SUBMISSION_REQUIRED = {"schema_version", "input_token", "pipeline_version", "idempotency_key"}
SUBMISSION_FORBIDDEN = {
    "patient_id",
    "patient_name",
    "mrn",
    "path",
    "url",
    "checkpoint",
    "model_name",
    "model_version",
    "free_text",
    "prompt",
}
WORKER_FORBIDDEN = {
    "python_code",
    "path",
    "checkpoint_path",
    "url",
    "model_name",
    "arbitrary_arguments",
}


def validate_submission(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    missing = sorted(SUBMISSION_REQUIRED - payload.keys())
    forbidden = sorted(SUBMISSION_FORBIDDEN & payload.keys())
    if missing:
        reasons.append("MISSING_REQUIRED_FIELDS")
    if forbidden:
        reasons.append("FORBIDDEN_REQUEST_FIELDS")
    return not reasons, tuple(reasons)


def validate_transition(
    current: str, target: str, transitions: dict[str, list[str]]
) -> tuple[bool, str]:
    if current not in transitions or target not in transitions[current]:
        return False, "ILLEGAL_JOB_STATE_TRANSITION"
    return True, "LEGAL_JOB_STATE_TRANSITION"


def validate_worker_request(
    payload: dict[str, Any], allowed_components: set[str]
) -> tuple[bool, tuple[str, ...]]:
    required = {
        "schema_version",
        "job_id",
        "component_id",
        "input_token",
        "server_model_version",
        "request_fingerprint",
    }
    reasons: list[str] = []
    if required - payload.keys():
        reasons.append("MISSING_WORKER_FIELDS")
    if WORKER_FORBIDDEN & payload.keys():
        reasons.append("FORBIDDEN_WORKER_FIELDS")
    if payload.get("component_id") not in allowed_components:
        reasons.append("UNAPPROVED_COMPONENT")
    return not reasons, tuple(reasons)


def sanitized_failure(job_id: str, reason_code: str) -> dict[str, Any]:
    return {
        "schema_version": "trustcxr-serving-contract-v1",
        "job_id": job_id,
        "state": "FAILED_SANITIZED",
        "disposition": "TECHNICAL_FAILURE",
        "reason_codes": [reason_code],
        "research_designation": RESEARCH_DESIGNATION,
    }


def validate_public_response(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    forbidden = {
        "patient_id",
        "patient_name",
        "mrn",
        "internal_path",
        "checkpoint_path",
        "stack_trace",
        "raw_exception",
        "clinical_approval",
        "autonomous_release",
        "treatment_recommendation",
        "clinical_diagnosis",
        "severity",
        "temporal_change",
        "ood_claim",
    }
    if forbidden & payload.keys():
        return False, ("FORBIDDEN_RESPONSE_FIELDS",)
    if payload.get("research_designation") != RESEARCH_DESIGNATION:
        return False, ("MISSING_RESEARCH_DESIGNATION",)
    return True, ()
