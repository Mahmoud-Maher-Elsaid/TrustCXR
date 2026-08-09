from __future__ import annotations

import hashlib
import json
from typing import Any

RESEARCH_DESIGNATION = "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"
SCHEMA_VERSION = "trustcxr-serving-contract-v1"
SUBMISSION_REQUIRED = {"schema_version", "input_token", "pipeline_version", "idempotency_key"}
SUBMISSION_FORBIDDEN = {
    "accession",
    "address",
    "checkpoint",
    "date_of_birth",
    "dob",
    "free_text",
    "job_id",
    "model_name",
    "model_version",
    "mrn",
    "name",
    "path",
    "patient_id",
    "patient_name",
    "prompt",
    "raw_phi",
    "study_uid",
    "url",
}
WORKER_REQUIRED = {
    "schema_version",
    "job_id",
    "component_id",
    "input_token",
    "server_model_version",
    "request_fingerprint",
}
WORKER_FORBIDDEN = {
    "arbitrary_arguments",
    "checkpoint",
    "checkpoint_path",
    "config_path",
    "model_name",
    "path",
    "python_code",
    "url",
}
PUBLIC_RESPONSE_FORBIDDEN = {
    "autonomous_release",
    "checkpoint_path",
    "clinical_approval",
    "clinical_diagnosis",
    "device_localization",
    "internal_path",
    "mrn",
    "ood_claim",
    "patient_id",
    "patient_name",
    "raw_exception",
    "severity",
    "stack_trace",
    "temporal_change",
    "treatment_recommendation",
}
TECHNICAL_FAILURES = {
    "INVALID_REQUEST",
    "CHECKPOINT_HASH_MISMATCH",
    "CUDA_UNAVAILABLE",
    "CUDA_OOM",
    "MODEL_LOAD_FAILURE",
    "INFERENCE_FAILURE",
    "DECISION_POLICY_FAILURE",
    "CLEANUP_FAILURE",
}
SAFETY_DEFERS = {"UNSUPPORTED_INPUT", "PROVENANCE_FAILURE", "VERIFIER_FAILURE"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def pseudonymous_job_id(request_fingerprint: str) -> str:
    digest = hashlib.sha256(f"trustcxr-job-v1:{request_fingerprint}".encode()).hexdigest()
    return f"job_{digest[:24]}"


def _canonical_reasons(reasons: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(reasons)))


def validate_submission(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if SUBMISSION_REQUIRED - payload.keys():
        reasons.append("MISSING_REQUIRED_FIELDS")
    if SUBMISSION_FORBIDDEN & payload.keys() or payload.keys() - SUBMISSION_REQUIRED:
        reasons.append("FORBIDDEN_REQUEST_FIELDS")
    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("INVALID_SCHEMA_VERSION")
    for field in SUBMISSION_REQUIRED - {"schema_version"}:
        if field in payload and (not isinstance(payload[field], str) or not payload[field].strip()):
            reasons.append("MALFORMED_STRUCTURED_FIELDS")
    return not reasons, _canonical_reasons(reasons)


def validate_transition(
    current: str, target: str, transitions: dict[str, list[str]]
) -> tuple[bool, str]:
    if current not in transitions or target not in transitions[current]:
        return False, "ILLEGAL_JOB_STATE_TRANSITION"
    return True, "LEGAL_JOB_STATE_TRANSITION"


def validate_worker_request(
    payload: dict[str, Any], frozen_versions: dict[str, str]
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if WORKER_REQUIRED - payload.keys():
        reasons.append("MISSING_WORKER_FIELDS")
    if WORKER_FORBIDDEN & payload.keys() or payload.keys() - WORKER_REQUIRED:
        reasons.append("FORBIDDEN_WORKER_FIELDS")
    component = payload.get("component_id")
    if component not in frozen_versions:
        reasons.append("UNAPPROVED_COMPONENT")
    elif payload.get("server_model_version") != frozen_versions[component]:
        reasons.append("FROZEN_MODEL_VERSION_MISMATCH")
    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("INVALID_SCHEMA_VERSION")
    return not reasons, _canonical_reasons(reasons)


def validate_server_provenance(
    expected: dict[str, str], observed: dict[str, str], checkpoint_mutation: bool
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    required = {"config_sha256", "checkpoint_sha256"}
    if required - expected.keys() or required - observed.keys():
        reasons.append("MISSING_SHA256_PROVENANCE")
    elif any(observed[key] != expected[key] for key in sorted(required)):
        reasons.append("CHECKPOINT_HASH_MISMATCH")
    if checkpoint_mutation:
        reasons.append("CHECKPOINT_MUTATION_PROHIBITED")
    return not reasons, _canonical_reasons(reasons)


def failure_response(job_id: str, failure: str) -> dict[str, Any]:
    if failure in TECHNICAL_FAILURES:
        state, disposition = "FAILED_SANITIZED", "TECHNICAL_FAILURE"
    elif failure in SAFETY_DEFERS:
        state, disposition = "DEFERRED", "SAFETY_DEFER"
    else:
        state, disposition = "FAILED_SANITIZED", "TECHNICAL_FAILURE"
        failure = "UNKNOWN_TECHNICAL_FAILURE"
    return {
        "disposition": disposition,
        "job_id": job_id,
        "reason_codes": [failure],
        "research_designation": RESEARCH_DESIGNATION,
        "schema_version": SCHEMA_VERSION,
        "state": state,
    }


def validate_public_response(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if PUBLIC_RESPONSE_FORBIDDEN & payload.keys():
        reasons.append("FORBIDDEN_RESPONSE_FIELDS")
    if payload.get("research_designation") != RESEARCH_DESIGNATION:
        reasons.append("MISSING_RESEARCH_DESIGNATION")
    return not reasons, _canonical_reasons(reasons)


def validate_orchestration(
    observed_order: list[str], expected_order: list[str], capabilities: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if observed_order != expected_order:
        reasons.append("ORCHESTRATION_ORDER_MISMATCH")
    required_limits = {
        "stage11_maximum_support": "PARTIALLY_SUPPORTED",
        "reliable_positive_localization": False,
        "localization_absence_contradiction": False,
        "stage13_selective_prediction_accepted": False,
        "ood_supported": False,
        "severity_supported": False,
        "temporal_change_supported": False,
        "device_localization_supported": False,
    }
    if any(capabilities.get(key) != value for key, value in required_limits.items()):
        reasons.append("FROZEN_CAPABILITY_UPGRADE_ATTEMPT")
    return not reasons, _canonical_reasons(reasons)


def serving_decision(
    *, stage17_defer: bool, stage19_statuses: list[str], stage20_candidate: str
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if stage17_defer:
        reasons.append("ACTIVE_STAGE17_DEFER")
    if any(
        status in {"PARTIALLY_VERIFIED", "WITHHELD_INSUFFICIENT_EVIDENCE"}
        for status in stage19_statuses
    ):
        reasons.append("VERIFIER_EVIDENCE_REQUIRES_DEFER")
    if reasons or stage20_candidate == "DEFER":
        if stage20_candidate == "DEFER":
            reasons.append("STAGE20_DEFER_PRECEDENCE")
        return "DEFER", _canonical_reasons(reasons)
    return stage20_candidate, ()


def validate_temporary_artifacts(
    request_scopes: dict[str, str],
    *,
    tracked_phi: bool,
    cleanup_complete: bool,
    crash_cleanup_complete: bool,
    cleanup_failure: bool,
) -> tuple[bool, str, tuple[str, ...]]:
    reasons: list[str] = []
    if len(set(request_scopes.values())) != len(request_scopes):
        reasons.append("REQUEST_STORAGE_NOT_ISOLATED")
    if tracked_phi:
        reasons.append("TRACKED_PHI_PROHIBITED")
    if not cleanup_complete or not crash_cleanup_complete:
        reasons.append("DETERMINISTIC_CLEANUP_INCOMPLETE")
    if cleanup_failure:
        reasons.append("CLEANUP_FAILURE")
    canonical = _canonical_reasons(reasons)
    return not reasons, "FAILED_SANITIZED" if reasons else "CLEAN", canonical
