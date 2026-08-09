from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21c_synthetic_api_worker_contract_validation import validate

from trustcxr.serving.contract_validation import (
    failure_response,
    pseudonymous_job_id,
    serving_decision,
    validate_public_response,
    validate_server_provenance,
    validate_submission,
    validate_temporary_artifacts,
    validate_transition,
    validate_worker_request,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21c_synthetic_api_worker_contract_validation.json"
SUMMARY = json.loads(
    (ROOT / "reports/stage21/stage21b_backend_api_worker_contract_summary.json").read_text(
        encoding="utf-8"
    )
)


def test_stage21c_prepared_fixture_matrix_passes_without_activation() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["fixtures_passed"] >= 60
    assert result["fixtures_failed"] == 0
    assert not result["server_started"]
    assert not result["worker_started"]
    assert not result["model_loaded"]
    assert not result["model_inference_performed"]
    assert result["locked_test_records_accessed"] == 0
    assert not result["language_model_used"]


def test_api_validation_is_closed_to_unknown_sensitive_and_malformed_fields() -> None:
    base = {
        "schema_version": "trustcxr-serving-contract-v1",
        "input_token": "synthetic",
        "pipeline_version": "frozen",
        "idempotency_key": "synthetic",
    }
    assert validate_submission(base)[0]
    for field in ("patient_id", "raw_phi", "path", "url", "free_text", "model_name"):
        valid, reasons = validate_submission({**base, field: "prohibited"})
        assert not valid
        assert "FORBIDDEN_REQUEST_FIELDS" in reasons
    assert not validate_submission({**base, "input_token": []})[0]


def test_all_legal_transitions_pass_and_terminal_states_are_terminal() -> None:
    transitions = SUMMARY["job_state_machine"]["transitions"]
    for source, targets in transitions.items():
        for target in targets:
            assert validate_transition(source, target, transitions)[0]
    for terminal in ("COMPLETED", "DEFERRED", "FAILED_SANITIZED"):
        assert transitions[terminal] == []
        assert not validate_transition(terminal, "GPU_PROCESSING", transitions)[0]


def test_worker_requires_frozen_version_hashes_and_immutable_checkpoint() -> None:
    frozen = {"stage9_classifier": "frozen:stage9_classifier"}
    request = {
        "schema_version": "trustcxr-serving-contract-v1",
        "job_id": pseudonymous_job_id("synthetic"),
        "component_id": "stage9_classifier",
        "input_token": "synthetic",
        "server_model_version": frozen["stage9_classifier"],
        "request_fingerprint": "synthetic",
    }
    assert validate_worker_request(request, frozen)[0]
    assert not validate_worker_request({**request, "server_model_version": "arbitrary"}, frozen)[0]
    provenance = {"config_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    assert validate_server_provenance(provenance, provenance, False)[0]
    assert not validate_server_provenance(provenance, provenance, True)[0]


def test_failure_semantics_keep_technical_failure_distinct_from_defer() -> None:
    assert failure_response("job_synthetic", "CUDA_OOM")["state"] == "FAILED_SANITIZED"
    assert failure_response("job_synthetic", "INFERENCE_FAILURE")["state"] == "FAILED_SANITIZED"
    assert failure_response("job_synthetic", "PROVENANCE_FAILURE")["state"] == "DEFERRED"
    assert failure_response("job_synthetic", "VERIFIER_FAILURE")["state"] == "DEFERRED"


def test_privacy_temporary_storage_and_defer_precedence_are_fail_closed() -> None:
    response = failure_response("job_synthetic", "CUDA_UNAVAILABLE")
    assert validate_public_response(response)[0]
    assert not validate_public_response({**response, "stack_trace": "prohibited"})[0]
    valid, state, reasons = validate_temporary_artifacts(
        {"job_a": "same", "job_b": "same"},
        tracked_phi=False,
        cleanup_complete=True,
        crash_cleanup_complete=True,
        cleanup_failure=False,
    )
    assert not valid and state == "FAILED_SANITIZED"
    assert "REQUEST_STORAGE_NOT_ISOLATED" in reasons
    decision, _ = serving_decision(
        stage17_defer=True,
        stage19_statuses=["VERIFIED"],
        stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    )
    assert decision == "DEFER"
