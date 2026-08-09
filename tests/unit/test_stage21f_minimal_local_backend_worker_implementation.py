from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.serving.run_stage21f_minimal_local_backend_worker_implementation import (
    validate_implementation,
)

from trustcxr.serving.api import create_app
from trustcxr.serving.registry import FrozenComponentRegistry
from trustcxr.serving.runtime import (
    BoundedWorker,
    JobStore,
    SyntheticRuntimeState,
    TemporaryArtifactManager,
    sanitized_disposition,
)
from trustcxr.serving.schemas import (
    ComponentId,
    JobState,
    JobSubmission,
    SanitizedDisposition,
    WorkerRequest,
)
from trustcxr.serving.state_machine import TRANSITIONS, IllegalTransitionError, transition

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21f_minimal_local_backend_worker_implementation.json"


def registry() -> FrozenComponentRegistry:
    return FrozenComponentRegistry.from_stage21b(ROOT)


def worker_request(component_id: ComponentId = ComponentId.STAGE9) -> WorkerRequest:
    component = registry().resolve(component_id)
    return WorkerRequest(
        job_id="job_0123456789abcdef01234567",
        component_id=component_id,
        input_token="synthetic-input",
        server_model_version=component.server_model_version,
        request_fingerprint="a" * 64,
        config_sha256=component.config_sha256,
        checkpoint_sha256=component.checkpoint_sha256,
    )


def test_stage21f_valid_request_is_deterministic_and_pseudonymous() -> None:
    submission = JobSubmission(input_token="synthetic-input", idempotency_key="same-key")
    store = JobStore()
    first = store.submit(submission)
    second = store.submit(submission)
    assert first == second
    assert first.job_id.startswith("job_")
    assert "synthetic-input" not in first.job_id


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"input_token": "https://example.invalid", "idempotency_key": "key"},
        {"input_token": "C:\\private\\file", "idempotency_key": "key"},
        {"input_token": "synthetic", "idempotency_key": "key", "patient_id": "prohibited"},
        {"input_token": "synthetic", "idempotency_key": "key", "model_name": "arbitrary"},
    ],
)
def test_stage21f_invalid_sensitive_or_arbitrary_request_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        JobSubmission.model_validate(payload)


def test_stage21f_state_machine_accepts_legal_and_rejects_illegal_transitions() -> None:
    for source, targets in TRANSITIONS.items():
        for target in targets:
            assert transition(source, target) == target
    with pytest.raises(IllegalTransitionError):
        transition(JobState.SUBMITTED, JobState.COMPLETED)
    for terminal in (JobState.COMPLETED, JobState.DEFERRED, JobState.FAILED_SANITIZED):
        assert not TRANSITIONS[terminal]


def test_stage21f_registry_is_complete_immutable_and_publicly_path_free() -> None:
    frozen = registry()
    assert len(frozen) == 8
    assert {item["component_id"] for item in frozen.public_registry()} == {
        component.value for component in ComponentId
    }
    assert all("path" not in key for item in frozen.public_registry() for key in item)


@pytest.mark.parametrize(
    ("failure", "expected_state"),
    [
        ("CUDA_UNAVAILABLE", JobState.FAILED_SANITIZED),
        ("CUDA_OOM", JobState.FAILED_SANITIZED),
        ("MODEL_LOAD_FAILURE", JobState.FAILED_SANITIZED),
        ("INFERENCE_FAILURE", JobState.FAILED_SANITIZED),
        ("PROVENANCE_FAILURE", JobState.DEFERRED),
        ("VERIFIER_FAILURE", JobState.DEFERRED),
        ("DECISION_POLICY_FAILURE", JobState.FAILED_SANITIZED),
    ],
)
def test_stage21f_failure_mapping_is_exact_and_sanitized(
    failure: str, expected_state: JobState
) -> None:
    response = sanitized_disposition("job_0123456789abcdef01234567", failure)
    assert response.state == expected_state
    dumped = response.model_dump(mode="json")
    assert "stack_trace" not in dumped
    assert "internal_path" not in dumped


def test_stage21f_worker_rejects_config_checkpoint_and_cuda_failures() -> None:
    frozen = registry()
    worker = BoundedWorker(frozen)
    request = worker_request()
    bad_config = request.model_copy(update={"config_sha256": "b" * 64})
    assert isinstance(
        worker.validate_synthetic(bad_config, SyntheticRuntimeState()), SanitizedDisposition
    )
    bad_checkpoint = request.model_copy(update={"checkpoint_sha256": "c" * 64})
    checkpoint_result = worker.validate_synthetic(bad_checkpoint, SyntheticRuntimeState())
    assert isinstance(checkpoint_result, SanitizedDisposition)
    assert checkpoint_result.reason_codes == ("CHECKPOINT_HASH_MISMATCH",)
    cuda_result = worker.validate_synthetic(request, SyntheticRuntimeState(cuda_available=False))
    assert isinstance(cuda_result, SanitizedDisposition)
    assert cuda_result.reason_codes == ("CUDA_UNAVAILABLE",)
    assert worker.resident_models == 0


def test_stage21f_worker_rejects_arbitrary_component_and_path_fields() -> None:
    payload = worker_request().model_dump(mode="json")
    payload["component_id"] = "arbitrary"
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(payload)
    payload = worker_request().model_dump(mode="json")
    payload["checkpoint_path"] = "C:\\private\\checkpoint.pt"
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(payload)


def test_stage21f_temporary_storage_is_isolated_and_cleanup_is_deterministic(
    tmp_path: Path,
) -> None:
    manager = TemporaryArtifactManager(tmp_path / "runtime")
    first = manager.create("job_aaaaaaaaaaaaaaaaaaaaaaaa")
    second = manager.create("job_bbbbbbbbbbbbbbbbbbbbbbbb")
    assert first != second
    manager.cleanup(first.name)
    assert not first.exists() and second.exists()
    assert manager.crash_recovery_cleanup() == 1
    assert not second.exists()


def test_stage21f_stage17_stage19_and_stage20_defer_cannot_be_upgraded() -> None:
    from trustcxr.serving.contract_validation import serving_decision

    assert (
        serving_decision(
            stage17_defer=True,
            stage19_statuses=["VERIFIED"],
            stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
        )[0]
        == "DEFER"
    )
    for status in ("PARTIALLY_VERIFIED", "WITHHELD_INSUFFICIENT_EVIDENCE"):
        assert (
            serving_decision(
                stage17_defer=False,
                stage19_statuses=[status],
                stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            )[0]
            == "DEFER"
        )
    assert (
        serving_decision(
            stage17_defer=False, stage19_statuses=["VERIFIED"], stage20_candidate="DEFER"
        )[0]
        == "DEFER"
    )


def test_stage21f_api_surface_is_minimal_and_has_no_llm_endpoint() -> None:
    app = create_app()
    routes = {(method, route.path) for route in app.routes for method in (route.methods or set())}
    api_routes = {item for item in routes if item[1].startswith("/v1/") or item[1] == "/health"}
    assert api_routes == {
        ("POST", "/v1/jobs"),
        ("GET", "/v1/jobs/{job_id}"),
        ("GET", "/health"),
    }
    assert all("chat" not in path and "llm" not in path for _, path in routes)


def test_stage21f_prepared_implementation_validation_has_no_runtime_execution() -> None:
    result = validate_implementation(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["frozen_component_count"] == 8
    assert not result["server_started"]
    assert not result["persistent_worker_started"]
    assert not result["real_model_loaded"]
    assert not result["real_model_inference_performed"]
    assert result["locked_test_records_accessed"] == 0
    assert not result["language_model_used"]
