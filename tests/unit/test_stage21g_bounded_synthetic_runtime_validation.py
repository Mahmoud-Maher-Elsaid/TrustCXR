from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.serving.run_stage21g_bounded_synthetic_runtime_validation import (
    _assert_sanitized,
    asgi_request,
    worker_request,
)

from trustcxr.serving.api import create_app
from trustcxr.serving.contract_validation import serving_decision, validate_orchestration
from trustcxr.serving.registry import FrozenComponentRegistry
from trustcxr.serving.runtime import (
    BoundedWorker,
    JobStore,
    SyntheticRuntimeState,
    TemporaryArtifactManager,
    sanitized_disposition,
)
from trustcxr.serving.schemas import ComponentId, JobState, JobSubmission, WorkerRequest
from trustcxr.serving.state_machine import IllegalTransitionError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/serving/stage21g_bounded_synthetic_runtime_validation.json").read_text(
        encoding="utf-8"
    )
)


def registry() -> FrozenComponentRegistry:
    return FrozenComponentRegistry.from_stage21b(ROOT)


def request(app, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    return asyncio.run(asgi_request(app, method, path, payload))


def test_stage21g_health_submission_status_and_idempotency_runtime() -> None:
    store = JobStore()
    app = create_app(store)
    code, health = request(app, "GET", "/health")
    assert code == 200 and health["status"] == "READY_CONTRACT_ONLY"
    payload = JobSubmission(
        input_token="synthetic-input", idempotency_key="runtime-key"
    ).model_dump(mode="json")
    code, created = request(app, "POST", "/v1/jobs", payload)
    assert code == 202 and created["job_id"].startswith("job_")
    assert "synthetic-input" not in created["job_id"]
    status_code, status = request(app, "GET", f"/v1/jobs/{created['job_id']}")
    assert status_code == 200 and status == created
    repeated_code, repeated = request(app, "POST", "/v1/jobs", payload)
    assert repeated_code == 202 and repeated == created


def test_stage21g_nonidentical_requests_receive_distinct_job_ids() -> None:
    app = create_app()
    base = JobSubmission(input_token="synthetic-input", idempotency_key="key")
    _, first = request(app, "POST", "/v1/jobs", base.model_dump(mode="json"))
    changed = base.model_copy(update={"idempotency_key": "different-key"})
    _, second = request(app, "POST", "/v1/jobs", changed.model_dump(mode="json"))
    assert first["job_id"] != second["job_id"]


@pytest.mark.parametrize(
    "extra",
    [
        {"model_name": "arbitrary"},
        {"checkpoint_path": "C:/private/model.pt"},
        {"url": "https://example.invalid"},
        {"free_text": "print('x')"},
        {"patient_id": "patient-1"},
        {"raw_phi": "name"},
    ],
)
def test_stage21g_runtime_rejects_unsafe_api_fields(extra: dict[str, str]) -> None:
    app = create_app()
    payload = JobSubmission(input_token="synthetic-input", idempotency_key="key").model_dump(
        mode="json"
    )
    code, response = request(app, "POST", "/v1/jobs", {**payload, **extra})
    assert code == 422
    assert response["state"] == "FAILED_SANITIZED"
    _assert_sanitized(response)


def test_stage21g_unknown_job_is_sanitized() -> None:
    code, response = request(create_app(), "GET", "/v1/jobs/job_unknown")
    assert code == 404 and response["state"] == "FAILED_SANITIZED"
    _assert_sanitized(response)


def test_stage21g_terminal_states_reject_runtime_transition() -> None:
    for terminal in (JobState.DEFERRED, JobState.FAILED_SANITIZED):
        store = JobStore()
        status = store.submit(
            JobSubmission(input_token=f"synthetic-{terminal.value}", idempotency_key="key")
        )
        store.advance(status.job_id, JobState.VALIDATING)
        store.advance(status.job_id, terminal)
        with pytest.raises(IllegalTransitionError):
            store.advance(status.job_id, JobState.GPU_PROCESSING)


def test_stage21g_completed_rejects_partial_success_or_reentry() -> None:
    store = JobStore()
    status = store.submit(JobSubmission(input_token="synthetic-complete", idempotency_key="key"))
    for state in (
        JobState.VALIDATING,
        JobState.QUEUED,
        JobState.GPU_PROCESSING,
        JobState.CPU_POSTPROCESSING,
        JobState.VERIFYING,
        JobState.DECIDING,
        JobState.COMPLETED,
    ):
        store.advance(status.job_id, state)
    with pytest.raises(IllegalTransitionError):
        store.advance(status.job_id, JobState.GPU_PROCESSING)


@pytest.mark.parametrize(
    ("failure", "state"),
    [
        ("CUDA_OOM", JobState.FAILED_SANITIZED),
        ("MODEL_LOAD_FAILURE", JobState.FAILED_SANITIZED),
        ("INFERENCE_FAILURE", JobState.FAILED_SANITIZED),
        ("PROVENANCE_FAILURE", JobState.DEFERRED),
        ("VERIFIER_FAILURE", JobState.DEFERRED),
        ("DECISION_POLICY_FAILURE", JobState.FAILED_SANITIZED),
        ("CLEANUP_FAILURE", JobState.FAILED_SANITIZED),
        ("UNSUPPORTED_INPUT", JobState.DEFERRED),
    ],
)
def test_stage21g_exact_failure_semantics(failure: str, state: JobState) -> None:
    result = sanitized_disposition("job_0123456789abcdef01234567", failure)
    assert result.state == state and result.reason_codes == (failure,)


def test_stage21g_worker_runtime_never_loads_a_model() -> None:
    frozen = registry()
    worker = BoundedWorker(frozen)
    valid = worker_request(frozen, "job_0123456789abcdef01234567")
    response = worker.validate_synthetic(valid, SyntheticRuntimeState())
    assert response.status == "SUCCESS"
    assert response.reason_codes == ("SYNTHETIC_NO_MODEL_EXECUTION",)
    assert worker.resident_models == 0


def test_stage21g_worker_rejects_unknown_component_and_arbitrary_path() -> None:
    payload = worker_request(registry(), "job_0123456789abcdef01234567").model_dump(mode="json")
    payload["component_id"] = "arbitrary"
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(payload)
    payload = worker_request(registry(), "job_0123456789abcdef01234567").model_dump(mode="json")
    payload["checkpoint_path"] = "C:/private/checkpoint.pt"
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(payload)


def test_stage21g_worker_runtime_hash_cuda_and_simulated_failures() -> None:
    frozen = registry()
    worker = BoundedWorker(frozen)
    valid = worker_request(frozen, "job_0123456789abcdef01234567")
    cases = (
        (
            valid.model_copy(update={"config_sha256": "b" * 64}),
            SyntheticRuntimeState(),
            "PROVENANCE_FAILURE",
        ),
        (
            valid.model_copy(update={"checkpoint_sha256": "c" * 64}),
            SyntheticRuntimeState(),
            "CHECKPOINT_HASH_MISMATCH",
        ),
        (valid, SyntheticRuntimeState(cuda_available=False), "CUDA_UNAVAILABLE"),
        (valid, SyntheticRuntimeState(simulated_failure="CUDA_OOM"), "CUDA_OOM"),
        (
            valid,
            SyntheticRuntimeState(simulated_failure="MODEL_LOAD_FAILURE"),
            "MODEL_LOAD_FAILURE",
        ),
        (valid, SyntheticRuntimeState(simulated_failure="INFERENCE_FAILURE"), "INFERENCE_FAILURE"),
    )
    for candidate, state, reason in cases:
        response = worker.validate_synthetic(candidate, state)
        assert response.reason_codes == (reason,)


def test_stage21g_registry_is_complete_immutable_and_publicly_path_free() -> None:
    frozen = registry()
    assert len(frozen) == 8
    assert all("path" not in key for item in frozen.public_registry() for key in item)
    with pytest.raises(TypeError):
        frozen._components[ComponentId.STAGE9] = frozen.resolve(ComponentId.STAGE9)  # type: ignore[index]


def test_stage21g_safety_propagation_cannot_upgrade_capabilities() -> None:
    assert (
        serving_decision(
            stage17_defer=True,
            stage19_statuses=["VERIFIED"],
            stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
        )[0]
        == "DEFER"
    )
    for verifier_status in ("PARTIALLY_VERIFIED", "WITHHELD_INSUFFICIENT_EVIDENCE"):
        assert (
            serving_decision(
                stage17_defer=False,
                stage19_statuses=[verifier_status],
                stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            )[0]
            == "DEFER"
        )
    ok, reasons = validate_orchestration(
        ["stage5", "stage9", "stage10_11", "stage16", "stage17", "stage18", "stage19", "stage20"],
        ["stage5", "stage9", "stage10_11", "stage16", "stage17", "stage18", "stage19", "stage20"],
        {
            "stage11_maximum_support": "PARTIALLY_SUPPORTED",
            "reliable_positive_localization": False,
            "localization_absence_contradiction": False,
            "stage13_selective_prediction_accepted": False,
            "ood_supported": False,
            "severity_supported": False,
            "temporal_change_supported": False,
            "device_localization_supported": False,
        },
    )
    assert ok and not reasons


def test_stage21g_request_scoped_cleanup_and_crash_recovery(tmp_path: Path) -> None:
    manager = TemporaryArtifactManager(tmp_path / "runtime")
    completed = manager.create("job_completed000000000000000")
    deferred = manager.create("job_deferred000000000000000")
    failed = manager.create("job_failed00000000000000000")
    assert len({completed, deferred, failed}) == 3
    manager.cleanup(completed.name)
    manager.cleanup(deferred.name)
    manager.cleanup(failed.name)
    manager.create("job_crash0000000000000000000")
    assert manager.crash_recovery_cleanup() == 1
    assert not any(manager.root.iterdir())


def test_stage21g_freeze_and_next_stage_authorizations() -> None:
    assert CONFIG["stage21b_contract_fingerprint"] == (
        "6d92a8ddab32c2669d9e41b0b3f98bcac7485188f3b0ead628a7c2f87ae15c5f"
    )
    assert CONFIG["next_canonical_stage"] == "21H_SYNTHETIC_RUNTIME_ACCEPTANCE_DECISION"
    assert not CONFIG["next_stage_authorizes_real_model_loading"]
    assert not CONFIG["next_stage_authorizes_bounded_real_inference"]
    assert not CONFIG["next_stage_authorizes_gpu_residency_profiling"]
    assert not CONFIG["next_stage_authorizes_language_model_work"]
