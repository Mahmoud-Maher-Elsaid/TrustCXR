from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trustcxr.serving.api import create_app
from trustcxr.serving.contract_validation import serving_decision, validate_orchestration
from trustcxr.serving.registry import FrozenComponentRegistry
from trustcxr.serving.runtime import (
    BoundedWorker,
    JobStore,
    SanitizedLogger,
    SyntheticRuntimeState,
    TemporaryArtifactManager,
    sanitized_disposition,
)
from trustcxr.serving.schemas import ComponentId, JobState, JobSubmission, WorkerRequest
from trustcxr.serving.state_machine import IllegalTransitionError


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def asgi_request(
    app: Any, method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """Exercise the real ASGI app without an HTTP client dependency or socket."""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    requests = [
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ]
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    headers = [(b"host", b"127.0.0.1")]
    if payload is not None:
        headers.append((b"content-type", b"application/json"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 41000),
        "server": ("127.0.0.1", 42121),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return int(start["status"]), json.loads(response_body or b"{}")


def worker_request(registry: FrozenComponentRegistry, job_id: str) -> WorkerRequest:
    component = registry.resolve(ComponentId.STAGE9)
    return WorkerRequest(
        job_id=job_id,
        component_id=component.component_id,
        input_token="synthetic-input",
        server_model_version=component.server_model_version,
        request_fingerprint="a" * 64,
        config_sha256=component.config_sha256,
        checkpoint_sha256=component.checkpoint_sha256,
    )


def _assert_sanitized(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in ("f:\\", "c:\\", "traceback", "stack_trace", "patient_id", "checkpoint_path"):
        if forbidden in serialized:
            raise AssertionError(f"Public response leaked prohibited content: {forbidden}")


def _case(
    category: str,
    name: str,
    operation: Callable[[], None] | Callable[[], Awaitable[None]],
    results: list[dict[str, str]],
) -> None:
    try:
        value = operation()
        if asyncio.iscoroutine(value):
            asyncio.run(value)
    except Exception as exc:
        results.append(
            {"category": category, "name": name, "status": "FAILED", "error": type(exc).__name__}
        )
    else:
        results.append({"category": category, "name": name, "status": "PASSED"})


def validate_runtime(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage21f_summary"]
    if sha256(summary_path) != config["stage21f_summary_sha256"]:
        raise RuntimeError("Stage 21F summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "stage21b_contract_fingerprint",
        "dependency_manifest_sha256",
        "dependency_fingerprint",
        "implementation_protocol_fingerprint",
    ):
        if summary[key] != config[key]:
            raise RuntimeError(f"Frozen Stage 21 evidence mismatch: {key}")

    store = JobStore()
    app = create_app(store)
    registry = FrozenComponentRegistry.from_stage21b(root)
    worker = BoundedWorker(registry)
    submission = JobSubmission(input_token="synthetic-input", idempotency_key="runtime-key")
    results: list[dict[str, str]] = []

    async def api_happy_path() -> None:
        health_code, health = await asgi_request(app, "GET", "/health")
        assert health_code == 200 and health["status"] == "READY_CONTRACT_ONLY"
        code, created = await asgi_request(
            app, "POST", "/v1/jobs", submission.model_dump(mode="json")
        )
        assert code == 202 and created["job_id"].startswith("job_")
        get_code, fetched = await asgi_request(app, "GET", f"/v1/jobs/{created['job_id']}")
        assert get_code == 200 and fetched == created
        again_code, again = await asgi_request(
            app, "POST", "/v1/jobs", submission.model_dump(mode="json")
        )
        assert again_code == 202 and again == created
        _assert_sanitized(created)

    async def api_invalid_cases() -> None:
        invalid_payloads = (
            {},
            {**submission.model_dump(mode="json"), "model_name": "arbitrary"},
            {**submission.model_dump(mode="json"), "checkpoint_path": "C:/private/model.pt"},
            {**submission.model_dump(mode="json"), "url": "https://example.invalid"},
            {**submission.model_dump(mode="json"), "free_text": "print('x')"},
            {**submission.model_dump(mode="json"), "patient_id": "patient-1"},
            {**submission.model_dump(mode="json"), "raw_phi": "name"},
        )
        for payload in invalid_payloads:
            code, response = await asgi_request(app, "POST", "/v1/jobs", payload)
            assert code == 422 and response["state"] == "FAILED_SANITIZED"
            _assert_sanitized(response)
        code, response = await asgi_request(app, "GET", "/v1/jobs/job_unknown")
        assert code == 404 and response["state"] == "FAILED_SANITIZED"
        _assert_sanitized(response)

    _case("API_RUNTIME", "health_submission_status_idempotency", api_happy_path, results)
    _case("API_RUNTIME", "invalid_and_sanitized_requests", api_invalid_cases, results)

    def idempotency_runtime() -> None:
        identical_store = JobStore()
        first = identical_store.submit(submission)
        second = identical_store.submit(submission)
        changed = identical_store.submit(
            submission.model_copy(update={"idempotency_key": "different-runtime-key"})
        )
        assert first == second and changed.job_id != first.job_id

    _case("IDEMPOTENCY", "identical_and_nonidentical_requests", idempotency_runtime, results)

    def state_machine_runtime() -> None:
        status = store.submit(JobSubmission(input_token="state-input", idempotency_key="state-key"))
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
        for terminal in (JobState.COMPLETED, JobState.DEFERRED, JobState.FAILED_SANITIZED):
            terminal_store = JobStore()
            item = terminal_store.submit(
                JobSubmission(input_token=f"terminal-{terminal.value}", idempotency_key="key")
            )
            terminal_store.advance(item.job_id, JobState.VALIDATING)
            if terminal == JobState.COMPLETED:
                for target in (
                    JobState.QUEUED,
                    JobState.GPU_PROCESSING,
                    JobState.CPU_POSTPROCESSING,
                    JobState.VERIFYING,
                    JobState.DECIDING,
                    JobState.COMPLETED,
                ):
                    terminal_store.advance(item.job_id, target)
            else:
                terminal_store.advance(item.job_id, terminal)
            try:
                terminal_store.advance(item.job_id, JobState.GPU_PROCESSING)
            except IllegalTransitionError:
                pass
            else:
                raise AssertionError("Terminal transition was accepted.")

    _case("STATE_MACHINE", "legal_and_terminal_transitions", state_machine_runtime, results)

    def worker_runtime() -> None:
        request = worker_request(registry, store.submit(submission).job_id)
        success = worker.validate_synthetic(request, SyntheticRuntimeState())
        assert getattr(success, "status", None) == "SUCCESS" and worker.resident_models == 0
        checks = (
            (
                request.model_copy(update={"config_sha256": "b" * 64}),
                SyntheticRuntimeState(),
                "PROVENANCE_FAILURE",
            ),
            (
                request.model_copy(update={"checkpoint_sha256": "c" * 64}),
                SyntheticRuntimeState(),
                "CHECKPOINT_HASH_MISMATCH",
            ),
            (request, SyntheticRuntimeState(cuda_available=False), "CUDA_UNAVAILABLE"),
            (request, SyntheticRuntimeState(simulated_failure="CUDA_OOM"), "CUDA_OOM"),
            (
                request,
                SyntheticRuntimeState(simulated_failure="MODEL_LOAD_FAILURE"),
                "MODEL_LOAD_FAILURE",
            ),
            (
                request,
                SyntheticRuntimeState(simulated_failure="INFERENCE_FAILURE"),
                "INFERENCE_FAILURE",
            ),
            (
                request,
                SyntheticRuntimeState(simulated_failure="VERIFIER_FAILURE"),
                "VERIFIER_FAILURE",
            ),
            (
                request,
                SyntheticRuntimeState(simulated_failure="DECISION_POLICY_FAILURE"),
                "DECISION_POLICY_FAILURE",
            ),
        )
        for candidate, runtime_state, reason in checks:
            response = worker.validate_synthetic(candidate, runtime_state)
            assert response.reason_codes == (reason,)
            _assert_sanitized(response.model_dump(mode="json"))
        invalid = request.model_dump(mode="json")
        invalid["component_id"] = "unknown_component"
        try:
            WorkerRequest.model_validate(invalid)
        except ValidationError:
            pass
        else:
            raise AssertionError("Unknown worker component was accepted.")

    _case("WORKER_RUNTIME", "synthetic_success_and_failure_control_flow", worker_runtime, results)

    def registry_runtime() -> None:
        assert len(registry) == 8
        public = registry.public_registry()
        assert all("path" not in key for item in public for key in item)
        try:
            registry._components[ComponentId.STAGE9] = registry.resolve(ComponentId.STAGE9)  # type: ignore[index]
        except TypeError:
            pass
        else:
            raise AssertionError("Frozen registry mutation succeeded.")

    _case("REGISTRY", "immutable_eight_component_registry", registry_runtime, results)

    def safety_runtime() -> None:
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
        ok, reasons = validate_orchestration(
            [
                "stage5",
                "stage9",
                "stage10_11",
                "stage16",
                "stage17",
                "stage18",
                "stage19",
                "stage20",
            ],
            [
                "stage5",
                "stage9",
                "stage10_11",
                "stage16",
                "stage17",
                "stage18",
                "stage19",
                "stage20",
            ],
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

    _case("SAFETY_PROPAGATION", "frozen_capabilities_and_defer_precedence", safety_runtime, results)

    runtime_root = root / config["runtime_root"]
    manager = TemporaryArtifactManager(runtime_root)

    def cleanup_runtime() -> None:
        manager.crash_recovery_cleanup()
        job_ids = (
            "job_completed000000000000000",
            "job_deferred000000000000000",
            "job_failed00000000000000000",
        )
        paths = [manager.create(job_id) for job_id in job_ids]
        assert len({path.resolve() for path in paths}) == 3
        for job_id in job_ids:
            manager.cleanup(job_id)
        manager.create("job_crash0000000000000000000")
        assert manager.crash_recovery_cleanup() == 1
        assert not any(runtime_root.iterdir())
        cleanup_failure = sanitized_disposition("job_cleanup00000000000000000", "CLEANUP_FAILURE")
        assert cleanup_failure.state == JobState.FAILED_SANITIZED

    _case("CLEANUP", "terminal_and_crash_cleanup", cleanup_runtime, results)

    def privacy_runtime() -> None:
        stream = io.StringIO()
        logger = logging.getLogger("trustcxr.stage21g.synthetic")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        SanitizedLogger(logger).state(
            "job_0123456789abcdef01234567", JobState.DEFERRED, ("UNSUPPORTED_INPUT",)
        )
        log_text = stream.getvalue().lower()
        assert "patient" not in log_text and "checkpoint" not in log_text and "f:\\" not in log_text

    _case("PRIVACY", "sanitized_runtime_logging", privacy_runtime, results)

    failed = [item for item in results if item["status"] != "PASSED"]
    if runtime_root.exists() and any(runtime_root.iterdir()):
        manager.crash_recovery_cleanup()
        failed.append({"category": "CLEANUP", "name": "final_cleanup", "status": "FAILED"})
    if failed:
        raise RuntimeError(f"Stage 21G synthetic runtime validation failed: {failed}")

    counts: dict[str, dict[str, int]] = {}
    for category in sorted({item["category"] for item in results}):
        category_items = [item for item in results if item["category"] == category]
        counts[category] = {
            "passed": sum(item["status"] == "PASSED" for item in category_items),
            "failed": sum(item["status"] != "PASSED" for item in category_items),
        }
    return {
        "stage": "21G",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21f_summary_sha256": config["stage21f_summary_sha256"],
        "runtime_case_counts": counts,
        "runtime_cases_passed": len(results),
        "runtime_cases_failed": 0,
        "server_process_started": False,
        "server_process_cleanup_status": "NOT_APPLICABLE_IN_PROCESS_ASGI_ONLY",
        "persistent_worker_started": False,
        "temporary_artifacts_cleanup_status": "COMPLETE",
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_real_model_loading": False,
        "next_stage_authorizes_bounded_real_inference": False,
        "next_stage_authorizes_gpu_residency_profiling": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate_runtime(config, root)
    report = root / "reports/stage21/stage21g_bounded_synthetic_runtime_validation_summary.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
