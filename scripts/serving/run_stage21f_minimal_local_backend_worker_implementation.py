from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from trustcxr.serving.api import create_app
from trustcxr.serving.registry import FrozenComponentRegistry
from trustcxr.serving.runtime import (
    BoundedWorker,
    JobStore,
    SyntheticRuntimeState,
    TemporaryArtifactManager,
)
from trustcxr.serving.schemas import ComponentId, JobSubmission, SanitizedDisposition, WorkerRequest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_implementation(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage21e_path = root / config["stage21e_summary"]
    if sha256(stage21e_path) != config["stage21e_summary_sha256"]:
        raise RuntimeError("Stage 21E summary SHA-256 mismatch.")
    stage21e = json.loads(stage21e_path.read_text(encoding="utf-8"))
    for key in (
        "stage21b_contract_fingerprint",
        "dependency_manifest_sha256",
        "dependency_fingerprint",
        "implementation_protocol_fingerprint",
    ):
        if stage21e[key] != config[key]:
            raise RuntimeError(f"Stage 21E evidence mismatch: {key}")
    installed = {
        package: importlib.metadata.version(package) for package in config["required_versions"]
    }
    if installed != config["required_versions"]:
        raise RuntimeError("Installed serving dependency versions do not match the frozen lock.")

    registry = FrozenComponentRegistry.from_stage21b(root)
    if [item["component_id"] for item in registry.public_registry()] != sorted(
        config["frozen_components"]
    ):
        raise RuntimeError("Frozen component registry mismatch.")
    if any("path" in key for item in registry.public_registry() for key in item):
        raise RuntimeError("Public component registry exposed an internal path.")
    app = create_app()
    routes = sorted(
        f"{next(iter(route.methods))} {route.path}"
        for route in app.routes
        if getattr(route, "methods", None)
    )
    if routes != sorted(config["public_endpoints"]):
        raise RuntimeError(f"Unexpected public API endpoints: {routes}")

    submission = JobSubmission(input_token="synthetic-input", idempotency_key="synthetic-key")
    store = JobStore()
    first = store.submit(submission)
    second = store.submit(submission)
    if first != second or not first.job_id.startswith("job_"):
        raise RuntimeError("Deterministic pseudonymous idempotency failed.")
    component = registry.resolve(ComponentId.STAGE9)
    request = WorkerRequest(
        job_id=first.job_id,
        component_id=component.component_id,
        input_token="synthetic-input",
        server_model_version=component.server_model_version,
        request_fingerprint="a" * 64,
        config_sha256=component.config_sha256,
        checkpoint_sha256=component.checkpoint_sha256,
    )
    worker = BoundedWorker(registry)
    response = worker.validate_synthetic(request, SyntheticRuntimeState(cuda_available=True))
    if isinstance(response, SanitizedDisposition) or worker.resident_models != 0:
        raise RuntimeError("Synthetic worker contract validation failed.")

    runtime_root = root / config["runtime_root"]
    manager = TemporaryArtifactManager(runtime_root)
    manager.crash_recovery_cleanup()
    manager.create(first.job_id)
    manager.cleanup(first.job_id)
    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise RuntimeError("Synthetic request cleanup failed.")

    for key in (
        "persistent_server_permitted",
        "persistent_worker_permitted",
        "real_model_loading_permitted",
        "real_model_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "real_patient_report_generation_permitted",
        "locked_test_access_permitted",
        "training_permitted",
        "fine_tuning_permitted",
        "checkpoint_mutation_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 21F prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is authorized.")

    return {
        "stage": "21F",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "dependency_manifest_sha256": config["dependency_manifest_sha256"],
        "dependency_fingerprint": config["dependency_fingerprint"],
        "implementation_protocol_fingerprint": config["implementation_protocol_fingerprint"],
        "installed_serving_dependencies": installed,
        "implemented_components": [
            "PYDANTIC_CONTRACT_SCHEMAS",
            "MINIMAL_FASTAPI_APPLICATION",
            "DETERMINISTIC_JOB_STATE_MACHINE",
            "IMMUTABLE_FROZEN_COMPONENT_REGISTRY",
            "BOUNDED_SYNTHETIC_WORKER_INTERFACE",
            "REQUEST_SCOPED_TEMPORARY_ARTIFACT_MANAGER",
            "SANITIZED_FAILURE_MAPPING",
            "DETERMINISTIC_IDEMPOTENCY",
        ],
        "public_endpoints": config["public_endpoints"],
        "frozen_component_count": len(registry),
        "server_started": False,
        "persistent_worker_started": False,
        "real_model_loaded": False,
        "real_model_inference_performed": False,
        "gpu_residency_profiled": False,
        "real_patient_records_used": 0,
        "real_patient_reports_generated": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_runtime_api_worker_validation": True,
        "next_stage_authorizes_real_model_inference": False,
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
    result = validate_implementation(config, root)
    report_dir = root / "reports/stage21"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage21f_minimal_local_backend_worker_implementation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
