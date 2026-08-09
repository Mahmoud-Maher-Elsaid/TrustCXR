from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21e_serving_dependency_implementation_protocol_freeze import freeze

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21e_serving_dependency_implementation_protocol_freeze.json"


def result() -> dict:
    return freeze(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)


def test_stage21e_freezes_minimal_exact_hashed_dependencies() -> None:
    evidence = result()
    assert evidence["approved_direct_dependencies"] == {
        "fastapi": "0.116.1",
        "pydantic": "2.11.7",
        "uvicorn": "0.35.0",
    }
    assert evidence["dependency_manifest_sha256"] == (
        "6b82df6db246280518aec3078f55675e02c41b9afd4e74c104836960b820d782"
    )
    assert len(evidence["dependency_fingerprint"]) == 64
    assert not evidence["packages_installed"]


def test_stage21e_freezes_smallest_sequential_local_architecture() -> None:
    architecture = result()["frozen_architecture"]
    assert architecture["api_processes"] == 1
    assert architecture["gpu_worker_execution_paths"] == 1
    assert not architecture["persistent_distributed_worker"]
    assert architecture["gpu_model_residency"] == (
        "ONE_GPU_MODEL_AT_A_TIME_UNTIL_MEASURED_RESIDENCY_AUDIT"
    )


def test_stage21e_preserves_loading_order_and_failure_semantics() -> None:
    evidence = result()
    assert evidence["model_loading_protocol"][0] == "VALIDATE_REQUEST"
    assert "VERIFY_CHECKPOINT_SHA256" in evidence["model_loading_protocol"]
    assert "RELEASE_GPU_MODEL_BEFORE_NEXT_GPU_MODEL" in evidence["model_loading_protocol"]
    assert evidence["failure_semantics"]["UNSUPPORTED_INPUT"] == "DEFER"
    assert evidence["failure_semantics"]["CUDA_OOM"] == "FAILED_SANITIZED"


def test_stage21e_authorizes_only_next_minimal_implementation_stage() -> None:
    evidence = result()
    assert not evidence["backend_worker_implemented"]
    assert not evidence["server_started"]
    assert not evidence["gpu_residency_profiled"]
    assert evidence["next_stage_authorizes_backend_worker_implementation"]
    assert not evidence["next_stage_authorizes_gpu_residency_profiling"]
    assert not evidence["next_stage_authorizes_language_model_work"]
    assert evidence["currently_planned_llm_authorized_gate"] is None
