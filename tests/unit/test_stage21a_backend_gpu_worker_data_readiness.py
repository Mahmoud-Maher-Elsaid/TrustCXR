from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21a_backend_gpu_worker_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21a_backend_gpu_worker_data_readiness.json"


def result() -> dict:
    return audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)


def test_stage21a_inventory_and_safety_propagation_are_complete() -> None:
    readiness = result()
    assert len(readiness["eligible_components"]) == 8
    assert readiness["gpu_worker_readiness"]["residency_policy"].startswith("ONE_GPU_MODEL")
    assert not readiness["gpu_worker_readiness"]["checkpoint_mutation_permitted"]
    assert "STAGE20_DEFER_PRECEDENCE" in readiness["safety_propagation_required"]
    assert readiness["implementation_hold_reasons"]


def test_stage21a_performs_no_runtime_or_llm_work() -> None:
    readiness = result()
    assert not readiness["language_model_used"]
    assert not readiness["language_model_endpoint_prepared"]
    assert not readiness["model_inference_performed"]
    assert not readiness["real_patient_pipeline_activated"]
    assert not readiness["production_server_started"]
    assert not readiness["persistent_worker_started"]
    assert readiness["locked_test_records_accessed"] == 0
    assert readiness["patient_identifiers_used"] == 0
    assert readiness["currently_planned_llm_authorized_gate"] is None
    assert not readiness["next_stage_authorizes_language_model_work"]
