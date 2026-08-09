from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21d_backend_worker_implementation_readiness_decision import decide

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21d_backend_worker_implementation_readiness_decision.json"


def result() -> dict:
    return decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)


def test_stage21d_preserves_evidence_and_dependency_governance_hold() -> None:
    decision = result()
    assert decision["stage21b_contract_fingerprint"] == (
        "6d92a8ddab32c2669d9e41b0b3f98bcac7485188f3b0ead628a7c2f87ae15c5f"
    )
    assert decision["stage21b_summary_sha256"] == (
        "649a046c31499bb1a280bff09793045683cb4f0fb760cfb72b56672c1c7d3d82"
    )
    assert decision["stage21c_fixtures_passed"] == 83
    assert decision["dependency_decision"]["fastapi"] == "UNPINNED_NOT_YET_APPROVED"
    assert not decision["dependency_decision"]["package_installation_permitted"]


def test_stage21d_preserves_single_model_residency_rule() -> None:
    residency = result()["gpu_residency_decision"]
    assert residency["current_rule"] == "ONE_GPU_MODEL_AT_A_TIME_UNTIL_MEASURED_RESIDENCY_AUDIT"
    assert not residency["required_before_sequential_one_model_implementation"]
    assert residency["required_before_multi_model_residency_authorization"]
    assert not residency["profile_permitted_in_stage21d"]


def test_stage21d_authorizes_no_execution_implementation_or_llm() -> None:
    decision = result()
    assert not decision["backend_worker_implementation_authorized"]
    assert not decision["gpu_residency_profiling_authorized"]
    assert not decision["server_started"]
    assert not decision["worker_started"]
    assert not decision["model_loaded"]
    assert decision["locked_test_records_accessed"] == 0
    assert not decision["language_model_used"]
    assert decision["currently_planned_llm_authorized_gate"] is None


def test_stage21d_preserves_defer_and_failure_distinction() -> None:
    semantics = result()["failure_semantics"]
    assert semantics["safety_or_evidence_limitations"] == "DEFER"
    assert semantics["technical_infrastructure_failures"] == "FAILED_SANITIZED"
    assert not semantics["technical_failure_as_model_or_clinical_decision_permitted"]
