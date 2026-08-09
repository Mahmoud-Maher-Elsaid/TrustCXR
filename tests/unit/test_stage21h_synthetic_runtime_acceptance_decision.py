from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.serving.run_stage21h_synthetic_runtime_acceptance_decision import (
    REQUIRED_PRIVACY_RULES,
    REQUIRED_SAFETY_LIMITATIONS,
    decide_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/serving/stage21h_synthetic_runtime_acceptance_decision.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage21h_acceptance_evidence_is_exact() -> None:
    cfg = config()
    result = decide_acceptance(cfg, ROOT)
    assert result["status"] == (
        "ACCEPTED_MINIMAL_LOCAL_RESEARCH_SERVING_ARCHITECTURE_SYNTHETICALLY_VALIDATED"
    )
    assert result["stage21b_contract_fingerprint"] == (
        "6d92a8ddab32c2669d9e41b0b3f98bcac7485188f3b0ead628a7c2f87ae15c5f"
    )
    assert result["stage21f_summary_sha256"] == (
        "02a07278b6ccbdf773749733b448cf80129de93a40bedbf6d867c400f29219c3"
    )


def test_stage21h_preserves_architecture_and_public_endpoints() -> None:
    result = decide_acceptance(config(), ROOT)
    assert result["architecture"] == config()["architecture"]
    assert result["public_endpoints"] == [
        "POST /v1/jobs",
        "GET /v1/jobs/{job_id}",
        "GET /health",
    ]


def test_stage21h_preserves_all_safety_and_privacy_rules() -> None:
    result = decide_acceptance(config(), ROOT)
    assert tuple(result["safety_limitations"]) == REQUIRED_SAFETY_LIMITATIONS
    assert tuple(result["privacy_rules"]) == REQUIRED_PRIVACY_RULES
    assert result["failure_semantics"] == {
        "safety_or_evidence_limitation": "DEFER",
        "technical_infrastructure_failure": "FAILED_SANITIZED",
    }
    assert result["no_partial_success_after_safety_critical_failure"]


def test_stage21h_rejects_failed_or_incomplete_runtime_evidence(tmp_path: Path) -> None:
    source = json.loads((ROOT / config()["stage21g_summary"]).read_text(encoding="utf-8"))
    source["runtime_cases_failed"] = 1
    altered = tmp_path / "stage21g.json"
    altered.write_text(json.dumps(source), encoding="utf-8")
    cfg = config()
    cfg["stage21g_summary"] = str(altered.relative_to(tmp_path))
    cfg["stage21g_summary_sha256"] = __import__("hashlib").sha256(altered.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="evidence mismatch"):
        decide_acceptance(cfg, tmp_path)


@pytest.mark.parametrize(
    "key",
    [
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "persistent_server_permitted",
        "persistent_worker_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage21h_rejects_runtime_scope_expansion(key: str) -> None:
    cfg = config()
    cfg[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        decide_acceptance(cfg, ROOT)


def test_stage21h_next_stage_has_no_runtime_or_llm_authorization() -> None:
    result = decide_acceptance(config(), ROOT)
    assert result["next_canonical_stage"] == "22A_RESEARCH_UI_MEDICAL_VIEWER_DATA_READINESS"
    assert not result["next_stage_authorizes_real_model_loading"]
    assert not result["next_stage_authorizes_bounded_real_inference"]
    assert not result["next_stage_authorizes_gpu_residency_profiling"]
    assert not result["next_stage_authorizes_real_patient_processing"]
    assert not result["next_stage_authorizes_language_model_work"]
    assert result["currently_planned_llm_authorized_gate"] is None


def test_stage21h_does_not_write_output_during_validation() -> None:
    output = ROOT / "reports/stage21/stage21h_synthetic_runtime_acceptance_decision_summary.json"
    before = output.read_bytes() if output.exists() else None
    decide_acceptance(config(), ROOT)
    after = output.read_bytes() if output.exists() else None
    assert after == before
