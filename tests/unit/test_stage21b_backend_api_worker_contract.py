from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21b_backend_api_worker_contract import freeze_contract

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21b_backend_api_worker_contract.json"


def result() -> dict:
    return freeze_contract(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)


def test_stage21b_state_machine_and_worker_are_fail_closed() -> None:
    contract = result()
    transitions = contract["job_state_machine"]["transitions"]
    assert transitions["DEFERRED"] == []
    assert transitions["FAILED_SANITIZED"] == []
    assert contract["job_state_machine"]["deferred_is_successful_safety_disposition"]
    assert not contract["job_state_machine"]["partial_success_after_safety_failure_permitted"]
    assert contract["worker_contract"]["maximum_resident_gpu_models"] == 1
    assert not contract["worker_contract"]["checkpoint_mutation_permitted"]


def test_stage21b_schemas_exclude_sensitive_and_arbitrary_inputs() -> None:
    contract = result()
    submission = contract["schemas"]["job_submission"]
    assert "patient_id" in submission["forbidden"]
    assert "path" in submission["forbidden"]
    assert "model_name" in submission["forbidden"]
    assert "job_id" in submission["server_generated"]
    assert "python_code" in contract["worker_contract"]["forbidden_inputs"]


def test_stage21b_does_not_authorize_implementation_or_llm() -> None:
    contract = result()
    assert contract["synthetic_contract_fixtures_prepared"] == 12
    assert not contract["server_started"]
    assert not contract["worker_started"]
    assert not contract["model_inference_performed"]
    assert contract["locked_test_records_accessed"] == 0
    assert not contract["language_model_used"]
    assert not contract["language_model_endpoint_prepared"]
    assert contract["currently_planned_llm_authorized_gate"] is None
    assert not contract["next_stage_authorizes_backend_worker_implementation"]
    assert not contract["next_stage_authorizes_language_model_work"]


def test_stage21b_launcher_resolves_project_venv_without_obsolete_path() -> None:
    launcher = (ROOT / "scripts/serving/run_stage21b_backend_api_worker_contract.ps1").read_text(
        encoding="utf-8"
    )
    assert 'Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"' in launcher
    assert "Resolve-Path -LiteralPath $PythonCandidate" in launcher
    assert "Project virtual-environment interpreter exists but is not runnable" in launcher
    assert "C:\\Users\\maher" not in launcher
    assert '"$Python"' not in launcher
