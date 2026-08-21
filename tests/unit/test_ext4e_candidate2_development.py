"""Offline orchestration tests for Candidate #2 development evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/training/run_ext4e_candidate2_development.py"
    spec = importlib.util.spec_from_file_location("candidate2_development_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bootstrap_module():
    path = ROOT / "scripts/training/bootstrap_ext4e_candidate2.py"
    spec = importlib.util.spec_from_file_location("candidate2_bootstrap_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response() -> bytes:
    return json.dumps(
        {
            "schema_id": "EXT4_OUTPUT_CONTRACT",
            "schema_version": "1",
            "case_reference": "research_case_synthetic_supported",
            "generation_status": "ABSTAINED",
            "research_summary": None,
            "summary_claim_ids": [],
            "evidence_references": [
                {
                    "evidence_id": "stage9_signal_01",
                    "status": "SUPPORTED",
                    "provenance_refs": ["stage9_signal_01"],
                }
            ],
            "claims": [],
            "uncertainty_summary": {"status": "NOT_AVAILABLE"},
            "limitations": [],
            "defer_state": {
                "decision": "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
                "defer_active": False,
                "reason_codes": [],
                "non_overridable": True,
            },
            "contradictions": [],
            "provenance_refs": ["stage9_signal_01"],
            "reviewer_flags": [],
            "output_scope": "RESEARCH_EXTENSION_ONLY",
        }
    ).encode()


def test_candidate2_offline_evaluation_runs_exact_six_cases(tmp_path):
    module = _module()
    requests: list[dict] = []

    def request(_url, payload):
        requests.append(payload)
        content = _response().decode()
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    run_root = tmp_path / "evaluation"
    result = module.run_development_evaluation(
        server_url="http://127.0.0.1:18080",
        model="candidate2.gguf",
        schema={"type": "object"},
        run_root=run_root,
        request_fn=request,
        prompt="frozen prompt",
    )
    assert len(requests) == 6
    assert result["total_cases"] == 6
    assert result["development_requests_attempted"] == 6
    assert result["final_cases_accessed"] == 0
    assert result["locked_test_accessed"] is False
    for payload in requests:
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["reasoning_format"] == "none"
        assert "reasoning_effort" not in payload
        assert payload["response_format"]["type"] == "json_object"
    ledger = json.loads((run_root / "case_attempt_ledger.json").read_text())
    assert tuple(ledger["case_order"]) == module.CASE_IDS
    assert all(entry["request_count"] == 1 for entry in ledger["cases"].values())


def test_candidate2_case_selection_rejects_final_or_unknown_ids(tmp_path):
    module = _module()
    cases = json.loads(module.CASES_PATH.read_text())
    cases["development_cases"][0]["case_id"] = "final_supported"
    altered = tmp_path / "cases.json"
    altered.write_text(json.dumps(cases))
    try:
        module.resolve_development_cases(altered)
    except RuntimeError as exc:
        assert str(exc) == "CANDIDATE2_DEVELOPMENT_CASE_SET_INVALID"
    else:
        raise AssertionError("unsafe case set was accepted")


def test_top_level_development_handoff_reaches_real_evaluator(tmp_path):
    bootstrap = _bootstrap_module()
    requests: list[dict] = []

    def request(_url, payload):
        requests.append(payload)
        return json.dumps({"choices": [{"message": {"content": _response().decode()}}]}).encode()

    result = bootstrap.run_candidate2_development_evaluation(
        server_url="http://127.0.0.1:18080",
        model="candidate2.gguf",
        schema={"type": "object"},
        run_root=tmp_path / "top-level-evaluation",
        request_fn=request,
    )
    assert result["total_cases"] == 6
    assert len(requests) == 6
    assert result["final_cases_accessed"] == 0
    assert result["locked_test_accessed"] is False


def test_scientific_decision_labels_are_binary_and_do_not_open_partitions():
    module = _module()
    assert module.scientific_decision(True) == ("DEVELOPMENT_GATE_PASSED / SCIENTIFICALLY_SELECTED")
    assert module.scientific_decision(False) == (
        "DEVELOPMENT_GATE_FAILED / NOT_SCIENTIFICALLY_SELECTED"
    )
