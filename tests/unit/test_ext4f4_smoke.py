from __future__ import annotations

import ast
from pathlib import Path

from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import build_ext4f_semantic_plan
from trustcxr.grounded_llm.ext4f_realization import (
    LLM_REALIZATION_ONLY_FIELDS,
    build_ext4f_realization_request,
    realization_schema_sha256,
    validate_ext4f_realization_request,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/training/run_ext4f4_synthetic_smoke.py"


def _request():
    evidence = build_synthetic_case("uncertainty").model_copy(
        update={"case_reference": "research_case_ext4f4_realization_001"}
    )
    plan = build_ext4f_semantic_plan(evidence)
    request = build_ext4f_realization_request(plan)
    validate_ext4f_realization_request(request)
    return plan, request


def test_ext4f4_uses_new_case_and_planner_authority():
    plan, request = _request()
    assert plan.source_case_reference == "research_case_ext4f4_realization_001"
    assert request.semantic_plan_sha256 == plan.semantic_plan_sha256


def test_ext4f4_schema_and_authority_are_frozen():
    assert realization_schema_sha256() == (
        "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
    )
    assert set(LLM_REALIZATION_ONLY_FIELDS) == {"slot_text"}


def test_ext4f4_runner_has_one_generate_call_and_no_retry_path():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    generate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]
    assert len(generate_calls) == 1
    assert "retry" not in RUNNER.read_text(encoding="utf-8").lower()


def test_ext4f4_runner_has_partition_tripwires_and_no_benchmark_ids():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"development_cases_accessed": 0' in source
    assert '"frozen_final_cases_accessed": 0' in source
    assert '"locked_test_accessed": False' in source
    for case_id in (
        "dev_supported",
        "dev_uncertainty",
        "dev_defer",
        "dev_withheld",
        "dev_missing",
        "dev_conflict",
    ):
        assert case_id not in source
