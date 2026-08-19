# ruff: noqa: E501
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.ext4f5_benchmark import (
    CASE_IDS,
    GENERATION_POLICY,
    build_development_cases,
    final_development_decision,
    import_review_results,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_benchmark_and_policy():
    manifest = json.loads(
        (ROOT / "configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json").read_text()
    )
    assert (
        manifest["benchmark_sha256"]
        == "671a04d2d859f1b1ffb9414a8c0f636596949748a00548e45abcbbfdb752db61"
    )
    assert tuple(manifest["ordered_case_ids"]) == CASE_IDS
    assert len(build_development_cases()) == 24
    assert GENERATION_POLICY["retry_count"] == 0
    assert GENERATION_POLICY["do_sample"] is False


def test_runner_has_single_generation_call_and_no_final_partition():
    path = ROOT / "scripts/training/run_ext4f6_development.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]
    assert len(calls) == 1
    source = path.read_text(encoding="utf-8")
    assert "EXT-4D" not in source
    assert 'final_cases_accessed"] = 0' in source
    assert 'locked_test_accessed"] = False' in source


def test_review_import_rejects_unknown_or_incomplete_results():
    cases = build_development_cases()
    with pytest.raises(ValueError, match="BENCHMARK_MISMATCH"):
        import_review_results("expected", cases, {"benchmark_sha256": "wrong", "cases": {}})
    with pytest.raises(ValueError, match="CASE_SET_INCOMPLETE"):
        import_review_results("x", cases, {"benchmark_sha256": "x", "cases": {}})


def test_final_decision_uses_frozen_thresholds():
    cases = build_development_cases()
    review_cases = {}
    for case in cases:
        review_cases[case.case_id] = {
            "case_sha256": case.case_sha256,
            "slots": {
                slot["slot_id"]: {dim: "PASS" for dim in slot["faithfulness_dimensions"]}
                for slot in case.expectations
            },
        }
    imported = import_review_results("b", cases, {"benchmark_sha256": "b", "cases": review_cases})
    automatic = {
        "protocol_deviation_count": 0,
        "structured_output_validity_rate": 1.0,
        "realization_contract_validity_rate": 1.0,
        "authority_preservation_rate": 1.0,
        "hard_safety_gate_pass": True,
    }
    assert "PASSED" in final_development_decision(automatic, imported)["decision"]
