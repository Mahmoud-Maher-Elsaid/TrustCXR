from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path

import pytest

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


def test_ext4f4_real_tokenizer_preflight_and_200064_probe():
    if importlib.metadata.version("llguidance") != "1.8.0":
        pytest.skip("governed llguidance is unavailable in this interpreter")
    import torch
    from transformers import AutoTokenizer

    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        assert_generation_constraint,
        build_llguidance_logits_processor,
    )
    from trustcxr.grounded_llm.ext4f_realization import realization_schema

    model_root = ROOT / "cache/research_extensions/ext4e_candidate3/models"
    if not model_root.is_dir():
        pytest.skip("local tokenizer is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    schema = realization_schema()
    schema_sha = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
    constraint = build_llguidance_logits_processor(
        tokenizer,
        schema=schema,
        expected_schema_sha256=schema_sha,
        prompt_length=0,
        model_vocab_size=200064,
    )
    assert_generation_constraint(constraint, expected_schema_sha256=schema_sha)
    assert constraint.vocab_alignment["initial_mask_available"] is True
    scores = constraint.logits_processor(
        torch.empty((1, 0), dtype=torch.long), torch.zeros((1, 200064))
    )
    assert tuple(scores.shape) == (1, 200064)
    assert torch.isneginf(scores[0, 200029:]).all()


def test_ext4f4_ext4c_schema_cannot_authorize_realization_gate():
    source = (ROOT / "src/trustcxr/grounded_llm/candidate3_constrained_decoding.py").read_text(
        encoding="utf-8"
    )
    assert "expected_schema_sha256" in source
    assert "GOVERNED_SCHEMA_SHA256" in source
    runner = RUNNER.read_text(encoding="utf-8")
    assert "EXPECTED_REALIZATION_SCHEMA_SHA" in runner
    assert "assert_generation_constraint(" in runner


def test_pending_or_wrong_schema_cannot_authorize_generation():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3Constraint,
        Candidate3StructuredOutputError,
        assert_generation_constraint,
    )

    constraint = Candidate3Constraint(
        logits_processor=object(),
        matcher=object(),
        backend="llguidance",
        backend_version="1.8.0",
        schema_sha256="99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1",
        vocab_alignment={"initial_mask_available": True},
    )
    with pytest.raises(Candidate3StructuredOutputError, match="STRUCTURED_OUTPUT_GATE_FAILED"):
        assert_generation_constraint(constraint)
