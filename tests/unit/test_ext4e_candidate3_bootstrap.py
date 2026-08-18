"""Offline Candidate #3 identity/bootstrap contract tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/training/bootstrap_ext4e_candidate3.py"
    spec = importlib.util.spec_from_file_location("candidate3_bootstrap_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate3_identity_is_official_and_immutable():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json").read_text()
    )
    assert config["repository"] == "microsoft/Phi-4-mini-instruct"
    assert config["resolved_revision"] == "cfbefacb99257ffa30c83adab238a50856ac3083"
    assert config["model_family"] == "Phi-4-mini"
    assert config["format"] == "safetensors"
    assert config["license"] == "MIT"
    assert all(item["sha256"] for item in config["artifacts"][:2])
    assert config["artifacts"][0]["size_bytes"] == 4903637712
    assert config["artifacts"][1]["size_bytes"] == 2768428504
    assert config["download"]["required"] is False
    assert {item["filename"] for item in config["artifacts"]} >= {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "configuration_phi3.py",
        "modeling_phi3.py",
    }


def test_candidate3_preflight_is_partition_safe_and_generation_free():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json").read_text()
    )
    assert config["partitions"]["development_cases_accessed"] == 0
    assert config["partitions"]["final_cases_accessed"] == 0
    assert config["partitions"]["locked_test_accessed"] is False
    assert config["generation"]["request_count"] == 1
    assert config["generation"]["retry_count"] == 0


def test_candidate3_request_builder_preserves_schema_without_reasoning_fallback():
    from trustcxr.grounded_llm.candidate3_request import build_candidate3_request_payload

    schema = {"type": "object", "additionalProperties": False}
    payload = build_candidate3_request_payload("phi4", [{"role": "user", "content": "x"}], schema)
    assert payload["response_format"] == {"type": "json_object", "schema": schema}
    assert "reasoning_effort" not in payload
    assert "grammar" not in payload
    assert payload["temperature"] == 0.0
    assert "retry_count" not in payload


def test_candidate3_structured_output_fails_closed_without_constrained_decoder():
    module = _module()
    result = module.structured_output_preflight()
    assert result["status"] == "CANDIDATE3_STRUCTURED_OUTPUT_MECHANISM_NOT_YET_ESTABLISHED"
    assert result["available_constrained_decoding"] == []


def test_candidate3_exact_schema_identity_is_stable():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        GOVERNED_SCHEMA_SHA256,
        governed_schema,
        schema_sha256,
    )

    assert schema_sha256(governed_schema()) == GOVERNED_SCHEMA_SHA256
    assert (
        GOVERNED_SCHEMA_SHA256 == "7e28f42cc574cf40d45a725ffac526fc469ac834ab86a574ac613ae79923c650"
    )


def test_candidate3_xgrammar_adapter_compiles_exact_schema_and_builds_processor():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        build_candidate3_logits_processor,
        governed_schema,
    )

    calls = {}

    class FakeTokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size):
            calls["tokenizer"] = (tokenizer, vocab_size)
            return "tokenizer-info"

    class FakeCompiler:
        def __init__(self, info):
            calls["info"] = info

        def compile_json_schema(self, schema):
            calls["schema"] = json.loads(schema)
            return "compiled"

    fake = SimpleNamespace(
        __version__="0.2.5",
        TokenizerInfo=FakeTokenizerInfo,
        GrammarCompiler=FakeCompiler,
        contrib=SimpleNamespace(hf=SimpleNamespace(LogitsProcessor=lambda value: ("lp", value))),
    )
    result = build_candidate3_logits_processor("tokenizer", vocab_size=200064, xgrammar_module=fake)
    assert result.backend == "xgrammar"
    assert result.backend_version == "0.2.5"
    assert result.logits_processor == ("lp", "compiled")
    assert calls["schema"] == governed_schema()


def test_candidate3_xgrammar_adapter_fails_closed_on_missing_backend():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3StructuredOutputError,
        build_candidate3_logits_processor,
    )

    try:
        build_candidate3_logits_processor("tokenizer", vocab_size=200064, xgrammar_module=None)
    except Candidate3StructuredOutputError as exc:
        assert str(exc) == "CANDIDATE3_STRUCTURED_OUTPUT_BACKEND_UNAVAILABLE"
    else:
        raise AssertionError("missing XGrammar must fail closed")
