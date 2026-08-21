"""Offline Candidate #3 identity/bootstrap contract tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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


def test_candidate3_structured_output_backend_is_llguidance_only():
    module = _module()
    result = module.structured_output_preflight()
    assert result["status"] in {
        "CANDIDATE3_STRUCTURED_OUTPUT_MECHANISM_REQUIRES_ADAPTER_PROOF",
        "CANDIDATE3_STRUCTURED_OUTPUT_MECHANISM_NOT_YET_ESTABLISHED",
    }
    assert result["available_constrained_decoding"] in ([], ["llguidance"])


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


def test_candidate3_llguidance_adapter_is_fail_closed_without_backend():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3StructuredOutputError,
        build_candidate3_prefix_allowed_tokens_fn,
    )

    try:
        build_candidate3_prefix_allowed_tokens_fn("tokenizer", prompt_length=0)
    except Candidate3StructuredOutputError as exc:
        assert str(exc) in {
            "CANDIDATE3_LLGUIDANCE_NOT_INSTALLED",
            "CANDIDATE3_LLGUIDANCE_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED",
        }
    else:
        raise AssertionError("backend must not be assumed installed")


def test_candidate3_llguidance_adapter_fails_closed_on_missing_backend(monkeypatch):
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3StructuredOutputError,
        build_candidate3_prefix_allowed_tokens_fn,
    )

    module = __import__("trustcxr.grounded_llm.candidate3_constrained_decoding", fromlist=["x"])
    original_import = module.importlib.import_module

    def missing(name):
        if name == "llguidance":
            raise ImportError("missing")
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    try:
        build_candidate3_prefix_allowed_tokens_fn("tokenizer", prompt_length=0)
    except Candidate3StructuredOutputError as exc:
        assert str(exc) == "CANDIDATE3_LLGUIDANCE_NOT_INSTALLED"
    else:
        raise AssertionError("missing LMFE must fail closed")


def test_candidate3_load_only_strategy_is_explicit_and_generation_free():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json").read_text()
    )
    strategy = config["load_only"]
    assert strategy["dtype"] == "bfloat16"
    assert strategy["device_strategy"] == "CPU_ONLY_BFLOAT16"
    assert strategy["device_policy"] == "cpu_only"
    assert strategy["transformers_device_map"] is None
    assert strategy["quantization"] == "none"
    assert strategy["local_files_only"] is True
    assert strategy["generation"] is False
    assert strategy["forward_pass"] is False


def test_candidate3_llguidance_schema_failure_fails_closed_without_generation():
    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3StructuredOutputError,
        build_candidate3_prefix_allowed_tokens_fn,
    )

    with pytest.raises(Candidate3StructuredOutputError):
        build_candidate3_prefix_allowed_tokens_fn("tokenizer", prompt_length=0)


def test_candidate3_llguidance_backend_identity_is_pinned():
    module = __import__("trustcxr.grounded_llm.candidate3_constrained_decoding", fromlist=["x"])
    assert module.BACKEND == "llguidance"
    assert module.PINNED_VERSION == "1.8.0"


def test_candidate3_llguidance_backend_is_configured():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json").read_text()
    )
    assert config["structured_output"]["backend"] == "llguidance"
    assert config["structured_output"]["version"] == "1.8.0"


def test_candidate3_llguidance_processor_preserves_prompt_boundary():
    import torch

    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3LLGuidanceLogitsProcessor,
    )

    class Matcher:
        def __init__(self):
            self.consumed = []

        def compute_bitmask(self):
            return bytes([0b00000100])

        def consume_token(self, token):
            self.consumed.append(token)
            return True

    matcher = Matcher()
    processor = Candidate3LLGuidanceLogitsProcessor(
        matcher,
        8,
        prompt_length=3,
        alignment={"mapping_identity_verified": True, "constraint_vocab_size": 8},
    )
    scores = torch.zeros((1, 8))
    processor(torch.tensor([[10, 11, 12]]), scores)
    assert matcher.consumed == []
    masked = processor(torch.tensor([[10, 11, 12, 2]]), scores)
    assert matcher.consumed == [2]
    assert torch.isfinite(masked[0, 2])
    assert torch.isneginf(masked[0, 0])


def test_candidate3_llguidance_expands_identity_domain_and_forbids_model_tail():
    import torch

    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3LLGuidanceLogitsProcessor,
    )

    class Matcher:
        def compute_bitmask(self):
            bits = bytearray(200029 // 8 + 1)
            bits[7 // 8] |= 1 << (7 % 8)
            return bytes(bits)

        def consume_token(self, token):
            return True

    processor = Candidate3LLGuidanceLogitsProcessor(
        Matcher(),
        200029,
        prompt_length=1,
        alignment={
            "mapping_identity_verified": True,
            "constraint_vocab_size": 200029,
            "model_vocab_size": 200064,
        },
    )
    masked = processor(torch.tensor([[42]]), torch.zeros((1, 200064)))
    assert tuple(masked.shape) == (1, 200064)
    assert torch.isfinite(masked[0, 7])
    assert torch.isneginf(masked[0, 6])
    assert torch.isneginf(masked[0, 200029:]).all()


def test_candidate3_vocab_alignment_rejects_unproven_mapping():
    import torch

    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        Candidate3LLGuidanceLogitsProcessor,
        Candidate3StructuredOutputError,
    )

    class Matcher:
        def compute_bitmask(self):
            return bytes([1])

    processor = Candidate3LLGuidanceLogitsProcessor(Matcher(), 8, prompt_length=0)
    with pytest.raises(Candidate3StructuredOutputError, match="VOCAB_ALIGNMENT_FAILED"):
        processor(torch.empty((1, 0), dtype=torch.long), torch.zeros((1, 8)))


def test_candidate3_deterministic_generation_omits_inert_temperature():
    path = ROOT / "scripts/training/run_ext4e_candidate3.py"
    spec = importlib.util.spec_from_file_location("candidate3_generation_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        eos_token_id = 199999

    class Constraint:
        logits_processor = object()

    kwargs = module.governed_generation_kwargs(Tokenizer(), Constraint())
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_candidate3_cpu_loader_uses_native_cpu_without_device_map(monkeypatch):
    import torch

    path = ROOT / "scripts/training/run_ext4e_candidate3_load_only.py"
    spec = importlib.util.spec_from_file_location("candidate3_load_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {}

    class Phi3ForCausalLM:
        class Tensor:
            dtype = torch.bfloat16
            device = torch.device("cpu")

            def numel(self):
                return 1

        def parameters(self):
            return iter((self.Tensor(),))

        def buffers(self):
            return iter((self.Tensor(),))

        def to(self, device):
            assert device == "cpu"
            return self

    def fake_load(*args, **kwargs):
        calls["model_path"] = args[0]
        calls.update(kwargs)
        return Phi3ForCausalLM(), {"missing_keys": [], "unexpected_keys": []}

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", fake_load)
    model, info = module.load_model_only()
    assert calls["dtype"].__str__() == "torch.bfloat16"
    assert "device_map" not in calls
    assert "torch_dtype" not in calls
    assert info["devices"] == ["cpu"]
