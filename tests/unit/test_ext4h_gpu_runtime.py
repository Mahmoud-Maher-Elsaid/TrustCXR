from pathlib import Path

import pytest

from trustcxr.grounded_llm.ext4h_gpu_runtime import (
    CONSTRAINT_VOCAB_SIZE,
    GPU_RUNTIME_ID,
    MODEL_REVISION,
    MODEL_VOCAB_SIZE,
    Ext4hGpuRuntimeError,
    validate_gemma_vocab_alignment,
)


def test_gpu_runtime_identity_and_vocab_policy_are_frozen():
    assert GPU_RUNTIME_ID == "EXT4H_GEMMA3_GPU_INT8_V1"
    assert MODEL_REVISION == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert MODEL_VOCAB_SIZE == 262208
    assert CONSTRAINT_VOCAB_SIZE == 262145


def test_vocab_alignment_rejects_holes_and_accepts_identity_prefix():
    class Tokenizer:
        def get_vocab(self):
            return {str(index): index for index in range(CONSTRAINT_VOCAB_SIZE)}

    result = validate_gemma_vocab_alignment(Tokenizer())
    assert result["tail_count"] == 63
    assert result["tail_policy"] == "always_forbidden"

    class BrokenTokenizer:
        def get_vocab(self):
            return {"a": 0, "b": 2}

    with pytest.raises(Ext4hGpuRuntimeError, match="VOCAB_ALIGNMENT"):
        validate_gemma_vocab_alignment(BrokenTokenizer())


def test_gpu_runner_is_three_slot_new_synthetic_only():
    runner = Path("scripts/research_extensions/run_ext4hg1_gpu_int8_smoke.py").read_text()
    assert "ext4h_gpu_int8_smoke_001" in runner
    assert "ext4h_slot_smoke_001" not in runner
    assert "EXT4F_DEVELOPMENT_BENCHMARK" not in runner
    assert "frozen_final_cases_accessed" in runner
    assert "locked_test_accessed" in runner
    assert runner.count("model.generate(") == 1
    assert "quantization_config=build_int8_quantization_config()" in runner
    assert 'attention_mask=inputs["attention_mask"]' in runner


def test_cpu_fallback_is_explicitly_rejected():
    runtime = Path("src/trustcxr/grounded_llm/ext4h_gpu_runtime.py").read_text()
    assert "EXT4HG1_CPU_MODEL_FALLBACK" in runtime
    assert "gpu_execution_verified" in runtime
