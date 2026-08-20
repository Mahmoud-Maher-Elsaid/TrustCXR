from __future__ import annotations

import json
from pathlib import Path

import torch

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.ext4f_realization import (
    realization_schema,
    realization_schema_sha256,
)

MODEL_ROOT = Path("cache/research_extensions/ext4g_candidate_gemma3_4b_it/models")
EXPECTED_SCHEMA_SHA = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"


def test_clean_manifest_and_index():
    manifest = json.loads(
        Path("configs/research_extensions/ext4g_gemma3_candidate_manifest.json").read_text()
    )
    assert len(manifest["files"]) == 15
    assert manifest["revision"] == "093f9f388b31de276ce2de164bdc2081324b9767"
    index = json.loads((MODEL_ROOT / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 883
    assert set(index["weight_map"].values()) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_gemma_llguidance_mask_alignment_without_inference():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer = processor.tokenizer
    assert realization_schema_sha256() == EXPECTED_SCHEMA_SHA
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "EXT-4G.1 probe."}]}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    constraint = build_llguidance_logits_processor(
        tokenizer,
        schema=realization_schema(),
        prompt_length=int(prompt.shape[-1]),
        model_vocab_size=262208,
        expected_schema_sha256=EXPECTED_SCHEMA_SHA,
    )
    scores = torch.zeros((1, 262208), dtype=torch.float32)
    masked = constraint.logits_processor(prompt, scores)
    assert tuple(masked.shape) == (1, 262208)
    assert torch.isinf(masked[0, 262145:]).all()
    assert constraint.vocab_alignment["constraint_vocab_size"] == 262145
    assert constraint.vocab_alignment["unregistered_model_tail_count"] == 63
    assert constraint.logits_processor._last_length == prompt.shape[-1]
