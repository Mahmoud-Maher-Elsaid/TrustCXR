from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: E501

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research_extensions/run_ext4g3_gemma_development.py"


def test_attention_mask_and_frozen_runner_policy():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"attention_mask"' in text
    assert "EXT4G3_ATTENTION_MASK_MISSING" in text
    assert 'attention_mask=inputs["attention_mask"]' in text
    assert "model.generate(" in text
    assert text.count("model.generate(") == 1
    assert "GENERATION_POLICY" in text
    assert "EXT4F_DEVELOPMENT_BENCHMARK_V1" in text
    assert "262145" in text and "262208" in text


def test_ext4g3_report_identity_and_tripwires():
    report = json.loads(
        (
            ROOT
            / "reports/research_extensions/ext4g/EXT4G3_GEMMA_DEVELOPMENT_EVALUATION_REPORT.json"
        ).read_text()
    )
    assert report["development_cases"] == 24
    assert (
        report["benchmark_sha256"]
        == "671a04d2d859f1b1ffb9414a8c0f636596949748a00548e45abcbbfdb752db61"
    )
    assert report["attention_mask_required"] is True
    assert report["development_cases_accessed"] == 0
    assert report["frozen_final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False


def test_gemma_attention_mask_matches_prompt_tokenization_without_inference():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        ROOT / "cache/research_extensions/ext4g_candidate_gemma3_4b_it/models",
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "mask probe"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
    assert "attention_mask" in inputs
    assert tuple(inputs["attention_mask"].shape) == tuple(inputs["input_ids"].shape)
    assert inputs["attention_mask"].tolist() == [[1] * inputs["input_ids"].shape[1]]
