from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import build_ext4f_semantic_plan
from trustcxr.grounded_llm.ext4f_realization import build_ext4f_realization_request
from trustcxr.grounded_llm.ext4h_ledger import (
    assert_generation_monotonic,
    mark_generation_completed,
    mark_validation_result,
)
from trustcxr.grounded_llm.ext4h_slot_orchestration import (
    EXT4H_SLOT_REALIZATION_CONTRACT_V1,
    Ext4hSlotError,
    SlotTextResponse,
    assemble_ext4h_realization,
    build_ext4h_slot_manifest,
    compile_ext4h_slot_prompt,
    slot_realization_schema,
    slot_realization_schema_sha256,
    validate_ext4h_slot_manifest,
    validate_slot_generation_terminal,
)


def _fixture():
    evidence = build_synthetic_case("uncertainty").model_copy(
        update={"case_reference": "ext4h_slot_smoke_001"}
    )
    plan = build_ext4f_semantic_plan(evidence)
    request = build_ext4f_realization_request(plan)
    manifest = build_ext4h_slot_manifest(plan, request)
    return plan, request, manifest


def test_manifest_is_deterministic_and_required_ordered():
    plan, request, manifest = _fixture()
    again = build_ext4h_slot_manifest(plan, request)
    assert manifest == again
    assert manifest.manifest_sha256 == again.manifest_sha256
    assert [slot.ordinal for slot in manifest.slots] == list(range(1, len(manifest.slots) + 1))
    assert all(slot.required for slot in manifest.slots)
    assert all(slot.semantic_plan_sha256 == plan.semantic_plan_sha256 for slot in manifest.slots)
    assert manifest.contract_version == EXT4H_SLOT_REALIZATION_CONTRACT_V1
    validate_ext4h_slot_manifest(manifest, plan, request)


def test_slot_response_has_only_slot_text_and_assembly_is_deterministic():
    plan, request, manifest = _fixture()
    outputs = [{"slot_text": f"Bounded wording {index}."} for index, _ in enumerate(manifest.slots)]
    assembled = assemble_ext4h_realization(plan, request, manifest, outputs)
    assert [slot.slot_id for slot in assembled.slots] == [slot.slot_id for slot in manifest.slots]
    assert assemble_ext4h_realization(plan, request, manifest, outputs) == assembled
    with pytest.raises(ValidationError):
        SlotTextResponse.model_validate({"slot_text": "ok", "slot_id": "forbidden"})
    with pytest.raises(Ext4hSlotError):
        assemble_ext4h_realization(plan, request, manifest, outputs[:-1])


def test_model_cannot_supply_identity_or_duplicate_slots():
    plan, request, manifest = _fixture()
    assert "slot_id" not in SlotTextResponse.model_fields
    assert "slot_type" not in SlotTextResponse.model_fields
    outputs = [{"slot_text": "one"}] * len(manifest.slots)
    assembled = assemble_ext4h_realization(plan, request, manifest, outputs)
    assert len({slot.slot_id for slot in assembled.slots}) == len(assembled.slots)
    with pytest.raises(Ext4hSlotError):
        assemble_ext4h_realization(plan, request, manifest, outputs + [{"slot_text": "extra"}])


def test_slot_schema_and_prompt_are_bounded():
    schema = slot_realization_schema()
    assert schema["additionalProperties"] is False
    assert slot_realization_schema_sha256()
    _, _, manifest = _fixture()
    prompt = compile_ext4h_slot_prompt(manifest.slots[0])
    assert EXT4H_SLOT_REALIZATION_CONTRACT_V1 in prompt
    properties = json.loads(json.dumps(SlotTextResponse.model_json_schema())).get("properties", {})
    assert "slot_id" not in properties
    assert properties["slot_text"]["minLength"] == 1
    assert properties["slot_text"]["maxLength"] == 512


def test_slot_schema_compiles_with_gemma_llguidance():
    from transformers import AutoProcessor

    from trustcxr.grounded_llm.candidate3_constrained_decoding import (
        build_llguidance_logits_processor,
    )

    root = Path("cache/research_extensions/ext4g_candidate_gemma3_4b_it/models")
    processor = AutoProcessor.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    rendered = "EXT4H_SLOT_REALIZATION_CONTRACT_V1"
    inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
    constraint = build_llguidance_logits_processor(
        processor.tokenizer,
        schema=slot_realization_schema(),
        expected_schema_sha256=slot_realization_schema_sha256(),
        prompt_length=int(inputs["input_ids"].shape[1]),
        model_vocab_size=262208,
    )
    assert constraint.matcher is not None
    assert constraint.vocab_alignment["mapping_identity_verified"] is True


def test_truncation_fails_closed():
    class Matcher:
        def is_accepting(self):
            return False

    with pytest.raises(Ext4hSlotError, match="SLOT_GENERATION_TRUNCATED"):
        validate_slot_generation_terminal(Matcher(), 128, 128)


def test_ledger_generation_completion_is_monotonic():
    entry = {"generation_completed": False, "generated_tokens": 0}
    mark_generation_completed(entry, 7)
    mark_validation_result(entry, "parse_status", "FAIL")
    assert_generation_monotonic(entry)
    assert entry["generation_completed"] is True
    assert entry["generated_tokens"] == 7


def test_ext4h_governance_boundaries_and_stage18_preserved():
    module_text = Path("src/trustcxr/grounded_llm/ext4h_slot_orchestration.py").read_text()
    report = json.loads(
        Path(
            "reports/research_extensions/ext4h/EXT4H1_SLOT_ORCHESTRATED_ARCHITECTURE_REPORT.json"
        ).read_text()
    )
    assert "ext4f_dev_" not in module_text
    assert "EXT-4D" not in module_text
    assert report["historical_benchmark_accessed"] == 0
    assert report["frozen_final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False
    assert report["stage18_changed"] is False


def test_ext4h2_runner_is_new_slot_only_and_has_no_benchmark_path():
    runner = Path("scripts/research_extensions/run_ext4h2_slot_smoke.py").read_text()
    assert "ext4h_slot_smoke_001" in runner
    assert "ext4f_dev_" not in runner
    assert "EXT4F_DEVELOPMENT_BENCHMARK" not in runner
    assert "frozen_final_cases_accessed" in runner
    assert "locked_test_accessed" in runner
    assert runner.count("model.generate(") == 1
    assert 'attention_mask=inputs["attention_mask"]' in runner
    assert 'retry_count": 0' in runner


def test_ext4h2_runner_uses_monotonic_generation_fields():
    runner = Path("scripts/research_extensions/run_ext4h2_slot_smoke.py").read_text()
    assert "mark_generation_completed" in runner
    assert "generation_completed" in runner
    assert "SLOT_MAX_NEW_TOKENS" in runner
