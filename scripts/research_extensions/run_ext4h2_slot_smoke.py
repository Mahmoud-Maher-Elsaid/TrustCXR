"""Run one EXT-4H.2 slot-orchestrated Gemma smoke.

This runner is intentionally separate from all EXT-4F benchmark runners.  It
creates one new synthetic plan, then performs one constrained generation per
deterministic manifest slot.  No historical, final, or locked partitions are
opened by this module.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import (
    build_ext4f_semantic_plan,
    validate_ext4f_semantic_plan,
)
from trustcxr.grounded_llm.ext4f_realization import (
    build_ext4f_realization_request,
    validate_ext4f_realization_request,
)
from trustcxr.grounded_llm.ext4h_ledger import mark_generation_completed, mark_validation_result
from trustcxr.grounded_llm.ext4h_slot_orchestration import (
    EXT4H_SLOT_REALIZATION_CONTRACT_V1,
    SLOT_MAX_NEW_TOKENS,
    SlotTextResponse,
    assemble_ext4h_realization,
    build_ext4h_slot_manifest,
    compile_ext4h_slot_prompt,
    slot_realization_schema,
    slot_realization_schema_sha256,
    validate_slot_generation_terminal,
    validate_slot_text_response,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "cache/research_extensions/ext4g_candidate_gemma3_4b_it/models"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4h2"
REPORT_PATH = ROOT / "reports/research_extensions/ext4h/EXT4H2_SLOT_SYNTHETIC_SMOKE_REPORT.json"
CASE_ID = "ext4h_slot_smoke_001"
SCHEMA_SHA = slot_realization_schema_sha256()
MODEL_VOCAB = 262208
CONSTRAINT_VOCAB = 262145
GENERATION_POLICY = {
    "version": "EXT4H_SLOT_GENERATION_POLICY_V1",
    "max_new_tokens": SLOT_MAX_NEW_TOKENS,
    "do_sample": False,
    "retry_count": 0,
    "tail_policy": "identity_prefix_0_262144_model_only_tail_forbidden",
}


def _sha(value: Any) -> str:
    raw = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        value if isinstance(value, str) else json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _evidence():
    return build_synthetic_case("uncertainty").model_copy(update={"case_reference": CASE_ID})


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt_dir = ARTIFACT_ROOT / run_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = attempt_dir / "run_ledger.json"
    record: dict[str, Any] = {
        "stage": "EXT-4H.2",
        "run_id": run_id,
        "attempt_id": f"ext4h2_slot_{run_id}",
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "repository": "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "synthetic_case_id": CASE_ID,
        "slot_contract": EXT4H_SLOT_REALIZATION_CONTRACT_V1,
        "realization_schema_sha256": SCHEMA_SHA,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "generate_call_count": 0,
        "slot_attempts": [],
        "authority_mutations": 0,
        "phase": "PREPARED",
        "terminal_status": "EXT4H2_NOT_STARTED",
    }
    _write(ledger_path, record)
    model = processor = None
    try:
        if (
            sys.version_info[:3] != (3, 12, 10)
            or importlib.metadata.version("llguidance") != "1.8.0"
        ):
            raise RuntimeError("EXT4H2_RUNTIME_PREFLIGHT_FAILED")
        evidence = _evidence()
        plan = build_ext4f_semantic_plan(evidence)
        request = build_ext4f_realization_request(plan)
        validate_ext4f_semantic_plan(plan)
        validate_ext4f_realization_request(request)
        manifest = build_ext4h_slot_manifest(plan, request)
        record.update(
            {
                "semantic_plan_sha256": plan.semantic_plan_sha256,
                "realization_request_sha256": request.realization_request_sha256,
                "manifest_sha256": manifest.manifest_sha256,
                "manifest_slot_ids": [item.slot_id for item in manifest.slots],
                "phase": "AUTHORITY_PREFLIGHT_PASS",
            }
        )
        _write(attempt_dir / "synthetic_evidence.json", evidence.model_dump_json(indent=2) + "\n")
        _write(attempt_dir / "semantic_plan.json", plan.model_dump_json(indent=2) + "\n")
        _write(attempt_dir / "realization_request.json", request.model_dump_json(indent=2) + "\n")
        _write(attempt_dir / "slot_manifest.json", manifest.model_dump_json(indent=2) + "\n")

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        tokenizer = processor.tokenizer
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16
        )
        if (
            type(model).__name__ != "Gemma3ForConditionalGeneration"
            or sum(p.numel() for p in model.parameters()) != 4300079472
        ):
            raise RuntimeError("EXT4H2_MODEL_IDENTITY_FAILED")
        if {str(p.device) for p in model.parameters()} != {"cpu"}:
            raise RuntimeError("EXT4H2_NON_CPU_PLACEMENT")
        record["phase"] = "MODEL_LOAD_PASS"

        outputs: list[SlotTextResponse] = []
        for item in manifest.slots:
            prompt = compile_ext4h_slot_prompt(item)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
            if "attention_mask" not in inputs:
                raise RuntimeError("EXT4H2_ATTENTION_MASK_MISSING")
            if any(
                key in inputs for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask")
            ):
                raise RuntimeError("EXT4H2_IMAGE_INPUT_FORBIDDEN")
            prompt_len = int(inputs["input_ids"].shape[-1])
            constraint = build_llguidance_logits_processor(
                tokenizer,
                schema=slot_realization_schema(),
                expected_schema_sha256=SCHEMA_SHA,
                prompt_length=prompt_len,
                model_vocab_size=MODEL_VOCAB,
            )
            assert_generation_constraint(constraint, expected_schema_sha256=SCHEMA_SHA)
            score_probe = constraint.logits_processor(
                inputs["input_ids"], torch.zeros((1, MODEL_VOCAB))
            )
            if (
                tuple(score_probe.shape) != (1, MODEL_VOCAB)
                or not torch.isinf(score_probe[0, CONSTRAINT_VOCAB:]).all()
            ):
                raise RuntimeError("EXT4H2_TAIL_MASK_FAILED")
            entry = {
                "slot_id": item.slot_id,
                "prompt_token_count": prompt_len,
                "attention_mask_present": True,
                "attention_mask_shape": list(inputs["attention_mask"].shape),
                "generation_started": True,
                "generation_completed": False,
                "request_count": 1,
            }
            record["slot_attempts"].append(entry)
            record["phase"] = "SLOT_GENERATION_AUTHORIZED"
            _write(ledger_path, record)
            started = time.perf_counter()
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=SLOT_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[constraint.logits_processor],
            )
            generated = output_ids[0][prompt_len:]
            mark_generation_completed(entry, int(generated.shape[0]))
            entry["generation_duration_seconds"] = time.perf_counter() - started
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            _write(attempt_dir / f"{item.slot_id}_raw.txt", raw)
            validate_slot_generation_terminal(
                constraint.matcher, int(generated.shape[0]), SLOT_MAX_NEW_TOKENS
            )
            parsed = validate_slot_text_response(json.loads(raw))
            mark_validation_result(entry, "parse_status", "PASS")
            mark_validation_result(entry, "slot_contract_status", "PASS")
            record["generate_call_count"] += 1
            outputs.append(parsed)
            _write(ledger_path, record)
        assembled = assemble_ext4h_realization(plan, request, manifest, outputs)
        _write(
            attempt_dir / "assembled_realization.json", assembled.model_dump_json(indent=2) + "\n"
        )
        record.update(
            {
                "assembly_status": "PASS",
                "final_validation_status": "PASS",
                "authority_mutations": 0,
                "cleanup_status": "PASS",
                "phase": "TERMINAL",
                "terminal_status": "EXT4H2_SLOT_ORCHESTRATED_SYNTHETIC_TECHNICAL_SMOKE_PASS",
            }
        )
    except Exception as exc:
        record.update(
            {
                "terminal_status": "EXT4H2_SLOT_ORCHESTRATED_TECHNICAL_FAILURE",
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        record.setdefault("cleanup_status", "PASS")
        _write(ledger_path, record)
        _write(REPORT_PATH, record)
    print(json.dumps(record, indent=2, default=str))
    return 0 if record["terminal_status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
