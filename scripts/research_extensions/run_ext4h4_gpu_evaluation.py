"""Governed EXT-4H.4 fresh benchmark evaluator.

The runner is intentionally GPU-only and has no retry or repair path. It
materializes only the frozen EXT-4H.3 recipes through the deterministic
planner and slot manifest builder.
"""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.ext4f_contracts import Ext4fSemanticPlan
from trustcxr.grounded_llm.ext4f_realization import RealizationRequest
from trustcxr.grounded_llm.ext4h3_benchmark import (
    build_ext4h3_cases,
    validate_ext4h3_design,
)
from trustcxr.grounded_llm.ext4h_gpu_runtime import (
    MODEL_REVISION,
    MODEL_VOCAB_SIZE,
    TAIL_END,
    TAIL_START,
    build_int8_quantization_config,
    cuda_preflight,
    validate_gemma_vocab_alignment,
    validate_int8_model_placement,
    verify_bitsandbytes_cuda,
)
from trustcxr.grounded_llm.ext4h_slot_orchestration import (
    SLOT_MAX_NEW_TOKENS,
    SlotManifest,
    SlotTextResponse,
    assemble_ext4h_realization,
    compile_ext4h_slot_prompt,
    slot_realization_schema,
    slot_realization_schema_sha256,
    validate_slot_generation_terminal,
    validate_slot_text_response,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "cache/research_extensions/ext4g_candidate_gemma3_4b_it/models"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4h4"
REPORT_PATH = ROOT / "reports/research_extensions/ext4h/EXT4H4_GPU_INT8_EVALUATION_REPORT.json"
BENCHMARK_ID = "EXT4H_FRESH_DEVELOPMENT_BENCHMARK_V1"
BENCHMARK_SHA = "1c34ce622fbf68af9b5114ddbf0f73fcfabffabd36dfc7536ad0c01e5402d324"
SCHEMA_SHA = slot_realization_schema_sha256()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _case_identity(case: dict) -> str:
    return _sha(
        {
            "evidence": case["evidence"],
            "semantic_plan": case["semantic_plan"],
            "slot_manifest": case["slot_manifest"],
        }
    )


def _preflight() -> dict:
    if sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError("EXT4H4_RUNTIME_VERSION_FAILED")
    if subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).returncode:
        raise RuntimeError("EXT4H4_PIP_CHECK_FAILED")
    cuda = cuda_preflight(torch)
    bnb = verify_bitsandbytes_cuda()
    if bnb["version"] != "0.50.0" or cuda["gpu_name"] != "NVIDIA GeForce RTX 3070 Ti Laptop GPU":
        raise RuntimeError("EXT4H4_GPU_IDENTITY_FAILED")
    return {"cuda": cuda, "bitsandbytes": bnb, "pip_check": "PASS"}


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "run_manifest.json"
    ledger = {
        "stage": "EXT-4H.4",
        "run_id": run_id,
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "model_revision": MODEL_REVISION,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_sha256": BENCHMARK_SHA,
        "realization_slot_schema_sha256": SCHEMA_SHA,
        "cases_attempted": 0,
        "slots_expected": 84,
        "slots_attempted": 0,
        "model_load_count": 0,
        "model_generate_calls": 0,
        "retry_count": 0,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "case_ledgers": [],
        "protocol_deviation_count": 0,
        "authority_mutations": 0,
        "phase": "PREPARED",
        "terminal_status": "EXT4H4_NOT_STARTED",
    }
    _write(ledger_path, ledger)
    model = processor = None
    try:
        ledger["runtime_preflight"] = _preflight()
        cases = build_ext4h3_cases()
        validate_ext4h3_design(cases)
        if len(cases) != 24 or sum(len(case["slot_manifest"]["slots"]) for case in cases) != 84:
            raise RuntimeError("EXT4H4_FROZEN_BENCHMARK_SHAPE_FAILED")
        ledger["phase"] = "BENCHMARK_PREFLIGHT_PASS"
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        tokenizer = processor.tokenizer
        if validate_gemma_vocab_alignment(tokenizer)["tail_count"] != 63:
            raise RuntimeError("EXT4H4_VOCAB_ALIGNMENT_FAILED")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ROOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            quantization_config=build_int8_quantization_config(),
            device_map={"": 0},
        )
        placement = validate_int8_model_placement(model)
        ledger["model_load_count"] = 1
        ledger["model_placement"] = placement
        if placement["quantized_linear8bitlt_modules"] != 400:
            raise RuntimeError("EXT4H4_QUANTIZED_MODULE_COUNT_FAILED")
        ledger["phase"] = "MODEL_LOAD_PASS"
        for case in cases:
            case_id = case["case_id"]
            manifest = case["slot_manifest"]
            case_entry = {
                "case_id": case_id,
                "case_sha256": _case_identity(case),
                "semantic_plan_sha256": case["semantic_plan"]["semantic_plan_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "slot_ledgers": [],
                "automatic_hard_gate": "NOT_RUN",
                "semantic_review_status": "NOT_RUN",
            }
            ledger["cases_attempted"] += 1
            ledger["development_cases_accessed"] += 1
            outputs: list[SlotTextResponse] = []
            case_failed = False
            for slot in manifest["slots"]:
                rendered = processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": compile_ext4h_slot_prompt(type("Slot", (), slot)()),
                                }
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
                if "attention_mask" not in inputs:
                    raise RuntimeError("EXT4H4_ATTENTION_MASK_MISSING")
                input_device = model.get_input_embeddings().weight.device
                inputs = {key: value.to(input_device) for key, value in inputs.items()}
                prompt_len = int(inputs["input_ids"].shape[-1])
                constraint = build_llguidance_logits_processor(
                    tokenizer,
                    schema=slot_realization_schema(),
                    expected_schema_sha256=SCHEMA_SHA,
                    prompt_length=prompt_len,
                    model_vocab_size=MODEL_VOCAB_SIZE,
                )
                assert_generation_constraint(constraint, expected_schema_sha256=SCHEMA_SHA)
                probe = constraint.logits_processor(
                    inputs["input_ids"], torch.zeros((1, MODEL_VOCAB_SIZE), device=input_device)
                )
                if (
                    tuple(probe.shape) != (1, MODEL_VOCAB_SIZE)
                    or not torch.isinf(probe[0, TAIL_START : TAIL_END + 1]).all()
                ):
                    raise RuntimeError("EXT4H4_TAIL_MASK_FAILED")
                slot_entry = {
                    "slot_id": slot["slot_id"],
                    "slot_type": slot["slot_type"],
                    "ordinal": slot["ordinal"],
                    "prompt_sha256": _sha(rendered),
                    "prompt_token_count": prompt_len,
                    "attention_mask_present": True,
                    "input_ids_shape": list(inputs["input_ids"].shape),
                    "attention_mask_shape": list(inputs["attention_mask"].shape),
                    "input_ids_device": str(inputs["input_ids"].device),
                    "attention_mask_device": str(inputs["attention_mask"].device),
                    "request_count": 1,
                    "generation_started": True,
                    "generation_completed": False,
                    "vram_before": int(torch.cuda.memory_allocated(0)),
                }
                case_entry["slot_ledgers"].append(slot_entry)
                _write(ledger_path, ledger)
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
                slot_entry.update(
                    {
                        "generation_completed": True,
                        "generated_tokens": int(generated.shape[0]),
                        "generation_duration_seconds": time.perf_counter() - started,
                        "vram_after": int(torch.cuda.memory_allocated(0)),
                        "vram_peak": int(torch.cuda.max_memory_allocated(0)),
                    }
                )
                ledger["model_generate_calls"] += 1
                ledger["slots_attempted"] += 1
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                raw_path = run_dir / f"{case_id}_{slot['ordinal']:02d}_{slot['slot_id']}_raw.txt"
                raw_path.write_text(raw, encoding="utf-8")
                slot_entry["raw_continuation_preserved"] = True
                try:
                    validate_slot_generation_terminal(
                        constraint.matcher, len(generated), SLOT_MAX_NEW_TOKENS
                    )
                    parsed = validate_slot_text_response(json.loads(raw))
                    slot_entry.update(
                        {
                            "parse_status": "PASS",
                            "slot_contract_status": "PASS",
                            "slot_truncation": False,
                        }
                    )
                    outputs.append(parsed)
                except Exception as exc:
                    slot_entry.update(
                        {
                            "parse_status": "FAIL",
                            "slot_contract_status": "FAIL",
                            "failure_code": str(exc),
                            "slot_truncation": "SLOT_GENERATION_TRUNCATED" in str(exc),
                        }
                    )
                    case_failed = True
                _write(ledger_path, ledger)
            if not case_failed and len(outputs) == len(manifest["slots"]):
                assembled = assemble_ext4h_realization(
                    Ext4fSemanticPlan.model_validate(case["semantic_plan"]),
                    RealizationRequest.model_validate(case["realization_request"]),
                    SlotManifest.model_validate(manifest),
                    outputs,
                )
                _write(run_dir / f"{case_id}_assembled.json", assembled.model_dump(mode="json"))
                case_entry["automatic_hard_gate"] = "PASS"
                case_entry["semantic_review_status"] = "REVIEW_REQUIRED"
            else:
                case_entry["automatic_hard_gate"] = "FAIL"
                case_entry["semantic_review_status"] = (
                    "NOT_REQUIRED_FOR_SELECTION_AFTER_AUTOMATIC_GATE_FAILURE"
                )
            ledger["case_ledgers"].append(case_entry)
            _write(ledger_path, ledger)
        gate_pass = sum(case["automatic_hard_gate"] == "PASS" for case in ledger["case_ledgers"])
        slot_ledgers = [slot for case in ledger["case_ledgers"] for slot in case["slot_ledgers"]]
        durations = [
            slot["generation_duration_seconds"]
            for slot in slot_ledgers
            if "generation_duration_seconds" in slot
        ]
        parse_valid = sum(slot.get("parse_status") == "PASS" for slot in slot_ledgers)
        contract_valid = sum(slot.get("slot_contract_status") == "PASS" for slot in slot_ledgers)
        ledger.update(
            {
                "slot_generation_completed": sum(
                    slot.get("generation_completed") is True for slot in slot_ledgers
                ),
                "slot_parse_valid": parse_valid,
                "slot_parse_valid_rate": parse_valid / len(slot_ledgers) if slot_ledgers else 0.0,
                "slot_contract_valid": contract_valid,
                "slot_contract_valid_rate": contract_valid / len(slot_ledgers)
                if slot_ledgers
                else 0.0,
                "slot_truncated": sum(slot.get("slot_truncation") is True for slot in slot_ledgers),
                "assembled_cases": gate_pass,
                "generation_duration_total_seconds": sum(durations),
                "generation_duration_mean_seconds": statistics.mean(durations)
                if durations
                else None,
                "generation_duration_median_seconds": statistics.median(durations)
                if durations
                else None,
                "generation_duration_p95_seconds": (
                    statistics.quantiles(durations, n=20, method="inclusive")[18]
                    if len(durations) > 1
                    else (durations[0] if durations else None)
                ),
                "automatic_case_hard_gate_pass": gate_pass,
                "automatic_case_hard_gate_fail": 24 - gate_pass,
                "semantic_review_required_cases": gate_pass,
                "phase": "TERMINAL",
                "terminal_status": "EXT4H4_AUTOMATIC_GATE_PASS_REVIEW_REQUIRED"
                if gate_pass == 24
                else "EXT4H4_DEVELOPMENT_GATE_FAILED",
            }
        )
    except Exception as exc:
        ledger.update(
            {
                "phase": "TERMINAL",
                "terminal_status": "EXT4H4_EXECUTION_INVALID_SYSTEMIC_DEFECT",
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ledger["cleanup_status"] = "PASS"
        _write(ledger_path, ledger)
        _write(REPORT_PATH, ledger)
    print(json.dumps(ledger, indent=2, default=str))
    return (
        0
        if ledger["terminal_status"]
        in {"EXT4H4_AUTOMATIC_GATE_PASS_REVIEW_REQUIRED", "EXT4H4_DEVELOPMENT_GATE_FAILED"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
