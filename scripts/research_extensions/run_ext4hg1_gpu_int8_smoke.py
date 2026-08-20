"""Run the governed EXT-4H.G1 CUDA INT8 three-slot technical smoke."""

from __future__ import annotations

import gc
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import build_ext4f_semantic_plan
from trustcxr.grounded_llm.ext4f_realization import build_ext4f_realization_request
from trustcxr.grounded_llm.ext4h_gpu_runtime import (
    GPU_RUNTIME_ID,
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
from trustcxr.grounded_llm.ext4h_ledger import mark_generation_completed
from trustcxr.grounded_llm.ext4h_slot_orchestration import (
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
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4hg1"
REPORT_PATH = ROOT / "reports/research_extensions/ext4h/EXT4HG1_GEMMA3_GPU_INT8_RUNTIME_REPORT.json"
CASE_ID = "ext4h_gpu_int8_smoke_001"
SCHEMA_SHA = slot_realization_schema_sha256()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = {
        "stage": "EXT-4H.G1",
        "runtime_id": GPU_RUNTIME_ID,
        "run_id": run_id,
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "model_revision": MODEL_REVISION,
        "synthetic_case_id": CASE_ID,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "slot_attempts": [],
        "retry_count": 0,
        "phase": "PREPARED",
        "terminal_status": "EXT4HG1_NOT_STARTED",
    }
    ledger_path = run_dir / "run_ledger.json"
    _write(ledger_path, ledger)
    model = processor = None
    try:
        ledger["cuda"] = cuda_preflight(torch)
        ledger["bitsandbytes"] = verify_bitsandbytes_cuda()
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        evidence = build_synthetic_case("uncertainty").model_copy(
            update={"case_reference": CASE_ID}
        )
        plan = build_ext4f_semantic_plan(evidence)
        request = build_ext4f_realization_request(plan)
        manifest = build_ext4h_slot_manifest(plan, request)
        if len(manifest.slots) != 3:
            raise RuntimeError("EXT4HG1_SYNTHETIC_SLOT_COUNT_FAILED")
        processor = AutoProcessor.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        tokenizer = processor.tokenizer
        ledger["vocab_alignment"] = validate_gemma_vocab_alignment(tokenizer)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ROOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            quantization_config=build_int8_quantization_config(),
            device_map={"": 0},
        )
        placement = validate_int8_model_placement(model)
        ledger["model_placement"] = placement
        ledger["phase"] = "STRUCTURED_PREFLIGHT_PASS"
        outputs: list[SlotTextResponse] = []
        for item in manifest.slots:
            rendered = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": compile_ext4h_slot_prompt(item)}],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
            if "attention_mask" not in inputs:
                raise RuntimeError("EXT4HG1_ATTENTION_MASK_MISSING")
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
                raise RuntimeError("EXT4HG1_TAIL_MASK_FAILED")
            entry = {
                "slot_id": item.slot_id,
                "attention_mask_present": True,
                "attention_mask_shape": list(inputs["attention_mask"].shape),
                "input_ids_device": str(inputs["input_ids"].device),
                "attention_mask_device": str(inputs["attention_mask"].device),
                "request_count": 1,
                "generation_started": True,
                "generation_completed": False,
                "vram_before": int(torch.cuda.memory_allocated(0)),
            }
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
            entry.update(
                {
                    "generation_duration_seconds": time.perf_counter() - started,
                    "vram_after": int(torch.cuda.memory_allocated(0)),
                    "vram_peak": int(torch.cuda.max_memory_allocated(0)),
                }
            )
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            (run_dir / f"{item.slot_id}_raw.txt").write_text(raw, encoding="utf-8")
            validate_slot_generation_terminal(
                constraint.matcher, len(generated), SLOT_MAX_NEW_TOKENS
            )
            outputs.append(validate_slot_text_response(json.loads(raw)))
            ledger["model_generate_calls"] += 1
            ledger["slot_attempts"].append(entry)
            _write(ledger_path, ledger)
        assembled = assemble_ext4h_realization(plan, request, manifest, outputs)
        _write(run_dir / "assembled_realization.json", assembled.model_dump(mode="json"))
        ledger.update(
            {
                "authority_mutations": 0,
                "cleanup_status": "PASS",
                "phase": "TERMINAL",
                "terminal_status": "EXT4HG1_GEMMA3_GPU_INT8_TECHNICAL_SMOKE_PASS",
            }
        )
    except Exception as exc:
        ledger.update(
            {
                "terminal_status": "EXT4HG1_TECHNICAL_FAILURE",
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _write(ledger_path, ledger)
        _write(REPORT_PATH, ledger)
    print(json.dumps(ledger, indent=2, default=str))
    return 0 if ledger["terminal_status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
