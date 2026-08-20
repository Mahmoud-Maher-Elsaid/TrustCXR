"""Run exactly one EXT-4G.2 Gemma realization-only technical smoke.

The runner creates one new synthetic evidence object, validates the frozen
EXT-4F contracts, and performs one constrained text-only generation. It never
opens the EXT-4F benchmark or any final/locked partition.
"""
# ruff: noqa: E501

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    Candidate3StructuredOutputError,
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import (
    build_ext4f_semantic_plan,
    validate_ext4f_semantic_plan,
)
from trustcxr.grounded_llm.ext4f_realization import (
    EXT4F_REALIZATION_CONTRACT_V1,
    LLM_REALIZATION_ONLY_FIELDS,
    build_ext4f_realization_request,
    compile_ext4f_realization_prompt,
    realization_schema,
    realization_schema_sha256,
    validate_ext4f_realization_request,
    validate_ext4f_realization_response,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "cache/research_extensions/ext4g_candidate_gemma3_4b_it/models"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4g2"
REPORT_PATH = (
    ROOT / "reports/research_extensions/ext4g/EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_SMOKE_REPORT.json"
)
CASE_ID = "ext4g2_gemma_synthetic_001"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
SCHEMA_SHA = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
EXPECTED_TORCH = "2.12.1+cu130"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_LLGUIDANCE = "1.8.0"
GENERATION_POLICY = {
    "version": "EXT4F5_REALIZATION_GENERATION_POLICY_V1",
    "max_new_tokens": 512,
    "do_sample": False,
    "top_p": 1.0,
    "seed": 20260819,
    "temperature": None,
    "retry_count": 0,
}


def _sha(value: Any) -> str:
    if isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _synthetic_evidence():
    return build_synthetic_case("uncertainty").model_copy(update={"case_reference": CASE_ID})


def _runtime_preflight() -> dict[str, Any]:
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llguidance": importlib.metadata.version("llguidance"),
    }
    if versions != {
        "python": "3.12.10",
        "torch": EXPECTED_TORCH,
        "transformers": EXPECTED_TRANSFORMERS,
        "llguidance": EXPECTED_LLGUIDANCE,
    }:
        raise RuntimeError("EXT4G2_RUNTIME_PREFLIGHT_FAILED")
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if pip_result.returncode:
        raise RuntimeError("EXT4G2_PIP_CHECK_FAILED")
    return {**versions, "pip_check": "PASS"}


def _authority_preflight(plan, request, schema_sha: str) -> None:
    validate_ext4f_semantic_plan(plan)
    validate_ext4f_realization_request(request)
    if schema_sha != SCHEMA_SHA or set(LLM_REALIZATION_ONLY_FIELDS) != {"slot_text"}:
        raise RuntimeError("EXT4G2_AUTHORITY_PREFLIGHT_FAILED")
    if any(field in request.model_dump(mode="json") for field in LLM_REALIZATION_ONLY_FIELDS):
        raise RuntimeError("EXT4G2_AUTHORITY_PREFLIGHT_FAILED")


def _write(path: Path, value: Any) -> None:
    path.write_text(
        value if isinstance(value, str) else json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt_id = f"ext4g2_gemma_{run_id}"
    attempt_dir = ARTIFACT_ROOT / run_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = attempt_dir / "run_ledger.json"
    record: dict[str, Any] = {
        "stage": "EXT-4G.2",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "repository": "google/gemma-3-4b-it",
        "revision": REVISION,
        "synthetic_case_id": CASE_ID,
        "semantic_contract": "EXT4F_SEMANTIC_GENERATION_CONTRACT_V1",
        "realization_contract": EXT4F_REALIZATION_CONTRACT_V1,
        "structured_backend": "llguidance",
        "backend_version": EXPECTED_LLGUIDANCE,
        "generate_call_count": 0,
        "generation_started": False,
        "generation_completed": False,
        "generated_tokens": 0,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "model_load_started": False,
        "model_load_completed": False,
        "authority_mutations": 0,
        "phase": "PREPARED",
        "terminal_status": "EXT4G2_NOT_STARTED",
    }
    model = processor = constraint = None
    try:
        record["runtime"] = _runtime_preflight()
        evidence = _synthetic_evidence()
        plan = build_ext4f_semantic_plan(evidence)
        request = build_ext4f_realization_request(plan)
        schema = realization_schema()
        schema_sha = realization_schema_sha256()
        prompt_text = compile_ext4f_realization_prompt(request)
        prompt_sha = _sha(prompt_text)
        _authority_preflight(plan, request, schema_sha)
        record.update(
            {
                "semantic_plan_sha256": plan.semantic_plan_sha256,
                "realization_request_sha256": request.realization_request_sha256,
                "realization_schema_sha256": schema_sha,
                "prompt_sha256": prompt_sha,
                "authority_preflight": "EXT4G2_AUTHORITY_PREFLIGHT_PASS",
                "phase": "AUTHORITY_PREFLIGHT_PASS",
            }
        )
        _write(attempt_dir / "synthetic_evidence.json", evidence.model_dump_json(indent=2) + "\n")
        _write(attempt_dir / "semantic_plan.json", plan.model_dump_json(indent=2) + "\n")
        _write(attempt_dir / "realization_request.json", request.model_dump_json(indent=2) + "\n")

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        tokenizer = processor.tokenizer
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Wording-only research realization. Semantic facts are immutable.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_length = int(inputs.shape[-1])
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        record.update(
            {
                "processor_class": type(processor).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "prompt_token_count": prompt_length,
                "prompt_sha256": _sha(rendered),
                "model_vocab_size": 262208,
                "constraint_vocab_size": 262145,
                "model_only_tail": {"start": 262145, "end": 262207, "count": 63},
            }
        )
        record["model_load_started"] = True
        ledger_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ROOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        if type(model).__name__ != "Gemma3ForConditionalGeneration":
            raise RuntimeError("EXT4G2_MODEL_CLASS_MISMATCH")
        if sum(parameter.numel() for parameter in model.parameters()) != 4300079472:
            raise RuntimeError("EXT4G2_PARAMETER_COUNT_MISMATCH")
        if sorted({str(parameter.device) for parameter in model.parameters()}) != ["cpu"]:
            raise RuntimeError("EXT4G2_NON_CPU_PLACEMENT")
        record["model_load_completed"] = True
        record["phase"] = "MODEL_LOAD_PASS"
        constraint = build_llguidance_logits_processor(
            tokenizer,
            schema=schema,
            expected_schema_sha256=SCHEMA_SHA,
            prompt_length=prompt_length,
            model_vocab_size=262208,
        )
        assert_generation_constraint(constraint, expected_schema_sha256=SCHEMA_SHA)
        probe = constraint.logits_processor(inputs, torch.zeros((1, 262208)))
        if tuple(probe.shape) != (1, 262208) or not torch.isinf(probe[0, 262145:]).all():
            raise RuntimeError("EXT4G2_STRUCTURED_PREFLIGHT_FAILED")
        record.update(
            {
                "structured_decoding_preflight": "EXT4G2_STRUCTURED_DECODING_PREFLIGHT_PASS",
                "vocab_alignment": constraint.vocab_alignment,
                "phase": "STRUCTURED_PREFLIGHT_PASS",
                "generation_policy": GENERATION_POLICY,
            }
        )
        _write(attempt_dir / "rendered_prompt.txt", rendered)
        record["phase"] = "GENERATION_AUTHORIZED"
        record["request_attempted"] = True
        ledger_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        record["generation_started"] = True
        record["phase"] = "GENERATION_STARTED"
        ledger_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        torch.manual_seed(GENERATION_POLICY["seed"])
        started = time.perf_counter()
        output_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=GENERATION_POLICY["max_new_tokens"],
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[constraint.logits_processor],
        )
        record.update(
            {
                "generate_call_count": 1,
                "generation_completed": True,
                "generation_duration_seconds": time.perf_counter() - started,
                "phase": "GENERATION_COMPLETED",
            }
        )
        continuation = output_ids[0][prompt_length:]
        record["generated_tokens"] = int(continuation.shape[0])
        raw_text = tokenizer.decode(continuation, skip_special_tokens=True)
        _write(attempt_dir / "raw_generated_continuation.txt", raw_text)
        candidate = json.loads(raw_text)
        record["json_parse"] = "PASS"
        response = validate_ext4f_realization_response(candidate, request)
        _write(
            attempt_dir / "parsed_realization_response.json",
            response.model_dump_json(indent=2) + "\n",
        )
        record.update(
            {
                "realization_validation": "PASS",
                "plan_binding": "PASS",
                "request_binding": "PASS",
                "slot_integrity": "PASS",
                "authority_mutations": 0,
                "technical_faithfulness_probes": "PASS; free-text faithfulness reserved for later evaluation",
                "phase": "VALIDATION_COMPLETED",
                "terminal_status": "EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_SMOKE_PASS",
            }
        )
    except json.JSONDecodeError as exc:
        record.update(
            {
                "json_parse": "FAIL",
                "terminal_status": "EXT4G2_GEMMA3_SYNTHETIC_REALIZATION_PARSE_FAILED",
                "failure": str(exc),
            }
        )
    except Candidate3StructuredOutputError as exc:
        record.update(
            {
                "terminal_status": "EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_GENERATION_FAILED",
                "failure": str(exc),
            }
        )
    except Exception as exc:
        record.update(
            {
                "terminal_status": "EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_GENERATION_FAILED"
                if record["generation_started"]
                else "EXT4G2_PREFLIGHT_OR_LOAD_FAILED",
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        record["cleanup_status"] = "PASS"
        del model, processor, constraint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if record["terminal_status"] == "EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_SMOKE_PASS":
            record["phase"] = "TERMINAL"
        record["terminal_time"] = datetime.now(UTC).isoformat()
        ledger_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        REPORT_PATH.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return 0 if record["terminal_status"] == "EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
