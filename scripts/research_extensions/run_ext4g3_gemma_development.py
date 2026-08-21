"""Governed EXT-4G.3 Gemma development evaluation.

The benchmark is opened only after identity, runtime, attention-mask, and
structured-decoding gates pass. This runner performs one generation per frozen
case, never retries, and never accesses final or locked data.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trustcxr.grounded_llm.candidate3_constrained_decoding import (  # noqa: E402
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.contracts import build_synthetic_case  # noqa: E402
from trustcxr.grounded_llm.ext4f5_benchmark import (  # noqa: E402
    BENCHMARK_VERSION,
    CASE_IDS,
    GENERATION_POLICY,
    GENERATION_POLICY_VERSION,
    build_development_cases,
    score_mock_realization,
)
from trustcxr.grounded_llm.ext4f_contracts import build_ext4f_semantic_plan  # noqa: E402
from trustcxr.grounded_llm.ext4f_realization import (  # noqa: E402
    build_ext4f_realization_request,
    compile_ext4f_realization_prompt,
    realization_schema,
    realization_schema_sha256,
    validate_ext4f_realization_response,
)

EXPECTED_BENCHMARK_SHA = "671a04d2d859f1b1ffb9414a8c0f636596949748a00548e45abcbbfdb752db61"
EXPECTED_BENCHMARK_NAME = "EXT4F_DEVELOPMENT_BENCHMARK_V1"
EXPECTED_SCHEMA_SHA = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
EXPECTED_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
EXPECTED_LLGUIDANCE = "1.8.0"
MODEL_ROOT = ROOT / "cache/research_extensions/ext4g_candidate_gemma3_4b_it/models"
MANIFEST_PATH = ROOT / "configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json"
CANDIDATE_MANIFEST = ROOT / "configs/research_extensions/ext4g_gemma3_candidate_manifest.json"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4g3"
REPORT_PATH = (
    ROOT / "reports/research_extensions/ext4g/EXT4G3_GEMMA_DEVELOPMENT_EVALUATION_REPORT.json"
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pip_check() -> None:
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("EXT4G3_PIP_CHECK_FAILED")


def _runtime_preflight() -> dict[str, str]:
    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llguidance": importlib.metadata.version("llguidance"),
    }
    if runtime != {
        "python": "3.12.10",
        "torch": "2.12.1+cu130",
        "transformers": "4.57.6",
        "llguidance": EXPECTED_LLGUIDANCE,
    }:
        raise RuntimeError("EXT4G3_RUNTIME_PREFLIGHT_FAILED")
    _pip_check()
    return {**runtime, "pip_check": "PASS"}


def _verify_candidate_manifest() -> None:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if manifest["revision"] != EXPECTED_REVISION or len(manifest["files"]) != 15:
        raise RuntimeError("EXT4G3_CANDIDATE_MANIFEST_FAILED")
    for item in manifest["files"]:
        path = MODEL_ROOT / item["filename"]
        if not path.is_file() or path.stat().st_size != item["byte_size"]:
            raise RuntimeError("EXT4G3_CANDIDATE_FILE_FAILED")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError("EXT4G3_CANDIDATE_HASH_FAILED")


def _tokenize(processor, rendered):
    inputs = processor(text=rendered, return_tensors="pt", add_special_tokens=False)
    if "attention_mask" not in inputs:
        raise RuntimeError("EXT4G3_ATTENTION_MASK_MISSING")
    if tuple(inputs["attention_mask"].shape) != tuple(inputs["input_ids"].shape):
        raise RuntimeError("EXT4G3_ATTENTION_MASK_SHAPE_FAILED")
    return inputs


def _structured_preflight(processor, schema):
    evidence = build_synthetic_case("uncertainty").model_copy(
        update={"case_reference": "ext4g3_preflight_only"}
    )
    plan = build_ext4f_semantic_plan(evidence)
    request = build_ext4f_realization_request(plan)
    prompt = compile_ext4f_realization_prompt(request)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Wording-only realization."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenize(processor, rendered)
    constraint = build_llguidance_logits_processor(
        processor.tokenizer,
        schema=schema,
        expected_schema_sha256=EXPECTED_SCHEMA_SHA,
        prompt_length=int(inputs["input_ids"].shape[1]),
        model_vocab_size=262208,
    )
    assert_generation_constraint(constraint, expected_schema_sha256=EXPECTED_SCHEMA_SHA)
    probe = constraint.logits_processor(inputs["input_ids"], torch.zeros((1, 262208)))
    if tuple(probe.shape) != (1, 262208) or not torch.isinf(probe[0, 262145:]).all():
        raise RuntimeError("EXT4G3_STRUCTURED_PREFLIGHT_FAILED")
    return {
        "status": "PASS",
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "attention_mask_shape": list(inputs["attention_mask"].shape),
        "vocab_alignment": constraint.vocab_alignment,
    }


def _preallocate(cases, runtime):
    return {
        "stage": "EXT-4G.3",
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "repository": "google/gemma-3-4b-it",
        "revision": EXPECTED_REVISION,
        "benchmark_name": EXPECTED_BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "generation_policy": GENERATION_POLICY,
        "runtime": runtime,
        "structured_backend": "llguidance",
        "realization_schema_sha256": EXPECTED_SCHEMA_SHA,
        "case_order": list(CASE_IDS),
        "cases": {
            case.case_id: {
                "case_id": case.case_id,
                "case_sha256": case.case_sha256,
                "semantic_plan_sha256": case.plan.semantic_plan_sha256,
                "realization_request_sha256": case.request.realization_request_sha256,
                "request_count": 0,
                "attention_mask_present": False,
                "attention_mask_shape": None,
                "generate_call_count": 0,
                "generation_completed": False,
                "automatic_hard_gate_status": "NOT_RUN",
                "semantic_review_status": "NOT_RUN",
                "terminal_state": "PREALLOCATED",
            }
            for case in cases
        },
        "generate_call_count": 0,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "protocol_deviation_count": 0,
        "phase": "PREPARED",
        "terminal_status": "PREPARED",
    }


def _aggregate(ledger):
    entries = list(ledger["cases"].values())
    return {
        "benchmark_cases": 24,
        "cases_attempted": sum(entry.get("request_count", 0) == 1 for entry in entries),
        "generate_call_count": ledger["generate_call_count"],
        "generation_completed_count": sum(entry["generation_completed"] for entry in entries),
        "json_valid_count": sum(entry.get("json_parse_status") == "PASS" for entry in entries),
        "realization_contract_valid_count": sum(
            entry.get("realization_contract_status") == "PASS" for entry in entries
        ),
        "automatic_hard_gate_pass_count": sum(
            entry.get("automatic_hard_gate_status") == "PASS" for entry in entries
        ),
        "automatic_hard_gate_fail_count": sum(
            entry.get("automatic_hard_gate_status") == "FAIL" for entry in entries
        ),
        "authority_mutation_count": sum(entry.get("authority_mutations", 0) for entry in entries),
        "protocol_deviation_count": ledger["protocol_deviation_count"],
        "structured_validity_rate": None,
        "contract_validity_rate": None,
        "faithfulness_pass_rate": None,
        "case_pass_rate": None,
    }


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "run_ledger.json"
    model = processor = None
    ledger = {}
    try:
        runtime = _runtime_preflight()
        _verify_candidate_manifest()
        schema = realization_schema()
        if realization_schema_sha256() != EXPECTED_SCHEMA_SHA:
            raise RuntimeError("EXT4G3_SCHEMA_SHA_FAILED")
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        structured = _structured_preflight(processor, schema)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16
        )
        if sum(parameter.numel() for parameter in model.parameters()) != 4300079472:
            raise RuntimeError("EXT4G3_PARAMETER_COUNT_FAILED")
        if sorted({str(parameter.device) for parameter in model.parameters()}) != ["cpu"]:
            raise RuntimeError("EXT4G3_CPU_PLACEMENT_FAILED")
        # The checkpoint's top_k=64 lives only in GenerationConfig. It is inert
        # under do_sample=False and is not supplied by this runner.
        ledger = _preallocate(build_development_cases(), runtime)
        ledger.update(
            {
                "run_id": run_id,
                "structured_decoding_preflight": structured,
                "phase": "STRUCTURED_PREFLIGHT_PASS",
                "model_load_completed": True,
            }
        )
        _write(ledger_path, ledger)
        cases = build_development_cases()
        for index, case in enumerate(cases, 1):
            print(f"[{index:02d}/24] {case.case_id}", flush=True)
            entry = ledger["cases"][case.case_id]
            prompt = compile_ext4f_realization_prompt(case.request)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Wording-only realization."}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _tokenize(processor, rendered)
            constraint = build_llguidance_logits_processor(
                processor.tokenizer,
                schema=schema,
                expected_schema_sha256=EXPECTED_SCHEMA_SHA,
                prompt_length=int(inputs["input_ids"].shape[1]),
                model_vocab_size=262208,
            )
            assert_generation_constraint(constraint, expected_schema_sha256=EXPECTED_SCHEMA_SHA)
            entry.update(
                {
                    "request_count": 1,
                    "attention_mask_present": True,
                    "attention_mask_shape": list(inputs["attention_mask"].shape),
                    "prompt_token_count": int(inputs["input_ids"].shape[1]),
                    "prompt_sha256": _sha(rendered),
                    "generation_started": True,
                }
            )
            ledger["generate_call_count"] += 1
            _write(ledger_path, ledger)
            started = time.perf_counter()
            try:
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=GENERATION_POLICY["max_new_tokens"],
                    do_sample=False,
                    top_p=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    logits_processor=[constraint.logits_processor],
                )
                continuation = output[0][inputs["input_ids"].shape[1] :]
                raw = processor.tokenizer.decode(continuation, skip_special_tokens=True)
                raw_path = run_dir / f"{case.case_id}_raw.txt"
                raw_path.write_text(raw, encoding="utf-8")
                entry.update(
                    {
                        "generation_completed": True,
                        "generated_tokens": int(continuation.shape[0]),
                        "generation_duration_seconds": time.perf_counter() - started,
                        "raw_output_path": str(raw_path),
                    }
                )
                candidate = json.loads(raw)
                response = validate_ext4f_realization_response(candidate, case.request)
                scored = score_mock_realization(case, response)
                entry.update(
                    {
                        "json_parse_status": "PASS",
                        "realization_contract_status": "PASS",
                        "plan_binding_status": "PASS",
                        "request_binding_status": "PASS",
                        "slot_integrity_status": "PASS",
                        "authority_mutations": 0,
                        "automatic_hard_gate_status": scored["automatic_status"],
                        "semantic_review_status": scored["semantic_adjudication"],
                        "semantic_review_required_slots": len(
                            scored.get("review_package", {}).get("slots", [])
                        ),
                        "terminal_state": "VALIDATION_COMPLETED",
                    }
                )
                if scored.get("review_package"):
                    _write(run_dir / "review" / f"{case.case_id}.json", scored["review_package"])
            except Exception as exc:
                entry.update(
                    {
                        "generation_completed": False,
                        "automatic_hard_gate_status": "FAIL",
                        "failure_classification": f"{type(exc).__name__}: {exc}",
                        "terminal_state": "CASE_FAILED",
                    }
                )
            ledger["development_cases_accessed"] = index
            _write(ledger_path, ledger)
        ledger["aggregate"] = _aggregate(ledger)
        ledger["phase"] = "TERMINAL"
        ledger["terminal_status"] = "EXT4G3_AUTOMATIC_EVALUATION_COMPLETE_REVIEW_PENDING"
    except Exception as exc:
        ledger = ledger or {"run_id": run_id}
        ledger["terminal_status"] = "EXT4G3_PREPARATION_OR_PREFLIGHT_FAILED"
        ledger["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        ledger["cleanup_status"] = "PASS"
        ledger["frozen_final_cases_accessed"] = 0
        ledger["locked_test_accessed"] = False
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _write(ledger_path, ledger)
        _write(
            REPORT_PATH,
            {
                "stage": "EXT-4G.3",
                "status": ledger.get("terminal_status"),
                "run_id": run_id,
                "candidate_id": ledger.get("candidate_id"),
                "aggregate": ledger.get("aggregate"),
                "development_cases_accessed": ledger.get("development_cases_accessed", 0),
                "final_cases_accessed": 0,
                "locked_test_accessed": False,
                "cleanup_status": ledger.get("cleanup_status"),
            },
        )
    print(json.dumps(ledger.get("aggregate", ledger), indent=2, default=str))
    return (
        0
        if ledger.get("terminal_status") == "EXT4G3_AUTOMATIC_EVALUATION_COMPLETE_REVIEW_PENDING"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
