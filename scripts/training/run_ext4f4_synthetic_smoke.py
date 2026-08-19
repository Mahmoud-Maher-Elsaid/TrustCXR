"""Run exactly one EXT-4F.4 realization-only technical smoke.

This runner is deliberately separate from all EXT-4E benchmark runners.  It
uses one newly authored synthetic evidence envelope, compiles a wording-only
realization request, and performs one constrained generation only when the
production preflight is complete.
"""

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

import psutil
import torch
import transformers
from transformers import AutoTokenizer

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
MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"
CONFIG_PATH = ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4f4"
REPORT_PATH = (
    ROOT / "reports/research_extensions/ext4f/EXT4F4_SYNTHETIC_LLM_TECHNICAL_SMOKE_REPORT.json"
)
EXPECTED_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
EXPECTED_REALIZATION_SCHEMA_SHA = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
EXPECTED_TORCH = "2.12.1+cu130"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_LLGUIDANCE = "1.8.0"
SEED = 20260819


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rss() -> int:
    return psutil.Process().memory_info().rss


def _gpu_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(0),
    }


def _pip_check() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _runtime_preflight() -> dict[str, Any]:
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llguidance": importlib.metadata.version("llguidance"),
    }
    pip_ok, pip_output = _pip_check()
    if versions["torch"] != EXPECTED_TORCH:
        raise RuntimeError("EXT4F4_TORCH_VERSION_MISMATCH")
    if versions["transformers"] != EXPECTED_TRANSFORMERS:
        raise RuntimeError("EXT4F4_TRANSFORMERS_VERSION_MISMATCH")
    if versions["llguidance"] != EXPECTED_LLGUIDANCE:
        raise RuntimeError("EXT4F4_LLGUIDANCE_VERSION_MISMATCH")
    if not pip_ok:
        raise RuntimeError(f"EXT4F4_PIP_CHECK_FAILED:{pip_output}")
    return {**versions, "pip_check": "PASS", "pip_check_output": pip_output}


def _synthetic_evidence():
    # New EXT-4F-only case identity; the underlying object is still governed
    # EvidenceEnvelope data and is never loaded from a benchmark partition.
    return build_synthetic_case("uncertainty").model_copy(
        update={"case_reference": "research_case_ext4f4_realization_001"}
    )


def _authority_preflight(plan, request, schema_sha: str) -> None:
    validate_ext4f_semantic_plan(plan)
    validate_ext4f_realization_request(request)
    if schema_sha != EXPECTED_REALIZATION_SCHEMA_SHA:
        raise RuntimeError("EXT4F4_REALIZATION_SCHEMA_IDENTITY_MISMATCH")
    if set(LLM_REALIZATION_ONLY_FIELDS) != {"slot_text"}:
        raise RuntimeError("EXT4F4_AUTHORITY_PREFLIGHT_FAILED")
    if any(field in request.model_dump(mode="json") for field in LLM_REALIZATION_ONLY_FIELDS):
        raise RuntimeError("EXT4F4_AUTHORITY_PREFLIGHT_FAILED")


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt_id = f"ext4f4_realization_{run_id}"
    attempt_dir = ARTIFACT_ROOT / run_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = attempt_dir / "run_ledger.json"
    record: dict[str, Any] = {
        "stage": "EXT-4F.4",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "synthetic_case_id": "research_case_ext4f4_realization_001",
        "model": "microsoft/Phi-4-mini-instruct",
        "model_revision": EXPECTED_REVISION,
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
        "terminal_status": "EXT4F4_NOT_STARTED",
    }
    model = tokenizer = constraint = None
    try:
        runtime = _runtime_preflight()
        record["runtime"] = runtime
        evidence = _synthetic_evidence()
        plan = build_ext4f_semantic_plan(evidence)
        request = build_ext4f_realization_request(plan)
        schema = realization_schema()
        schema_sha = realization_schema_sha256()
        _authority_preflight(plan, request, schema_sha)
        prompt_text = compile_ext4f_realization_prompt(request)
        record.update(
            {
                "semantic_plan_sha256": plan.semantic_plan_sha256,
                "realization_request_sha256": request.realization_request_sha256,
                "realization_schema_sha256": schema_sha,
                "authority_preflight": "PASS",
                "structured_decoding_preflight": "PENDING_TOKENIZER",
            }
        )
        (attempt_dir / "synthetic_evidence.json").write_text(
            evidence.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (attempt_dir / "semantic_plan.json").write_text(
            plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (attempt_dir / "realization_request.json").write_text(
            request.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        record["prompt_token_count"] = None
        record["model_load_started"] = True
        ledger_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        from run_ext4e_candidate3_load_only import load_model_only, validate_identity

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        identity = validate_identity(config)
        if config["resolved_revision"] != EXPECTED_REVISION:
            raise RuntimeError("EXT4F4_MODEL_IDENTITY_MISMATCH")
        model, load_info = load_model_only()
        record.update({"model_load_completed": True, "model_load": load_info, "identity": identity})
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a wording-only realization component for a fictional, non-medical "
                    "research inspection record. Semantic facts, identifiers, and states are "
                    "immutable."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        prompt_length = int(inputs["input_ids"].shape[1])
        record["prompt_token_count"] = prompt_length
        constraint = build_llguidance_logits_processor(
            tokenizer,
            schema=schema,
            expected_schema_sha256=EXPECTED_REALIZATION_SCHEMA_SHA,
            prompt_length=prompt_length,
            model_vocab_size=int(model.config.vocab_size),
        )
        assert_generation_constraint(constraint)
        alignment = constraint.vocab_alignment
        if alignment.get("unregistered_model_tail_count") != 35:
            raise RuntimeError("EXT4F4_VOCAB_ALIGNMENT_FAILED")
        record["structured_decoding_preflight"] = "EXT4F4_STRUCTURED_DECODING_PREFLIGHT_PASS"
        (attempt_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        record["generation_parameters"] = {
            "max_new_tokens": 512,
            "do_sample": False,
            "top_p": 1.0,
            "seed": SEED,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature": None,
        }
        record["request_attempted"] = True
        record["generation_started"] = True
        record["generation_start_time"] = datetime.now(UTC).isoformat()
        ledger_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        torch.manual_seed(SEED)
        started = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[constraint.logits_processor],
        )
        record["generate_call_count"] = 1
        record["generation_completed"] = True
        record["generation_duration_seconds"] = time.perf_counter() - started
        continuation = output_ids[0][prompt_length:]
        record["generated_tokens"] = int(continuation.shape[0])
        raw_text = tokenizer.decode(continuation, skip_special_tokens=True)
        (attempt_dir / "raw_generated_continuation.txt").write_text(raw_text, encoding="utf-8")
        candidate = json.loads(raw_text)
        record["json_parse"] = "PASS"
        response = validate_ext4f_realization_response(candidate, request)
        record.update(
            {
                "realization_validation": "PASS",
                "plan_identity_binding": "PASS",
                "request_identity_binding": "PASS",
                "slot_integrity": "PASS",
                "authority_mutations": 0,
                "technical_faithfulness_probes": (
                    "PASS; free-text faithfulness reserved for later evaluation"
                ),
                "terminal_status": "EXT4F4_SYNTHETIC_LLM_TECHNICAL_SMOKE_PASS",
            }
        )
        (attempt_dir / "parsed_realization_response.json").write_text(
            response.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except json.JSONDecodeError as exc:
        record.update(
            {
                "json_parse": "FAIL",
                "realization_validation": "NOT_RUN",
                "terminal_status": "EXT4F4_SYNTHETIC_REALIZATION_PARSE_FAILED",
                "failure": f"JSONDecodeError: {exc}",
            }
        )
    except Candidate3StructuredOutputError as exc:
        record.update(
            {
                "terminal_status": "EXT4F4_SYNTHETIC_TECHNICAL_GENERATION_FAILED",
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    except Exception as exc:
        record.update(
            {
                "terminal_status": (
                    "EXT4F4_SYNTHETIC_TECHNICAL_GENERATION_FAILED"
                    if record["generation_started"]
                    else "EXT4F4_PREFLIGHT_OR_MODEL_LOAD_FAILED"
                ),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        record["cleanup_status"] = "PASS"
        del model, tokenizer, constraint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        record["real_model_inference_requests"] = record["generate_call_count"]
        record["terminal_time"] = datetime.now(UTC).isoformat()
        ledger_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        REPORT_PATH.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return 0 if record["terminal_status"] == "EXT4F4_SYNTHETIC_LLM_TECHNICAL_SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
