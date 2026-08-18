"""Exactly-one-request Candidate #3 non-medical constrained-generation smoke."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

from trustcxr.grounded_llm.candidate3_constrained_decoding import (
    GOVERNED_SCHEMA_SHA256,
    Candidate3StructuredOutputError,
    assert_generation_constraint,
    build_candidate3_prefix_allowed_tokens_fn,
)
from trustcxr.grounded_llm.contracts import GroundedOutputEnvelope

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4e_candidate3/synthetic_smoke"
REPORT_PATH = ROOT / "reports/research_extensions/ext4e/EXT4E_CANDIDATE3_SYNTHETIC_SMOKE.json"
EXPECTED_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"


def canonical_schema_sha(schema: dict) -> str:
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rss() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def gpu_memory() -> dict:
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


def main() -> int:
    from run_ext4e_candidate3_load_only import load_model_only, validate_identity

    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json").read_text()
    )
    identity = validate_identity(config)
    schema = GroundedOutputEnvelope.model_json_schema()
    schema_sha = canonical_schema_sha(schema)
    if schema_sha != GOVERNED_SCHEMA_SHA256:
        raise RuntimeError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    if config["resolved_revision"] != EXPECTED_REVISION:
        raise RuntimeError("CANDIDATE3_MODEL_IDENTITY_MISMATCH")
    attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt_dir = ARTIFACT_ROOT / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    messages = [
        {
            "role": "system",
            "content": (
                "You are recording a fictional object-inspection research record. "
                "This is not medical, clinical, radiology, or benchmark data."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly one JSON object conforming to the supplied schema. "
                "For this synthetic inspection record, report no available evidence, "
                "no claims, unavailable uncertainty, and a DEFER decision. "
                "Do not use markdown or commentary."
            ),
        },
    ]
    (attempt_dir / "input_messages.json").write_text(
        json.dumps(messages, indent=2) + "\n", encoding="utf-8"
    )
    record = {
        "synthetic_attempt_id": attempt_id,
        "request_attempted": False,
        "request_count": 0,
        "generation_started": False,
        "generation_completed": False,
        "generate_call_count": 0,
        "model_revision": EXPECTED_REVISION,
        "schema_sha256": schema_sha,
        "llguidance_version": identity["llguidance_version"],
        "generation_parameters": {
            "max_new_tokens": 768,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "seed": 20260806,
            "stream": False,
        },
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "real_model_inference_requests": 0,
    }
    model = None
    tokenizer = None
    try:
        cpu_before, gpu_before = rss(), gpu_memory()
        model, load_info = load_model_only()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        constraint = build_candidate3_prefix_allowed_tokens_fn(
            tokenizer, schema=schema, prompt_length=int(inputs["input_ids"].shape[1])
        )
        assert_generation_constraint(constraint)
        torch.manual_seed(20260806)
        (attempt_dir / "rendered_input.txt").write_text(rendered, encoding="utf-8")
        record["request_attempted"] = True
        record["request_count"] = 1
        record["real_model_inference_requests"] = 1
        (attempt_dir / "generation_attempt.json").write_text(
            json.dumps({**record, "generation_started": True}, indent=2) + "\n",
            encoding="utf-8",
        )
        record["generation_started"] = True
        record["start_time"] = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[constraint.logits_processor],
        )
        record["generate_call_count"] = 1
        record["generation_completed"] = True
        record["generation_duration_seconds"] = time.perf_counter() - started
        continuation = output_ids[0][inputs["input_ids"].shape[1] :]
        record["generated_tokens"] = int(continuation.shape[0])
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        (attempt_dir / "raw_generated_text.txt").write_text(text, encoding="utf-8")
        record["json_parse"] = "PASS"
        candidate = json.loads(text)
        validated = GroundedOutputEnvelope.model_validate(candidate)
        (attempt_dir / "parsed_output.json").write_text(
            validated.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        record["ext4c_validation"] = "PASS"
        record["technical_status"] = "CANDIDATE3_SYNTHETIC_STRUCTURED_GENERATION_PASS"
        record["cpu_rss_before_bytes"] = cpu_before
        record["cpu_rss_after_generation_bytes"] = rss()
        record["gpu_before"] = gpu_before
        record["gpu_after_generation"] = gpu_memory()
        record["load_info"] = load_info
    except Candidate3StructuredOutputError as exc:
        record.update(
            technical_status=(
                "CANDIDATE3_LLGUIDANCE_SCHEMA_INCOMPATIBLE"
                if "SEMANTICS_INCOMPATIBLE" in str(exc)
                else "CANDIDATE3_SYNTHETIC_GENERATION_FAILURE"
            ),
            validation_failure=f"{type(exc).__name__}: {exc}",
        )
    except json.JSONDecodeError as exc:
        record.update(
            json_parse="FAIL",
            ext4c_validation="NOT_RUN",
            technical_status="CANDIDATE3_SYNTHETIC_STRUCTURED_DECODING_FAILURE",
            validation_failure=f"JSONDecodeError: {exc}",
        )
    except Exception as exc:
        record.update(
            ext4c_validation="FAIL" if record["generation_completed"] else "NOT_RUN",
            technical_status=(
                "CANDIDATE3_SYNTHETIC_EXT4C_FAIL"
                if record["generation_completed"]
                else "CANDIDATE3_SYNTHETIC_GENERATION_FAILURE"
            ),
            validation_failure=f"{type(exc).__name__}: {exc}",
        )
    finally:
        record["cleanup_confirmed"] = True
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (attempt_dir / "run_metadata.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        REPORT_PATH.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return (
        0
        if record.get("technical_status") == "CANDIDATE3_SYNTHETIC_STRUCTURED_GENERATION_PASS"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
