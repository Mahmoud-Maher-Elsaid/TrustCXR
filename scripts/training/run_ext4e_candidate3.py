"""Final governed Candidate #3 technical smoke and six-case evaluator."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trustcxr.grounded_llm.benchmark import score_case  # noqa: E402
from trustcxr.grounded_llm.candidate3_constrained_decoding import (  # noqa: E402
    GOVERNED_SCHEMA_SHA256,
    assert_generation_constraint,
    build_candidate3_logits_processor,
)
from trustcxr.grounded_llm.contracts import (  # noqa: E402
    GroundedOutputEnvelope,
    build_synthetic_case,
)
from trustcxr.grounded_llm.development_evaluation import (  # noqa: E402
    aggregate_evidence,
    scoring_case,
)

MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"
CASES_PATH = ROOT / "tests/fixtures/ext4d_benchmark_cases.json"
PROMPT_PATH = ROOT / "configs/research_extensions/ext4e2d_candidate1_prompt.txt"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4e_candidate3/development_evaluation"
REPORT_PATH = ROOT / "reports/research_extensions/ext4e/EXT4E_FINAL_CANDIDATE_SELECTION_REPORT.json"
CASE_IDS = (
    "dev_supported",
    "dev_uncertainty",
    "dev_defer",
    "dev_withheld",
    "dev_missing",
    "dev_conflict",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def schema() -> dict[str, Any]:
    value = GroundedOutputEnvelope.model_json_schema()
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != GOVERNED_SCHEMA_SHA256:
        raise RuntimeError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    return value


def load_model() -> tuple[Any, Any]:
    import importlib.metadata

    if tuple(sys.version_info[:2]) != (3, 12):
        raise RuntimeError("CANDIDATE3_PYTHON_VERSION_MISMATCH")
    if importlib.metadata.version("llguidance") != "1.8.0":
        raise RuntimeError("CANDIDATE3_LLGUIDANCE_VERSION_MISMATCH")
    if importlib.metadata.version("transformers") != "4.57.6":
        raise RuntimeError("CANDIDATE3_TRANSFORMERS_VERSION_MISMATCH")
    if importlib.metadata.version("torch") != "2.12.1+cu130":
        raise RuntimeError("CANDIDATE3_TORCH_VERSION_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    devices = {str(parameter.device) for parameter in model.parameters()}
    devices.update(str(buffer.device) for buffer in model.buffers())
    if devices != {"cpu"}:
        raise RuntimeError("CANDIDATE3_CPU_PLACEMENT_FAILED")
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    run_dir: Path,
    label: str,
    schema_value: dict[str, Any],
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    constraint = build_candidate3_logits_processor(
        tokenizer, schema=schema_value, prompt_length=int(inputs["input_ids"].shape[1])
    )
    assert_generation_constraint(constraint)
    write_json(
        run_dir / "request.json",
        {
            "messages": messages,
            "generation": {
                "max_new_tokens": 768,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "seed": 20260806,
            },
        },
    )
    input_len = int(inputs["input_ids"].shape[1])
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
    continuation = output_ids[0][input_len:]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    (run_dir / "raw_model_content.txt").write_text(text, encoding="utf-8")
    return {
        "text": text,
        "generated_tokens": int(continuation.shape[0]),
        "duration_seconds": time.perf_counter() - started,
    }


def main() -> int:
    schema_value = schema()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_root = ARTIFACT_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    ledger = {
        "run_id": run_id,
        "candidate_id": 3,
        "backend": "llguidance",
        "backend_version": "1.8.0",
        "schema_sha256": GOVERNED_SCHEMA_SHA256,
        "case_order": list(CASE_IDS),
        "generate_call_count": 0,
        "development_cases_accessed": 0,
        "development_requests_attempted": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "cases": {
            case_id: {"case_id": case_id, "order": i, "request_count": 0, "terminal_state": False}
            for i, case_id in enumerate(CASE_IDS)
        },
    }
    write_json(run_root / "run_ledger.json", ledger)
    model = tokenizer = None
    try:
        model, tokenizer = load_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are recording a fictional object-inspection research record. "
                    "This is not medical or benchmark data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return one JSON object conforming to the supplied schema with no "
                    "available evidence and a DEFER decision."
                ),
            },
        ]
        smoke_dir = run_root / "synthetic_smoke"
        smoke_dir.mkdir()
        smoke = generate_one(model, tokenizer, messages, smoke_dir, "synthetic", schema_value)
        ledger["generate_call_count"] = 1
        candidate = json.loads(smoke["text"])
        GroundedOutputEnvelope.model_validate(candidate)
        write_json(
            smoke_dir / "run_metadata.json", {**smoke, "json_parse": "PASS", "ext4c": "PASS"}
        )

        cases_payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cases = tuple(cases_payload.get("development_cases", ()))
        if (
            tuple(c.get("case_id") for c in cases) != CASE_IDS
            or len(cases_payload.get("final_cases", ())) != 24
            or cases_payload.get("locked_test_data") is not False
        ):
            raise RuntimeError("CANDIDATE3_DEVELOPMENT_PARTITION_INVALID")
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        evidence = {}
        for case in cases:
            case_id = case["case_id"]
            case_dir = run_root / case_id
            case_dir.mkdir()
            envelope = build_synthetic_case(case["grounding_kind"])
            write_json(case_dir / "input_envelope.json", envelope.model_dump(mode="json"))
            ledger["cases"][case_id].update(
                {
                    "request_count": 1,
                    "request_attempted": True,
                    "request_started_at": datetime.now(UTC).isoformat(),
                }
            )
            ledger["development_cases_accessed"] += 1
            ledger["development_requests_attempted"] += 1
            write_json(run_root / "run_ledger.json", ledger)
            result = generate_one(
                model,
                tokenizer,
                [{"role": "user", "content": prompt + "\n\n" + envelope.model_dump_json()}],
                case_dir,
                case_id,
                schema_value,
            )
            (case_dir / "raw_http_response.json").write_text(
                json.dumps({"choices": [{"message": {"content": result["text"]}}]}) + "\n",
                encoding="utf-8",
            )
            write_json(case_dir / "ext4c_output_schema.json", schema_value)
            ledger["generate_call_count"] += 1
            ledger["cases"][case_id].update(
                {
                    "generation_started": True,
                    "generation_completed": True,
                    "generated_tokens": result["generated_tokens"],
                    "terminal_state": True,
                    "evidence_path": str(case_dir),
                }
            )
            try:
                candidate = json.loads(result["text"])
                write_json(case_dir / "parsed_output.json", candidate)
                ledger["cases"][case_id]["parse_valid"] = True
                validated = GroundedOutputEnvelope.model_validate(candidate)
                score = score_case(scoring_case(case), validated.model_dump(mode="json"))
                write_json(case_dir / "score.json", score)
                ledger["cases"][case_id].update(
                    {
                        "ext4c_valid": True,
                        "scorer_executed": True,
                        "case_pass": score["case_passed"],
                    }
                )
            except Exception as exc:
                write_json(
                    case_dir / "validation_error.json",
                    {"classification": "EXT4C_SEMANTIC_VALIDATION_FAIL", "error": str(exc)},
                )
                ledger["cases"][case_id].update(
                    {
                        "parse_valid": True,
                        "ext4c_valid": False,
                        "scorer_executed": False,
                        "case_pass": False,
                        "failure_classification": "EXT4C_SEMANTIC_VALIDATION_FAIL",
                    }
                )
            write_json(
                case_dir / "run_metadata.json",
                {
                    "case_id": case_id,
                    "partition": "development",
                    "inference_request_count": 1,
                    "retry_count": 0,
                    "generation_started": True,
                    "generation_completed": True,
                    "response_parse_valid": ledger["cases"][case_id].get("parse_valid", False),
                    "attempt_state": "ATTEMPTED_COMPLETE",
                    "ext4c_valid": ledger["cases"][case_id].get("ext4c_valid"),
                    "scorer_executed": ledger["cases"][case_id].get("scorer_executed", False),
                    "case_passed": ledger["cases"][case_id].get("case_pass", False),
                    "development_cases_accessed": 1,
                    "frozen_final_cases_accessed": 0,
                    "locked_test_accessed": False,
                },
            )
            write_json(run_root / "run_ledger.json", ledger)
            evidence[case_id] = case_dir
        aggregate = aggregate_evidence(CASES_PATH, evidence)
        summary = {
            **aggregate,
            "development_cases_accessed": 6,
            "development_requests_attempted": 6,
            "final_cases_accessed": 0,
            "locked_test_accessed": False,
            "scientific_decision": "DEVELOPMENT_GATE_PASSED / SCIENTIFICALLY_SELECTED"
            if aggregate["benchmark_pass"]
            else "DEVELOPMENT_GATE_FAILED / NOT_SCIENTIFICALLY_SELECTED",
        }
        write_json(run_root / "development_summary.json", summary)
        write_json(
            REPORT_PATH,
            {
                "candidate3": {
                    "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
                    "backend": "llguidance",
                    "schema_sha256": GOVERNED_SCHEMA_SHA256,
                },
                "run_id": run_id,
                "aggregate": summary,
                "final_cases_accessed": 0,
                "locked_test_accessed": False,
                "final_disposition": "CANDIDATE_SELECTED_PENDING_FROZEN_FINAL_GOVERNANCE"
                if aggregate["benchmark_pass"]
                else "RETAIN_DETERMINISTIC_REPORTING",
            },
        )
        print(
            json.dumps(
                {
                    "EXT4E_RUN_ID": run_id,
                    "CANDIDATE3_PREFLIGHT": "PASS",
                    "MODEL_LOAD": "PASS",
                    "STRUCTURED_BACKEND": "llguidance==1.8.0",
                    "SYNTHETIC_GENERATION": "PASS",
                    "DEVELOPMENT_CASES_EXECUTED": 6,
                    "COMPLETED_GENERATIONS": summary.get("completed_generations"),
                    "PARSE_VALID": summary.get("parse_valid_count"),
                    "EXT4C_VALID": summary.get("ext4c_valid_count"),
                    "EXT4C_INVALID": summary.get("ext4c_invalid_count"),
                    "CASE_PASS": summary.get("case_pass_count"),
                    "CASE_FAIL": summary.get("case_fail_count"),
                    "SCIENTIFIC_DECISION": summary["scientific_decision"],
                    "FINAL_CASES_ACCESSED": 0,
                    "LOCKED_TEST_ACCESSED": False,
                    "REPORT_PATH": str(REPORT_PATH),
                },
                indent=2,
            )
        )
        return 0
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
