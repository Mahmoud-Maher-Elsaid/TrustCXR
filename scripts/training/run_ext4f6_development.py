"""Execute the frozen EXT-4F.6 development evaluation.

This runner is intentionally not invoked by repository tests.  It is the
single governed path for the later local 24-case run: one model load and at
most one constrained generation request per frozen case, with no retry.
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
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trustcxr.grounded_llm.candidate3_constrained_decoding import (  # noqa: E402
    assert_generation_constraint,
    build_llguidance_logits_processor,
)
from trustcxr.grounded_llm.ext4f5_benchmark import (  # noqa: E402
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    CASE_IDS,
    GENERATION_POLICY,
    GENERATION_POLICY_VERSION,
    build_development_cases,
    score_mock_realization,
    validate_benchmark_manifest,
)
from trustcxr.grounded_llm.ext4f_realization import (  # noqa: E402
    compile_ext4f_realization_prompt,
    realization_schema,
    realization_schema_sha256,
    validate_ext4f_realization_response,
)

EXPECTED_BENCHMARK_SHA = "671a04d2d859f1b1ffb9414a8c0f636596949748a00548e45abcbbfdb752db61"
EXPECTED_SCHEMA_SHA = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
EXPECTED_MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
EXPECTED_LLGUIDANCE = "1.8.0"
MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"
MANIFEST_PATH = ROOT / "configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/research_extensions/ext4f6"
REPORT_PATH = (
    ROOT / "reports/research_extensions/ext4f/EXT4F6_DEVELOPMENT_CANDIDATE_EVALUATION_REPORT.json"
)
CONFIG_PATH = ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pip_check() -> str:
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"EXT4F6_PIP_CHECK_FAILED:{result.stdout}{result.stderr}")
    return (result.stdout + result.stderr).strip()


def _preflight() -> dict[str, Any]:
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llguidance": importlib.metadata.version("llguidance"),
    }
    if (
        versions["python"] != "3.12.10"
        or versions["torch"] != "2.12.1+cu130"
        or versions["transformers"] != "4.57.6"
        or versions["llguidance"] != EXPECTED_LLGUIDANCE
    ):
        raise RuntimeError(f"EXT4F6_RUNTIME_MISMATCH:{versions}")
    return {**versions, "pip_check": "PASS", "pip_check_output": _pip_check()}


def _verify_manifest() -> tuple[dict[str, Any], tuple[Any, ...]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_benchmark_manifest(manifest)
    if manifest["benchmark_sha256"] != EXPECTED_BENCHMARK_SHA:
        raise RuntimeError("EXT4F6_BENCHMARK_SHA_MISMATCH")
    cases = build_development_cases()
    if tuple(case.case_id for case in cases) != CASE_IDS:
        raise RuntimeError("EXT4F6_CASE_ORDER_MISMATCH")
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    for case in cases:
        frozen = by_id[case.case_id]
        if (
            case.case_sha256 != frozen["case_sha256"]
            or case.plan.semantic_plan_sha256 != frozen["semantic_plan_sha256"]
            or case.request.realization_request_sha256 != frozen["realization_request_sha256"]
        ):
            raise RuntimeError(f"EXT4F6_CASE_IDENTITY_MISMATCH:{case.case_id}")
    return manifest, cases


def _ledger(
    run_id: str, manifest: dict[str, Any], cases: tuple[Any, ...], runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "stage": "EXT-4F.6",
        "run_id": run_id,
        "candidate_id": "EXT4F_CANDIDATE_PHI4_MINI_V1",
        "model": "microsoft/Phi-4-mini-instruct",
        "model_revision": EXPECTED_MODEL_REVISION,
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_sha256": manifest["benchmark_sha256"],
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "generation_policy": GENERATION_POLICY,
        "runtime": runtime,
        "semantic_contract": "EXT4F_SEMANTIC_GENERATION_CONTRACT_V1",
        "realization_contract": "EXT4F_REALIZATION_CONTRACT_V1",
        "realization_schema_sha256": EXPECTED_SCHEMA_SHA,
        "case_order": list(CASE_IDS),
        "cases": {
            case.case_id: {
                "case_id": case.case_id,
                "case_sha256": case.case_sha256,
                "order": index,
                "semantic_plan_sha256": case.plan.semantic_plan_sha256,
                "realization_request_sha256": case.request.realization_request_sha256,
                "risk_tags": list(case.risk_tags),
                "expected_slot_ids": [slot.slot_id for slot in case.request.slots],
                "request_attempted": False,
                "request_count": 0,
                "generation_started": False,
                "generation_completed": False,
                "generated_tokens": 0,
                "json_parse_status": "NOT_RUN",
                "automatic_hard_gate_status": "NOT_RUN",
                "semantic_review_status": "NOT_RUN",
                "case_pass": None,
                "terminal_state": "PREALLOCATED",
            }
            for index, case in enumerate(cases, 1)
        },
        "generate_call_count": 0,
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "protocol_deviation_count": 0,
        "terminal_status": "PREPARED",
        "cleanup_status": "NOT_RUN",
    }


def _aggregate(ledger: dict[str, Any]) -> dict[str, Any]:
    entries = list(ledger["cases"].values())
    completed = [entry for entry in entries if entry["generation_completed"]]
    return {
        "benchmark_cases": 24,
        "cases_attempted": sum(entry["request_attempted"] for entry in entries),
        "generate_call_count": ledger["generate_call_count"],
        "generation_completed_count": len(completed),
        "generation_failed_count": sum(
            entry["request_attempted"] and not entry["generation_completed"] for entry in entries
        ),
        "generated_token_total": sum(entry["generated_tokens"] for entry in entries),
        "json_valid_count": sum(entry["json_parse_status"] == "PASS" for entry in entries),
        "json_invalid_count": sum(entry["json_parse_status"] == "FAIL" for entry in entries),
        "realization_contract_valid_count": sum(
            entry.get("realization_contract_status") == "PASS" for entry in entries
        ),
        "realization_contract_invalid_count": sum(
            entry.get("realization_contract_status") == "FAIL" for entry in entries
        ),
        "plan_binding_pass_count": sum(
            entry.get("plan_binding_status") == "PASS" for entry in entries
        ),
        "request_binding_pass_count": sum(
            entry.get("request_binding_status") == "PASS" for entry in entries
        ),
        "slot_integrity_pass_count": sum(
            entry.get("slot_integrity_status") == "PASS" for entry in entries
        ),
        "authority_preservation_count": sum(
            entry.get("authority_mutations") == 0 and entry.get("authority_mutations") is not None
            for entry in entries
        ),
        "authority_mutation_count": sum(entry.get("authority_mutations", 0) for entry in entries),
        "automatic_hard_gate_pass_count": sum(
            entry.get("automatic_hard_gate_status") == "PASS" for entry in entries
        ),
        "automatic_hard_gate_fail_count": sum(
            entry.get("automatic_hard_gate_status") == "FAIL" for entry in entries
        ),
        "semantic_review_required_case_count": sum(
            entry.get("semantic_review_status") == "REVIEW_REQUIRED" for entry in entries
        ),
        "semantic_review_required_slot_count": sum(
            entry.get("semantic_review_required_slots", 0) for entry in entries
        ),
        "protocol_deviation_count": ledger["protocol_deviation_count"],
        "structured_output_validity_rate": None,
        "realization_contract_validity_rate": None,
        "authority_preservation_rate": None,
        "hard_safety_gate_pass": None,
        "faithfulness_pass_rate": None,
        "case_pass_rate": None,
    }


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "run_ledger.json"
    model = tokenizer = constraint = None
    ledger: dict[str, Any] = {}
    try:
        manifest, cases = _verify_manifest()
        runtime = _preflight()
        ledger = _ledger(run_id, manifest, cases, runtime)
        _write(ledger_path, ledger)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ROOT, local_files_only=True, trust_remote_code=False
        )
        schema = realization_schema()
        if realization_schema_sha256() != EXPECTED_SCHEMA_SHA:
            raise RuntimeError("EXT4F6_SCHEMA_SHA_MISMATCH")
        # Importing the audited load helper is deferred until the preallocated ledger exists.
        from run_ext4e_candidate3_load_only import load_model_only, validate_identity

        validate_identity(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        model, load_info = load_model_only()
        ledger["model_load"] = load_info
        first_prompt = compile_ext4f_realization_prompt(cases[0].request)
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a wording-only realization component."},
                {"role": "user", "content": first_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = tokenizer(rendered, return_tensors="pt")
        constraint = build_llguidance_logits_processor(
            tokenizer,
            schema=schema,
            expected_schema_sha256=EXPECTED_SCHEMA_SHA,
            prompt_length=int(prompt_inputs["input_ids"].shape[1]),
            model_vocab_size=int(model.config.vocab_size),
        )
        assert_generation_constraint(constraint, expected_schema_sha256=EXPECTED_SCHEMA_SHA)
        ledger["structured_decoding_preflight"] = "EXT4F6_STRUCTURED_DECODING_PREFLIGHT_PASS"
        for index, case in enumerate(cases, 1):
            print(f"[{index:02d}/24] {case.case_id}", flush=True)
            entry = ledger["cases"][case.case_id]
            prompt_text = compile_ext4f_realization_prompt(case.request)
            messages = [
                {"role": "system", "content": "You are a wording-only realization component."},
                {"role": "user", "content": prompt_text},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt")
            prompt_length = int(inputs["input_ids"].shape[1])
            constraint = build_llguidance_logits_processor(
                tokenizer,
                schema=schema,
                expected_schema_sha256=EXPECTED_SCHEMA_SHA,
                prompt_length=prompt_length,
                model_vocab_size=int(model.config.vocab_size),
            )
            assert_generation_constraint(constraint, expected_schema_sha256=EXPECTED_SCHEMA_SHA)
            entry.update(
                {
                    "request_attempted": True,
                    "request_count": 1,
                    "generation_started": True,
                    "prompt_sha256": _sha(rendered),
                    "prompt_token_count": prompt_length,
                    "terminal_state": "GENERATION_STARTED",
                }
            )
            _write(ledger_path, ledger)
            torch.manual_seed(GENERATION_POLICY["seed"])
            started = time.perf_counter()
            ledger["generate_call_count"] += 1
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_POLICY["max_new_tokens"],
                    do_sample=False,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=[constraint.logits_processor],
                )
            except Exception as exc:
                entry.update(
                    {
                        "generation_completed": False,
                        "generation_duration_seconds": time.perf_counter() - started,
                        "automatic_hard_gate_status": "FAIL",
                        "failure_classification": f"{type(exc).__name__}: {exc}",
                        "terminal_state": "GENERATION_FAILED",
                    }
                )
                ledger["development_cases_accessed"] = index
                _write(ledger_path, ledger)
                continue
            continuation = output_ids[0][prompt_length:]
            raw_text = tokenizer.decode(continuation, skip_special_tokens=True)
            raw_path = run_dir / f"{case.case_id}_raw.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            entry.update(
                {
                    "generation_completed": True,
                    "generated_tokens": int(continuation.shape[0]),
                    "generation_duration_seconds": time.perf_counter() - started,
                    "raw_output_path": str(raw_path),
                    "terminal_state": "GENERATION_COMPLETED",
                }
            )
            ledger["development_cases_accessed"] = index
            try:
                candidate = json.loads(raw_text)
                entry["json_parse_status"] = "PASS"
                response = validate_ext4f_realization_response(candidate, case.request)
                entry.update(
                    {
                        "realization_contract_status": "PASS",
                        "plan_binding_status": "PASS",
                        "request_binding_status": "PASS",
                        "slot_integrity_status": "PASS",
                        "authority_mutations": 0,
                    }
                )
                scored = score_mock_realization(case, response)
                entry["automatic_hard_gate_status"] = scored["automatic_status"]
                entry["semantic_review_status"] = scored["semantic_adjudication"]
                entry["semantic_review_required_slots"] = len(
                    scored.get("review_package", {}).get("slots", [])
                )
                if scored.get("review_package"):
                    _write(run_dir / "review" / f"{case.case_id}.json", scored["review_package"])
            except json.JSONDecodeError as exc:
                entry.update(
                    {
                        "json_parse_status": "FAIL",
                        "automatic_hard_gate_status": "FAIL",
                        "failure_classification": str(exc),
                    }
                )
            except Exception as exc:
                entry.update(
                    {
                        "realization_contract_status": "FAIL",
                        "automatic_hard_gate_status": "FAIL",
                        "failure_classification": str(exc),
                    }
                )
            entry["terminal_state"] = "VALIDATION_COMPLETED"
            _write(ledger_path, ledger)
        ledger["terminal_status"] = "EXT4F6_AUTOMATIC_EVALUATION_COMPLETE_REVIEW_PENDING"
        ledger["development_cases_accessed"] = 24
        ledger["aggregate"] = _aggregate(ledger)
    except Exception as exc:
        ledger["terminal_status"] = "EXT4F6_PREPARATION_OR_RUNTIME_FAILED"
        ledger["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        ledger["cleanup_status"] = "PASS"
        ledger["final_cases_accessed"] = 0
        ledger["locked_test_accessed"] = False
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if constraint is not None:
            del constraint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ledger:
            _write(ledger_path, ledger)
            _write(
                run_dir / "review" / "index.json",
                {
                    "benchmark_sha256": ledger.get("benchmark_sha256"),
                    "candidate_identity": None,
                    "cases": list(ledger.get("cases", {})),
                },
            )
            _write(
                REPORT_PATH,
                {
                    "stage": "EXT-4F.6",
                    "status": ledger.get("terminal_status"),
                    "run_id": run_id,
                    "candidate_id": ledger.get("candidate_id"),
                    "benchmark_sha256": ledger.get("benchmark_sha256"),
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
        if ledger.get("terminal_status") == "EXT4F6_AUTOMATIC_EVALUATION_COMPLETE_REVIEW_PENDING"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
