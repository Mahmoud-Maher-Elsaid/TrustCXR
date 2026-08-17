"""Crash-safe, exactly-once Candidate #2 six-case development evaluator.

This module reuses the frozen EXT-4B/EXT-4C/EXT-4D semantics while keeping
Candidate #2 transport construction in the centralized request builder.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trustcxr.grounded_llm.benchmark import score_case  # noqa: E402
from trustcxr.grounded_llm.candidate2_request import (  # noqa: E402
    build_candidate2_request_payload,
)
from trustcxr.grounded_llm.contracts import (  # noqa: E402
    GroundedOutputEnvelope,
    build_synthetic_case,
)
from trustcxr.grounded_llm.development_evaluation import (  # noqa: E402
    aggregate_evidence,
    scoring_case,
)

CASES_PATH = ROOT / "tests/fixtures/ext4d_benchmark_cases.json"
PROMPT_PATH = ROOT / "configs/research_extensions/ext4e2d_candidate1_prompt.txt"
CASE_IDS = (
    "dev_supported",
    "dev_uncertainty",
    "dev_defer",
    "dev_withheld",
    "dev_missing",
    "dev_conflict",
)
TERMINAL_STATES = {"ATTEMPTED_COMPLETE", "ATTEMPTED_INCOMPLETE"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def resolve_development_cases(path: Path = CASES_PATH) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    development = tuple(payload.get("development_cases", ()))
    ids = tuple(case.get("case_id") for case in development)
    if ids != CASE_IDS:
        raise RuntimeError("CANDIDATE2_DEVELOPMENT_CASE_SET_INVALID")
    if len(payload.get("final_cases", ())) != 24 or payload.get("locked_test_data") is not False:
        raise RuntimeError("CANDIDATE2_PARTITION_CONTRACT_INVALID")
    return development


def _default_request(url: str, payload: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _safe_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    return json.loads(exc.json(include_url=False, include_context=False))


def _messages(prompt: str, envelope: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt + "\n\n" + envelope.model_dump_json()}]


def _initial_ledger(run_root: Path) -> dict[str, Any]:
    return {
        "schema_id": "EXT4E_CANDIDATE2_CASE_ATTEMPT_LEDGER",
        "schema_version": "1",
        "evaluation_run_id": run_root.name,
        "case_order": list(CASE_IDS),
        "cases": {
            case_id: {
                "case_id": case_id,
                "case_order": index,
                "partition": "development",
                "attempt_state": "NOT_ATTEMPTED",
                "request_attempted": False,
                "request_count": 0,
                "terminal_state": False,
            }
            for index, case_id in enumerate(CASE_IDS)
        },
        "development_cases_accessed": 0,
        "development_requests_attempted": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
    }


def scientific_decision(benchmark_pass: bool) -> str:
    """Apply the frozen binary development-gate decision labels."""

    return (
        "DEVELOPMENT_GATE_PASSED / SCIENTIFICALLY_SELECTED"
        if benchmark_pass
        else "DEVELOPMENT_GATE_FAILED / NOT_SCIENTIFICALLY_SELECTED"
    )


def run_development_evaluation(
    *,
    server_url: str,
    model: str,
    schema: dict[str, Any],
    prompt: str | None = None,
    run_root: Path | None = None,
    request_fn: Callable[[str, dict[str, Any]], bytes] | None = None,
    cases_path: Path = CASES_PATH,
) -> dict[str, Any]:
    """Run/resume the six-case evaluator; request_fn enables offline tests."""

    cases = resolve_development_cases(cases_path)
    if run_root is None:
        base = ROOT / "artifacts/research_extensions/ext4e_candidate2/development_evaluation"
        base.mkdir(parents=True, exist_ok=True)
        resumable = [
            path
            for path in base.iterdir()
            if path.is_dir() and (path / "case_attempt_ledger.json").is_file()
        ]
        run_root = (
            max(resumable, key=lambda path: path.name)
            if resumable
            else base / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    run_root.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / "case_attempt_ledger.json"
    ledger = _initial_ledger(run_root)
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _write_atomic(ledger_path, ledger)
    prompt = prompt if prompt is not None else PROMPT_PATH.read_text(encoding="utf-8")
    _write_json(
        run_root / "evaluation_plan.json", {"case_ids": list(CASE_IDS), "final_cases_accessed": 0}
    )
    _write_json(run_root / "ext4c_output_schema.json", schema)
    request_fn = request_fn or _default_request
    evidence_paths: dict[str, Path] = {}
    for case in cases:
        case_id = case["case_id"]
        entry = ledger["cases"][case_id]
        if entry.get("attempt_state") in TERMINAL_STATES:
            evidence_paths[case_id] = Path(entry["evidence_path"])
            continue
        case_root = run_root / case_id
        case_root.mkdir(parents=True, exist_ok=False)
        envelope = build_synthetic_case(case["grounding_kind"])
        input_payload = envelope.model_dump(mode="json")
        _write_json(case_root / "input_envelope.json", input_payload)
        _write_json(case_root / "ext4c_output_schema.json", schema)
        payload = build_candidate2_request_payload(model, _messages(prompt, envelope), schema)
        _write_json(case_root / "request.json", payload)
        now = datetime.now(UTC).isoformat()
        entry.update(
            {
                "attempt_state": "REQUEST_INTENT_RECORDED",
                "request_attempted": True,
                "request_count": 1,
                "request_started_at": now,
                "raw_request_path": str(case_root / "request.json"),
            }
        )
        ledger["development_cases_accessed"] = sum(
            item.get("request_attempted", False) for item in ledger["cases"].values()
        )
        ledger["development_requests_attempted"] = ledger["development_cases_accessed"]
        _write_atomic(ledger_path, ledger)
        try:
            raw = request_fn(server_url.rstrip("/") + "/v1/chat/completions", payload)
            entry.update(
                {
                    "attempt_state": "REQUEST_SENT",
                    "http_status": 200,
                    "generation_started": True,
                    "generation_completed": True,
                }
            )
            (case_root / "raw_http_response.json").write_bytes(raw)
            response = json.loads(raw)
            content = response["choices"][0]["message"]["content"]
            (case_root / "raw_model_content.txt").write_text(content, encoding="utf-8")
            candidate = json.loads(content)
            _write_json(case_root / "parsed_output.json", candidate)
            entry["parse_valid"] = True
            try:
                validated = GroundedOutputEnvelope.model_validate(candidate)
            except ValidationError as exc:
                safe = _safe_validation_error(exc)
                failure = {
                    "error_type": "ValidationError",
                    "classification": "EXT4C_SEMANTIC_VALIDATION_FAIL",
                    "validation_stage": "GroundedOutputEnvelope.model_validate",
                    "errors": safe,
                    "message": str(exc),
                }
                _write_json(case_root / "validation_error.json", failure)
                entry.update(
                    {
                        "ext4c_validation_executed": True,
                        "ext4c_valid": False,
                        "scorer_executed": False,
                        "case_pass": False,
                        "failure_classification": failure["classification"],
                        "validation_error": failure,
                    }
                )
            else:
                references = {item.evidence_id for item in envelope.evidence_items}
                if any(ref.evidence_id not in references for ref in validated.evidence_references):
                    raise RuntimeError("EXT4C_GROUNDING_REFERENCE_FAILURE")
                score = score_case(scoring_case(case), candidate)
                _write_json(case_root / "score.json", score)
                entry.update(
                    {
                        "ext4c_validation_executed": True,
                        "ext4c_valid": True,
                        "scorer_executed": True,
                        "case_pass": score["case_passed"],
                        "violations": score["violations"],
                    }
                )
            entry.update({"attempt_state": "ATTEMPTED_COMPLETE", "terminal_state": True})
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            failure = {
                "error_type": type(exc).__name__,
                "classification": "RESPONSE_PROCESSING_FAILURE",
                "validation_stage": "response_processing",
                "errors": [],
                "message": str(exc),
            }
            _write_json(case_root / "validation_error.json", failure)
            entry.update(
                {
                    "attempt_state": "ATTEMPTED_COMPLETE",
                    "terminal_state": True,
                    "generation_completed": True,
                    "failure_classification": failure["classification"],
                    "validation_error": failure,
                    "parse_valid": False,
                    "case_pass": False,
                }
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            failure = {
                "classification": "HTTP_REQUEST_FAILURE",
                "status_code": exc.code,
                "reason": exc.reason,
                "headers": dict(exc.headers.items()),
                "response_body": body,
            }
            _write_json(case_root / "http_error.json", failure)
            entry.update(
                {
                    "attempt_state": "ATTEMPTED_INCOMPLETE",
                    "terminal_state": True,
                    "failure_classification": failure["classification"],
                    "error": failure,
                }
            )
            raise RuntimeError(f"Technical request failure for {case_id}") from exc
        finally:
            entry["request_completed_at"] = datetime.now(UTC).isoformat()
            entry["evidence_path"] = str(case_root)
            metadata = {
                "case_id": case_id,
                "case_category": case["category"],
                "partition": "development",
                "inference_request_count": entry.get("request_count", 0),
                "retry_count": 0,
                "generation_started": entry.get("generation_started", False),
                "generation_completed": entry.get("generation_completed", False),
                "response_parse_valid": entry.get("parse_valid", False),
                "attempt_state": entry.get("attempt_state"),
                "ext4c_valid": entry.get("ext4c_valid"),
                "scorer_executed": entry.get("scorer_executed", False),
                "case_passed": entry.get("case_pass", False),
                "contract_status": entry.get("failure_classification", "EXT4C_VALID"),
                "development_cases_accessed": 1,
                "frozen_final_cases_accessed": 0,
                "locked_test_accessed": False,
            }
            _write_json(case_root / "run_metadata.json", metadata)
            _write_atomic(ledger_path, ledger)
        evidence_paths[case_id] = case_root
    aggregate = aggregate_evidence(cases_path, evidence_paths)
    _write_json(run_root / "development_case_results.json", aggregate["canonical_case_records"])
    _write_json(run_root / "development_violation_totals.json", aggregate["violation_counts"])
    decision = scientific_decision(aggregate["benchmark_pass"])
    summary = {
        **aggregate,
        "scientific_decision": decision,
        "development_requests_attempted": ledger["development_requests_attempted"],
        "development_cases_accessed": ledger["development_cases_accessed"],
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
    }
    _write_json(run_root / "development_summary.json", summary)
    _write_json(run_root / "case_attempt_ledger.json", ledger)
    if run_root.is_relative_to(ROOT / "artifacts"):
        report = {
            "schema_id": "EXT4E_CANDIDATE2_DEVELOPMENT_CLOSURE",
            "schema_version": "1",
            "candidate_id": 2,
            "candidate": "Ministral-3-8B-Instruct-2512-Q4_K_M",
            "runtime": {
                "release": "b8233",
                "commit": "c5a778891ba0ddbd4cbb507c823f970595b1adc2",
            },
            "development_case_ids": list(CASE_IDS),
            "aggregate": {
                key: summary[key]
                for key in (
                    "total_cases",
                    "completed_generations",
                    "parse_valid_count",
                    "ext4c_valid_count",
                    "ext4c_invalid_count",
                    "case_pass_count",
                    "case_fail_count",
                    "hard_safety_gate_pass",
                    "quality_gate_pass",
                    "benchmark_pass",
                )
            },
            "scientific_decision": decision,
            "development_cases_accessed": summary["development_cases_accessed"],
            "final_cases_accessed": 0,
            "locked_test_accessed": False,
            "next_stage": (
                "EXPLICIT_GOVERNANCE_DECISION_BEFORE_FROZEN_FINAL_ACCESS"
                if summary["benchmark_pass"]
                else "CANDIDATE_3_IDENTITY_AND_RUNTIME_BOOTSTRAP"
            ),
        }
        _write_json(
            ROOT / "reports/research_extensions/ext4e/EXT4E_CANDIDATE2_DEVELOPMENT_CLOSURE.json",
            report,
        )
    return summary


if __name__ == "__main__":
    raise SystemExit("Use run_development_evaluation from the Candidate #2 bootstrap.")
