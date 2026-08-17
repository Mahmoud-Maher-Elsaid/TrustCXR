"""Offline preparation and aggregation helpers for the EXT-4E1 six-case run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.grounded_llm.benchmark import score_benchmark

DEV_CASE_COUNT = 6
FINAL_CASE_COUNT = 24
EXCLUDED_CASE_ID = "dev_supported"
EXPECTED_CONFIG_SHA256 = "df4495f507eb2d05576f66de4d7f7c7d8fefbc9076956d128f1d5959472c6cab"
EXPECTED_CASES_SHA256 = "ddef17b136f558934295deae506fb8e9ff34f60e97008c290f2e0067c4a2e548"
EXPECTED_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
EXPECTED_PROMPT_SHA256 = "41ef8d42303bdcfc238d64f9528796bf42c94935c55296c8c7a361c74b5d6a61"
EXPECTED_RUNTIME_RELEASE = "b10453"
EXPECTED_RUNTIME_COMMIT_PREFIX = "3cb7ffb"


class DevelopmentEvaluationContractFailure(RuntimeError):
    """The frozen six-case evaluation contract is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_cases(path: Path) -> tuple[tuple[dict[str, Any], ...], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    development = tuple(payload.get("development_cases", ()))
    final_count = len(payload.get("final_cases", ()))
    if len(development) != DEV_CASE_COUNT or final_count != FINAL_CASE_COUNT:
        raise DevelopmentEvaluationContractFailure("EXT-4D partition counts are not frozen.")
    if any("final" in case.get("case_id", "") for case in development):
        raise DevelopmentEvaluationContractFailure(
            "A final case entered the development partition."
        )
    if len({case.get("case_id") for case in development}) != DEV_CASE_COUNT:
        raise DevelopmentEvaluationContractFailure("Development case IDs are not unique.")
    return development, final_count


def validate_frozen_identity(config: dict[str, Any]) -> None:
    expected = {
        "model_sha256": EXPECTED_MODEL_SHA256,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "runtime_release": EXPECTED_RUNTIME_RELEASE,
        "runtime_commit_prefix": EXPECTED_RUNTIME_COMMIT_PREFIX,
        "request_reasoning_effort": "none",
        "structured_output_mechanism": "REQUEST_RESPONSE_FORMAT_JSON_OBJECT_WITH_SCHEMA",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise DevelopmentEvaluationContractFailure("Candidate #1 frozen identity differs.")
    generation = config.get("generation", {})
    if any(
        generation.get(key) != value
        for key, value in {
            "request_count": 1,
            "retry_count": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260806,
            "max_tokens": 768,
            "stream": False,
            "free_form_fallback": False,
        }.items()
    ):
        raise DevelopmentEvaluationContractFailure("Candidate #1 generation policy differs.")


def validate_historical_dev_supported(
    evidence_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    run_root = evidence_root / "20260817T091916Z"
    metadata = json.loads((run_root / "run_metadata.json").read_text(encoding="utf-8"))
    score = json.loads((run_root / "score.json").read_text(encoding="utf-8"))
    required = {
        "status": "DEV_CASE_SMOKE_PASS",
        "case_id": EXCLUDED_CASE_ID,
        "partition": "development",
        "inference_request_count": 1,
        "retry_count": 0,
        "response_parse_valid": True,
        "reasoning_content_present": False,
        "server_cleanup_confirmed": True,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise DevelopmentEvaluationContractFailure(
            "Preserved dev_supported evidence is incompatible."
        )
    if not score.get("case_passed") or not score.get("valid"):
        raise DevelopmentEvaluationContractFailure("Preserved dev_supported score did not pass.")
    if metadata.get("model_sha256") != config["model_sha256"]:
        raise DevelopmentEvaluationContractFailure("Preserved model identity differs.")
    if metadata.get("prompt_sha256") != config["prompt_sha256"]:
        raise DevelopmentEvaluationContractFailure("Preserved prompt identity differs.")
    if metadata.get("runtime_release") != config["runtime_release"]:
        raise DevelopmentEvaluationContractFailure("Preserved runtime identity differs.")
    if metadata.get("runtime_commit_prefix") != config["runtime_commit_prefix"]:
        raise DevelopmentEvaluationContractFailure("Preserved runtime commit differs.")
    return {"case_id": EXCLUDED_CASE_ID, "evidence_path": str(run_root), "score": score}


def build_evaluation_plan(
    cases_path: Path, config_path: Path, historical_root: Path
) -> dict[str, Any]:
    if sha256_file(cases_path) != EXPECTED_CASES_SHA256:
        raise DevelopmentEvaluationContractFailure("EXT-4D cases hash mismatch.")
    development, final_count = load_development_cases(cases_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_identity(config)
    if config.get("partition") != "development":
        raise DevelopmentEvaluationContractFailure("Evaluation is not development-only.")
    historical = validate_historical_dev_supported(historical_root, config)
    all_ids = [case["case_id"] for case in development]
    remaining = [case_id for case_id in all_ids if case_id != EXCLUDED_CASE_ID]
    if len(remaining) != 5 or len(set(remaining)) != 5:
        raise DevelopmentEvaluationContractFailure("Expected exactly five remaining cases.")
    return {
        "development_case_ids": all_ids,
        "remaining_case_ids": remaining,
        "development_case_count": len(development),
        "final_case_count": final_count,
        "historical_reuse": historical,
        "new_request_count_per_case": 1,
        "new_retry_count_per_case": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
    }


def aggregate_evidence(cases_path: Path, evidence_paths: dict[str, Path]) -> dict[str, Any]:
    development, _ = load_development_cases(cases_path)
    candidates: dict[str, dict[str, Any]] = {}
    for case in development:
        case_id = case["case_id"]
        if case_id not in evidence_paths or case_id in candidates:
            raise DevelopmentEvaluationContractFailure("Missing or duplicate case evidence.")
        root = evidence_paths[case_id]
        metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("case_id") != case_id or metadata.get("retry_count") != 0:
            raise DevelopmentEvaluationContractFailure("Case evidence violates execution policy.")
        candidates[case_id] = json.loads((root / "parsed_output.json").read_text(encoding="utf-8"))
    result = score_benchmark(development, candidates)
    return {
        "total_cases": len(development),
        "previously_executed": 1,
        "newly_executed": len(development) - 1,
        "completed_generations": sum(
            json.loads((evidence_paths[case["case_id"]] / "run_metadata.json").read_text()).get(
                "generation_completed", False
            )
            for case in development
        ),
        "parse_valid_count": sum(
            json.loads((evidence_paths[case["case_id"]] / "run_metadata.json").read_text()).get(
                "response_parse_valid", False
            )
            for case in development
        ),
        "case_pass_count": sum(item["case_passed"] for item in result["case_results"]),
        "case_fail_count": sum(not item["case_passed"] for item in result["case_results"]),
        "violation_counts": result["violation_counts"],
        "case_results": result["case_results"],
        "metrics": result["metrics"],
        "hard_safety_gate_pass": result["hard_safety_gate_pass"],
        "quality_gate_pass": result["quality_gate_pass"],
        "benchmark_pass": result["benchmark_pass"],
    }
