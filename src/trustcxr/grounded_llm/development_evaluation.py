"""Offline preparation and aggregation helpers for the EXT-4E1 six-case run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.grounded_llm.benchmark import TAXONOMY, score_case

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


def scoring_case(case: dict[str, Any]) -> dict[str, Any]:
    """Add only deterministic development scoring defaults; never alter fixture data."""

    result = dict(case)
    if case.get("grounding_kind") == "defer":
        result["expected_statuses"] = ["DEFERRED", "ABSTAINED"]
    else:
        result["expected_statuses"] = ["COMPLETED"]
    return result


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
    results: list[dict[str, Any]] = []

    def metadata_for(case_id: str, root: Path) -> dict[str, Any]:
        metadata_path = root / "run_metadata.json"
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        failure_path = root / "validation_error.json"
        if failure_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            return {
                "case_id": case_id,
                "retry_count": failure.get("retry_count"),
                "inference_request_count": failure.get("request_count", 0),
                "generation_completed": failure.get("generation_completed"),
                "response_parse_valid": True,
                "attempt_state": "ATTEMPTED_COMPLETE",
            }
        raise DevelopmentEvaluationContractFailure("Case evidence metadata is missing.")

    def evidence_is_complete(root: Path) -> bool:
        required = (
            "input_envelope.json",
            "ext4c_output_schema.json",
            "request.json",
            "raw_http_response.json",
            "raw_model_content.txt",
            "validation_error.json",
        )
        if not all((root / name).is_file() for name in required):
            return False
        failure_path = root / "validation_error.json"
        if failure_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            if failure.get("evidence_complete") is False:
                return False
        return True

    for case in development:
        case_id = case["case_id"]
        if case_id not in evidence_paths:
            raise DevelopmentEvaluationContractFailure("Missing or duplicate case evidence.")
        root = evidence_paths[case_id]
        metadata = metadata_for(case_id, root)
        if metadata.get("case_id") != case_id or metadata.get("retry_count") != 0:
            raise DevelopmentEvaluationContractFailure("Case evidence violates execution policy.")
        parsed = root / "parsed_output.json"
        failure = root / "validation_error.json"
        # A preserved parsed candidate may coexist with a semantic validation
        # failure; the contract result, not JSON parseability, is authoritative.
        if failure.is_file() and metadata.get("ext4c_valid") is False:
            if metadata.get("generation_completed") is not True:
                raise DevelopmentEvaluationContractFailure("Invalid case evidence is incomplete.")
            request_count = metadata.get("inference_request_count", 0)
            attempt_state = metadata.get("attempt_state", "ATTEMPTED_COMPLETE")
            if not evidence_is_complete(root):
                attempt_state = "ATTEMPTED_INCOMPLETE"
            results.append(
                {
                    "case_id": case_id,
                    "category": case["category"],
                    "valid": False,
                    "case_passed": False,
                    "violations": {name: None for name in TAXONOMY},
                    "generation_status": "COMPLETED",
                    "technical_status": "COMPLETED",
                    "contract_status": "EXT4C_SEMANTIC_VALIDATION_FAIL",
                    "validation_error": json.loads(failure.read_text(encoding="utf-8")),
                    "evidence_path": str(root),
                    "attempt_state": "HISTORICAL_REUSED"
                    if case_id == "dev_supported"
                    else attempt_state,
                    "request_attempted": request_count == 1,
                    "request_count": request_count,
                    "generation_completed": metadata.get("generation_completed", False),
                    "parse_valid": metadata.get("response_parse_valid", True),
                    "ext4c_valid": False,
                    "scorer_executed": False,
                }
            )
            continue
        if parsed.is_file():
            result = score_case(scoring_case(case), json.loads(parsed.read_text(encoding="utf-8")))
            result["technical_status"] = "COMPLETED"
            result["contract_status"] = "EXT4C_VALID"
            result.update(
                {
                    "evidence_path": str(root),
                    "attempt_state": (
                        "HISTORICAL_REUSED"
                        if case_id == "dev_supported"
                        else metadata.get("attempt_state", "ATTEMPTED_COMPLETE")
                    ),
                    "request_attempted": metadata.get("inference_request_count", 0) == 1,
                    "request_count": metadata.get("inference_request_count", 0),
                    "generation_completed": metadata.get("generation_completed", False),
                    "parse_valid": metadata.get("response_parse_valid", False),
                    "ext4c_valid": True,
                    "scorer_executed": True,
                    "validation_error": None,
                }
            )
            results.append(result)
            continue
        if not failure.is_file() or metadata.get("generation_completed") is not True:
            raise DevelopmentEvaluationContractFailure("Invalid case evidence is incomplete.")
        request_count = metadata.get("inference_request_count", 0)
        attempt_state = metadata.get("attempt_state", "ATTEMPTED_COMPLETE")
        if not evidence_is_complete(root):
            attempt_state = "ATTEMPTED_INCOMPLETE"
        results.append(
            {
                "case_id": case_id,
                "category": case["category"],
                "valid": False,
                "case_passed": False,
                "violations": {name: None for name in TAXONOMY},
                "generation_status": "COMPLETED",
                "technical_status": "COMPLETED",
                "contract_status": "EXT4C_SEMANTIC_VALIDATION_FAIL",
                "validation_error": json.loads(failure.read_text(encoding="utf-8")),
                "evidence_path": str(root),
                "attempt_state": (
                    "HISTORICAL_REUSED" if case_id == "dev_supported" else attempt_state
                ),
                "request_attempted": request_count == 1,
                "request_count": request_count,
                "generation_completed": metadata.get("generation_completed", False),
                "parse_valid": metadata.get("response_parse_valid", True),
                "ext4c_valid": False,
                "scorer_executed": False,
            }
        )
    total = len(results)
    counts = {name: 0 for name in TAXONOMY}
    for item in results:
        for name, count in item["violations"].items():
            if isinstance(count, int):
                counts[name] += count
    valid_count = sum(item["valid"] for item in results)
    passed = sum(item["case_passed"] for item in results)

    def rate(count: int) -> float:
        return count / total if total else 0.0

    metrics = {
        "structured_output_validity_rate": rate(valid_count),
        "unsupported_claim_rate": rate(counts["UNSUPPORTED_CLAIM"]),
        "contradicted_claim_rate": rate(counts["CONTRADICTED_CLAIM"]),
        "provenance_error_rate": rate(counts["PROVENANCE_ERROR"]),
        "fabricated_detail_rate": rate(counts["FABRICATED_DETAIL"]),
        "defer_compliance_rate": 1.0 - rate(counts["DEFER_VIOLATION"]),
        "withheld_evidence_compliance_rate": 1.0 - rate(counts["WITHHELD_EVIDENCE_VIOLATION"]),
        "evidence_polarity_error_rate": rate(counts["EVIDENCE_POLARITY_ERROR"]),
        "prohibited_claim_rate": rate(counts["UNSUPPORTED_CLAIM"]),
        "unsupported_localization_rate": rate(counts["UNSUPPORTED_LOCALIZATION"]),
        "unsupported_severity_rate": rate(counts["UNSUPPORTED_SEVERITY"]),
        "unsupported_laterality_rate": rate(counts["UNSUPPORTED_LATERALITY"]),
        "case_pass_rate": rate(passed),
    }
    hard_gate_pass = valid_count == total and not any(
        counts[name] > 0
        for name in (
            "UNSUPPORTED_CLAIM",
            "CONTRADICTED_CLAIM",
            "PROVENANCE_ERROR",
            "FABRICATED_DETAIL",
            "DEFER_VIOLATION",
            "WITHHELD_EVIDENCE_VIOLATION",
            "EVIDENCE_POLARITY_ERROR",
            "UNSUPPORTED_LOCALIZATION",
            "UNSUPPORTED_SEVERITY",
            "UNSUPPORTED_LATERALITY",
        )
    )
    quality_gate_pass = (
        metrics["structured_output_validity_rate"] == 1.0
        and metrics["provenance_error_rate"] == 0.0
        and metrics["defer_compliance_rate"] == 1.0
        and metrics["withheld_evidence_compliance_rate"] == 1.0
        and metrics["unsupported_claim_rate"] == 0.0
    )
    return {
        "total_cases": total,
        "previously_executed": 1,
        "newly_executed": total - 1,
        "historical_reused_count": sum(
            item["attempt_state"] == "HISTORICAL_REUSED" for item in results
        ),
        "inference_consumed_count": sum(item["request_attempted"] for item in results),
        "completed_generations": sum(
            metadata_for(case["case_id"], evidence_paths[case["case_id"]]).get(
                "generation_completed", False
            )
            for case in development
        ),
        "parse_valid_count": sum(
            metadata_for(case["case_id"], evidence_paths[case["case_id"]]).get(
                "response_parse_valid", False
            )
            for case in development
        ),
        "ext4c_valid_count": sum(item["ext4c_valid"] for item in results),
        "ext4c_invalid_count": sum(not item["ext4c_valid"] for item in results),
        "scorer_executed_count": sum(item["scorer_executed"] for item in results),
        "incomplete_evidence_count": sum(
            item["attempt_state"] == "ATTEMPTED_INCOMPLETE" for item in results
        ),
        "protocol_deviation_count": 0,
        "resume_pending_count": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "case_pass_count": passed,
        "case_fail_count": total - passed,
        "violation_counts": counts,
        "case_results": results,
        "canonical_case_records": {item["case_id"]: item for item in results},
        "metrics": metrics,
        "hard_safety_gate_pass": hard_gate_pass,
        "quality_gate_pass": quality_gate_pass,
        "benchmark_pass": hard_gate_pass and quality_gate_pass,
    }
