"""Offline EXT-4E1 six-development-case preparation tests."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import GroundedOutputEnvelope
from trustcxr.grounded_llm.development_evaluation import (
    DevelopmentEvaluationContractFailure,
    build_evaluation_plan,
    load_development_cases,
    scoring_case,
)

ROOT = Path(__file__).parents[2]
CASES = ROOT / "tests/fixtures/ext4d_benchmark_cases.json"
CONFIG = ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json"
EVIDENCE = ROOT / "artifacts/research_extensions/ext4e2_candidate1/dev_case_smoke"


def test_six_case_partition_and_historical_reuse_are_frozen():
    development, final_count = load_development_cases(CASES)
    assert len(development) == 6
    assert final_count == 24
    plan = build_evaluation_plan(CASES, CONFIG, EVIDENCE)
    assert plan["development_case_ids"] == [
        "dev_supported",
        "dev_uncertainty",
        "dev_defer",
        "dev_withheld",
        "dev_missing",
        "dev_conflict",
    ]
    assert plan["remaining_case_ids"] == [
        "dev_uncertainty",
        "dev_defer",
        "dev_withheld",
        "dev_missing",
        "dev_conflict",
    ]
    assert plan["historical_reuse"]["case_id"] == "dev_supported"
    assert plan["new_request_count_per_case"] == 1
    assert plan["new_retry_count_per_case"] == 0


def test_final_partition_and_invalid_counts_fail_closed(tmp_path):
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    payload["development_cases"].append(payload["final_cases"][0])
    mutated = tmp_path / "cases.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentEvaluationContractFailure):
        load_development_cases(mutated)


def test_historical_evidence_is_not_rerun_or_rewritten():
    plan = build_evaluation_plan(CASES, CONFIG, EVIDENCE)
    historical = Path(plan["historical_reuse"]["evidence_path"])
    assert (historical / "raw_model_content.txt").is_file()
    assert plan["historical_reuse"]["score"]["case_passed"] is True
    assert plan["historical_reuse"]["score"]["valid"] is True


def test_duplicate_case_ids_fail_closed(tmp_path):
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    payload["development_cases"].append(
        dict(payload["development_cases"][0], case_id="dev_uncertainty")
    )
    mutated = tmp_path / "cases.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentEvaluationContractFailure):
        load_development_cases(mutated)


def test_defer_scoring_policy_is_explicit_without_mutating_fixture():
    development, _ = load_development_cases(CASES)
    defer_case = next(case for case in development if case["case_id"] == "dev_defer")
    assert "expected_statuses" not in defer_case
    assert scoring_case(defer_case)["expected_statuses"] == ["DEFERRED", "ABSTAINED"]


def test_consumed_case_discovery_never_selects_failed_case_again():
    path = ROOT / "scripts/training/run_ext4e_candidate1_development_evaluation.py"
    spec = importlib.util.spec_from_file_location("batch_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root, consumed = module.discover_consumed_cases(
        ROOT / "artifacts/research_extensions/ext4e_candidate1/development_evaluation",
        {"dev_uncertainty", "dev_defer", "dev_withheld", "dev_missing", "dev_conflict"},
    )
    assert root is not None
    assert consumed == {
        "dev_uncertainty",
        "dev_defer",
        "dev_withheld",
        "dev_missing",
        "dev_conflict",
    }


def test_aggregate_accepts_completed_semantic_failure(tmp_path):
    from trustcxr.grounded_llm.development_evaluation import aggregate_evidence

    source = EVIDENCE / "20260817T091916Z"
    paths = {"dev_supported": source}
    failed = ROOT / "artifacts/research_extensions/ext4e_candidate1/development_evaluation"
    failed = failed / "20260817T095827Z/dev_uncertainty"
    paths["dev_uncertainty"] = failed
    for case_id in ["dev_defer", "dev_withheld", "dev_missing", "dev_conflict"]:
        target = tmp_path / case_id
        target.mkdir()
        metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
        metadata.update(
            {"case_id": case_id, "generation_completed": True, "response_parse_valid": True}
        )
        (target / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        shutil.copy2(source / "parsed_output.json", target / "parsed_output.json")
        paths[case_id] = target
    aggregate = aggregate_evidence(CASES, paths)
    assert aggregate["total_cases"] == 6
    assert set(aggregate["canonical_case_records"]) == {
        "dev_supported",
        "dev_uncertainty",
        "dev_defer",
        "dev_withheld",
        "dev_missing",
        "dev_conflict",
    }
    assert aggregate["case_fail_count"] >= 1
    failed_result = next(
        item for item in aggregate["case_results"] if item["case_id"] == "dev_uncertainty"
    )
    assert failed_result["contract_status"] == "EXT4C_SEMANTIC_VALIDATION_FAIL"


def test_validation_error_with_value_error_context_is_json_safe():
    path = ROOT / "scripts/training/run_ext4e_candidate1_development_evaluation.py"
    spec = importlib.util.spec_from_file_location("batch_runner_errors", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    candidate_path = (
        ROOT
        / "artifacts/research_extensions/ext4e_candidate1/development_evaluation"
        / "20260817T095827Z/dev_uncertainty/raw_model_content.txt"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError) as captured:
        GroundedOutputEnvelope.model_validate(candidate)
    safe = module.serialize_validation_error(captured.value)
    assert json.loads(json.dumps(safe)) == safe
    assert all("ctx" not in item for item in safe)
