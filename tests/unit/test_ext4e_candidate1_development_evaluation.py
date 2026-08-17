"""Offline EXT-4E1 six-development-case preparation tests."""

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.development_evaluation import (
    DevelopmentEvaluationContractFailure,
    build_evaluation_plan,
    load_development_cases,
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
