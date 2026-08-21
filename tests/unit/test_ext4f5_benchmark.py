from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.ext4f5_benchmark import (
    BENCHMARK_VERSION,
    CASE_IDS,
    benchmark_manifest,
    build_development_cases,
    score_mock_realization,
    validate_benchmark_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def _valid_response(case):
    return {
        "contract_version": "EXT4F_REALIZATION_CONTRACT_V1",
        "semantic_plan_sha256": case.plan.semantic_plan_sha256,
        "realization_request_sha256": case.request.realization_request_sha256,
        "slots": [
            {"slot_id": slot.slot_id, "text": "Synthetic wording for the authorized fact."}
            for slot in case.request.slots
            if slot.required
        ],
    }


def test_benchmark_has_exact_new_order_and_all_cases_validate():
    cases = build_development_cases()
    assert tuple(case.case_id for case in cases) == CASE_IDS
    assert len({case.case_sha256 for case in cases}) == 24
    assert all(case.plan.semantic_plan_sha256 for case in cases)
    assert all(case.request.realization_request_sha256 for case in cases)


def test_manifest_is_frozen_and_matches_case_factory():
    manifest = benchmark_manifest()
    validate_benchmark_manifest(manifest)
    on_disk = json.loads(
        (ROOT / "configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json").read_text()
    )
    assert manifest == on_disk
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert manifest["case_count"] == 24
    assert all(case.expectations for case in build_development_cases())


def test_coverage_includes_states_slots_and_risk_taxonomy():
    cases = build_development_cases()
    states = {ref.status.value for case in cases for ref in case.plan.evidence_references}
    slots = {slot.slot_type for case in cases for slot in case.request.slots}
    risks = {risk for case in cases for risk in case.risk_tags}
    assert states == {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "WITHHELD",
        "CONTRADICTED",
        "NOT_AVAILABLE",
        "NOT_APPLICABLE",
    }
    assert slots == {
        "CLAIM_EXPLANATION",
        "UNCERTAINTY_EXPLANATION",
        "LIMITATION_EXPLANATION",
        "CONTRADICTION_EXPLANATION",
        "DEFER_EXPLANATION",
        "REVIEWER_QUESTION",
    }
    assert "WITHHELD_AS_NEGATIVE" in risks
    assert "NOT_AVAILABLE_AS_NEGATIVE" in risks
    assert "NOT_APPLICABLE_AS_MISSING" in risks
    assert "DEFER_OVERRIDE" in risks


def test_scorer_hard_gates_and_review_required_semantics():
    case = build_development_cases()[0]
    valid = score_mock_realization(case, _valid_response(case))
    assert valid["automatic_status"] == "PASS"
    assert valid["semantic_adjudication"] == "REVIEW_REQUIRED"
    assert valid["case_pass"] is False

    wrong_plan = _valid_response(case)
    wrong_plan["semantic_plan_sha256"] = "0" * 64
    assert score_mock_realization(case, wrong_plan)["automatic_status"] == "FAIL"

    unknown_slot = _valid_response(case)
    unknown_slot["slots"][0]["slot_id"] = "slot_unknown"
    assert score_mock_realization(case, unknown_slot)["automatic_status"] == "FAIL"

    semantic_risk_text = _valid_response(case)
    semantic_risk_text["slots"][0]["text"] = "The observation is negative and definitive."
    semantic_risk = score_mock_realization(case, semantic_risk_text)
    assert semantic_risk["automatic_status"] == "PASS"
    assert semantic_risk["semantic_adjudication"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "leakage"])
def test_scorer_rejects_authority_and_boundary_mutations(mutation):
    case = build_development_cases()[0]
    response = _valid_response(case)
    if mutation == "duplicate":
        response["slots"].append(dict(response["slots"][0]))
    elif mutation == "missing":
        response["slots"] = []
    else:
        response["slots"][0]["text"] = "``` EXT4F_SEMANTIC_GENERATION_CONTRACT_V1 ```"
    assert score_mock_realization(case, response)["automatic_status"] == "FAIL"


def test_benchmark_isolated_from_prior_partitions_and_smoke_case():
    source = (ROOT / "src/trustcxr/grounded_llm/ext4f5_benchmark.py").read_text()
    assert "ext4d_benchmark_cases" not in source
    assert "research_case_ext4f4_realization_001" not in source
    manifest = benchmark_manifest()
    assert manifest["partition"] == "EXT4F_DEVELOPMENT_ONLY"
    assert manifest["final_case_access_policy"] == "CLOSED"
    assert manifest["locked_test_policy"] == "CLOSED"
