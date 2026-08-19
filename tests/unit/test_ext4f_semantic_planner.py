"""Deterministic, non-LLM EXT-4F.1 semantic planner tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import (
    EXT4F_CANONICALIZATION_V1,
    EXT4F_SEMANTIC_GENERATION_CONTRACT_V1,
    AvailableUncertainty,
    Ext4fSemanticPlanError,
    SemanticContradiction,
    UnavailableUncertainty,
    build_ext4f_semantic_plan,
    canonical_plan_json,
    load_ext4f_semantic_plan,
    validate_ext4f_semantic_plan,
)


def test_ext4f_contract_version_fixed():
    assert EXT4F_SEMANTIC_GENERATION_CONTRACT_V1 == "EXT4F_SEMANTIC_GENERATION_CONTRACT_V1"
    assert EXT4F_CANONICALIZATION_V1 == "EXT4F_CANONICAL_JSON_V1"


def test_ext4f_planner_is_deterministic_and_round_trips():
    evidence = build_synthetic_case("supported")
    first = build_ext4f_semantic_plan(evidence)
    second = build_ext4f_semantic_plan(evidence)
    assert first.semantic_plan_sha256 == second.semantic_plan_sha256
    assert canonical_plan_json(first) == canonical_plan_json(second)
    assert load_ext4f_semantic_plan(canonical_plan_json(first)) == first


def test_ext4f_semantic_content_changes_hash():
    first = build_ext4f_semantic_plan(build_synthetic_case("supported"))
    second = build_ext4f_semantic_plan(build_synthetic_case("uncertainty"))
    assert first.semantic_plan_sha256 != second.semantic_plan_sha256


def test_ext4f_state_variants_preserve_governed_states():
    available = build_ext4f_semantic_plan(build_synthetic_case("uncertainty"))
    unavailable = build_ext4f_semantic_plan(build_synthetic_case("missing"))
    assert isinstance(available.uncertainty, AvailableUncertainty)
    assert isinstance(unavailable.uncertainty, UnavailableUncertainty)
    assert unavailable.uncertainty.model_dump() == {"status": "NOT_AVAILABLE"}
    withheld = build_ext4f_semantic_plan(build_synthetic_case("withheld"))
    assert all(claim.support_status.name != "WITHHELD" for claim in withheld.claims)


def test_ext4f_all_deterministic_input_kinds_plan_without_llm():
    for kind in ("supported", "uncertainty", "defer", "withheld", "missing", "conflict"):
        plan = build_ext4f_semantic_plan(build_synthetic_case(kind))
        assert plan.semantic_plan_sha256
        validate_ext4f_semantic_plan(plan)


def test_ext4f_defer_is_deterministic_and_restricts_claims():
    plan = build_ext4f_semantic_plan(build_synthetic_case("defer"))
    assert plan.defer_state.defer_active is True
    assert plan.defer_state.decision == "DEFER"
    assert plan.claims == ()
    assert plan.allowed_realization.reviewer_topics == ("defer_reason",)


def test_ext4f_contradiction_requires_two_sources():
    with pytest.raises(ValueError, match="REQUIRES_TWO_EVIDENCE_IDS"):
        SemanticContradiction(contradiction_id="c1", evidence_ids=("e1",))


def test_ext4f_invalid_plan_reference_and_identity_fail_closed():
    plan = build_ext4f_semantic_plan(build_synthetic_case("supported"))
    payload = plan.model_dump(mode="json")
    payload["allowed_realization"]["evidence_ids"] = ["unknown"]
    with pytest.raises(Ext4fSemanticPlanError, match="ALLOWED_REALIZATION_REFERENCE_INVALID"):
        validate_ext4f_semantic_plan(type(plan).model_validate(payload))
    payload = plan.model_dump(mode="json")
    payload["semantic_plan_sha256"] = "0" * 64
    with pytest.raises(Ext4fSemanticPlanError, match="IDENTITY_MISMATCH"):
        validate_ext4f_semantic_plan(type(plan).model_validate(payload))


def test_ext4f_invalid_uncertainty_cannot_enter_plan():
    with pytest.raises(ValidationError):
        AvailableUncertainty(
            status="AVAILABLE",
            source_stage=None,
            value=0.2,
            interpretation="PREDICTIVE_ONLY_NOT_EPISTEMIC",
        )


def test_ext4f_fixture_policy_is_new_and_non_benchmark():
    fixture = json.loads(Path("tests/fixtures/ext4f_semantic_contract_cases.json").read_text())
    assert fixture["benchmark"] is False
    assert len(fixture["valid"]) == 6
    assert len(fixture["invalid"]) == 8
