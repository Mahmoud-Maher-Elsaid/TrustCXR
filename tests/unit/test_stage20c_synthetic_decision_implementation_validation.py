from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.decision.run_stage20c_synthetic_decision_implementation_validation import (
    candidate,
    validate,
)

from trustcxr.decision.deterministic_policy import decide, validate_decision_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/decision/stage20c_synthetic_decision_implementation_validation.json"
CONTRACT = ROOT / "reports/stage20/stage20b_deterministic_decision_contract_summary.json"


def contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_stage20c_synthetic_fixture_coverage_and_safety() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["fixtures_passed"] == 17
    assert result["fixtures_failed"] == 0
    assert not result["real_patient_policy_activated"]
    assert not result["language_model_used"]
    assert result["locked_test_records_accessed"] == 0
    assert result["patient_identifiers_used"] == 0


def test_multiple_defer_reasons_are_canonical_and_preserved() -> None:
    output = decide(
        candidate(active_stage17_defer=True, forbidden_claim=True, exact_identity=False),
        contract(),
    )
    assert output["decision"] == "DEFER"
    assert output["reason_codes"] == [
        "IDENTITY_MISMATCH",
        "ACTIVE_STAGE17_DEFER",
        "FORBIDDEN_CLAIM",
    ]


def test_output_validator_rejects_fabricated_reason_or_reference() -> None:
    source = candidate()
    output = decide(source, contract())
    fabricated_reason = output | {"reason_codes": ["FABRICATED_REASON"]}
    with pytest.raises(ValueError, match="fabricated"):
        validate_decision_output(fabricated_reason, contract(), source["evidence_references"])
    fabricated_reference = output | {"evidence_references": ["synthetic/fabricated"]}
    with pytest.raises(ValueError, match="fabricated"):
        validate_decision_output(fabricated_reference, contract(), source["evidence_references"])


def test_patient_identifier_field_is_rejected() -> None:
    source = candidate()
    source["patient_id"] = "synthetic-prohibited"
    with pytest.raises(ValueError, match="fields"):
        decide(source, contract())
