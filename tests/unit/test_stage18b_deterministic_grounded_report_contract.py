from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.reporting.run_stage18b_deterministic_grounded_report_contract import (
    freeze_contract,
)

from trustcxr.reporting.grounded_contract import validate_payload, validate_statement

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reporting/stage18b_deterministic_grounded_report_contract.json"


def contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def grounded_statement() -> dict:
    return {
        "evidence_type": "stage9_classifier_finding_signal",
        "grounding_status": "EXPLICIT_UNCERTAINTY",
        "template_id": "CLASSIFIER_SIGNAL_UNCERTAIN",
        "template_parameters": {"finding": "Atelectasis", "model_score": "0.42"},
        "source_stage": "9",
        "source_version": "frozen",
        "evidence_code": "CLASSIFIER_SIGNAL",
        "structured_source_field": "probabilities/Atelectasis",
    }


def test_grounded_statement_requires_provenance_and_uncertainty() -> None:
    validate_statement(grounded_statement(), contract())
    invalid = grounded_statement()
    invalid.pop("evidence_code")
    with pytest.raises(ValueError, match="provenance"):
        validate_statement(invalid, contract())
    invalid = grounded_statement() | {"grounding_status": "DIRECT_STRUCTURED"}
    with pytest.raises(ValueError, match="uncertainty"):
        validate_statement(invalid, contract())


def test_forbidden_claim_and_missing_evidence_fail_closed() -> None:
    forbidden = grounded_statement() | {"evidence_type": "severity"}
    with pytest.raises(ValueError, match="Forbidden"):
        validate_statement(forbidden, contract())


def test_reason_codes_and_patient_privacy_are_enforced() -> None:
    payload = {
        "report_identity": contract()["report_identity"],
        "research_use_disclaimer": contract()["research_use_disclaimer"],
        "statements": [grounded_statement()],
        "omitted_capabilities": [{"capability": "severity", "reason_code": "WITHHELD"}],
    }
    validate_payload(payload, contract())
    with pytest.raises(ValueError, match="Patient-identifying"):
        validate_payload(payload | {"patient_id": "prohibited"}, contract())
    invalid = payload | {"omitted_capabilities": [{"capability": "severity"}]}
    with pytest.raises(ValueError, match="reason code"):
        validate_payload(invalid, contract())


def test_stage18b_is_contract_only_and_indiana_is_withheld() -> None:
    result = freeze_contract(contract(), ROOT)
    assert result["indiana_reports_status"] == "WITHHELD_PATIENT_IDENTITY_UNRESOLVED"
    assert not result["indiana_reports_used"]
    assert not result["report_generation_performed"]
    assert result["locked_test_records_accessed"] == 0
