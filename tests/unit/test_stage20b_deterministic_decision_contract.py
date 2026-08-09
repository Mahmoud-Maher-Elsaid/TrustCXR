from __future__ import annotations

import json
from pathlib import Path

from scripts.decision.run_stage20b_deterministic_decision_contract import freeze_contract

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/decision/stage20b_deterministic_decision_contract.json"
FIXTURES = ROOT / "configs/decision/stage20b_synthetic_contract_fixtures.json"


def result() -> dict:
    return freeze_contract(
        json.loads(CONFIG.read_text(encoding="utf-8")),
        json.loads(FIXTURES.read_text(encoding="utf-8")),
        ROOT,
    )


def test_stage20b_contract_is_traceable_and_defer_first() -> None:
    frozen = result()
    assert frozen["decision_precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert frozen["evidence_references_required"]
    assert frozen["synthetic_fixtures_prepared"] == 17
    assert not frozen["policy_activated"]


def test_stage20b_contract_preserves_status_safety() -> None:
    frozen = result()
    assert frozen["verifier_status_behavior"]["PARTIALLY_VERIFIED"] == "DEFER_NOT_ACCEPT"
    assert frozen["verifier_status_behavior"]["CONTRADICTED"] == "DEFER"
    assert frozen["verifier_status_behavior"]["WITHHELD_INSUFFICIENT_EVIDENCE"] == "DEFER"
    assert not frozen["next_stage_authorizes_language_model_work"]
    assert not frozen["language_model_used"]
    assert frozen["locked_test_records_accessed"] == 0
    assert "NO_CLINICAL_DIAGNOSIS" in frozen["mandatory_safety_limitations"]
    assert "EXPERT_REVIEW_REQUIRED" in frozen["mandatory_safety_limitations"]
