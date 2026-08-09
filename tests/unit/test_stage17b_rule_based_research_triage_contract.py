from __future__ import annotations

import json
from pathlib import Path

from scripts.triage.run_stage17b_rule_based_research_triage_contract import (
    freeze_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/triage/stage17b_rule_based_research_triage_contract.json"


def test_stage17b_activates_only_evidence_supported_deferral() -> None:
    result = freeze_contract(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert {rule["decision"] for rule in result["active_rules"]} == {"DEFER"}
    assert result["precedence"] == ["INPUT_REJECTED", "DEFER", "REVIEW_PRIORITY"]
    assert result["gate"] == "HOLD_FOR_EXPERT_APPROVED_FINDING_PRIORITY_POLICY"
    assert result["locked_test_records_accessed"] == 0


def test_stage17b_preserves_stage13_and_ood_withholdings() -> None:
    result = freeze_contract(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert "stage13_selective_prediction" in result["mandatory_withholdings"]
    assert "ood_detection" in result["mandatory_withholdings"]
    assert result["inactive_decisions"]["INPUT_REJECTED"].startswith("WITHHELD_")
