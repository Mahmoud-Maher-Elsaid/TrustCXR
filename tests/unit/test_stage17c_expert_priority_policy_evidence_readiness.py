from __future__ import annotations

import json
from pathlib import Path

from scripts.triage.run_stage17c_expert_priority_policy_evidence_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/triage/stage17c_expert_priority_policy_evidence_readiness.json"


def test_stage17c_preserves_defer_only_policy() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["active_decisions"] == ["DEFER"]
    assert set(result["withheld_decisions"]) == {
        "INPUT_REJECTED",
        "ROUTINE",
        "PRIORITY",
        "URGENT_REVIEW",
        "CRITICAL_REVIEW",
    }
    assert not result["authoritative_priority_policy_found"]
    assert result["locked_test_records_accessed"] == 0


def test_stage17c_identifies_independent_next_stage() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["next_independent_stage"] == "18A_GROUNDED_REPORT_DATA_READINESS"
    assert result["limitation"] == "GOVERNANCE_LIMITATION_NOT_MODEL_FAILURE"
