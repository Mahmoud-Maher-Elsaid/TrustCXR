from __future__ import annotations

import json
from pathlib import Path

from scripts.reporting.run_stage18e_grounded_report_acceptance_decision import decide

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reporting/stage18e_grounded_report_acceptance_decision.json"


def test_stage18e_acceptance_is_research_only_and_limited() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["decision"] == "ACCEPT_DETERMINISTIC_GROUNDED_RESEARCH_REPORT_DRAFTING_ONLY"
    assert result["designation"] == "RESEARCH_ONLY_EXPERT_REVIEW_REQUIRED"
    assert not result["clinical_deployment_permitted"]
    assert not result["autonomous_report_use_permitted"]
    assert result["real_patient_reports_generated"] == 0
    assert result["locked_test_records_accessed"] == 0


def test_stage18e_preserves_all_mandatory_limitations() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert len(result["mandatory_limitations"]) == 14
    assert "NO_NEGATION_FROM_LOCALIZATION_ABSENCE" in result["mandatory_limitations"]
    assert "NO_CLINICAL_CERTAINTY_FROM_PROBABILITIES" in result["mandatory_limitations"]
    assert (
        "EVERY_STATEMENT_REQUIRES_PROVENANCE_AND_EVIDENCE_CODE" in result["mandatory_limitations"]
    )
    assert result["indiana_reports_status"] == "WITHHELD_PATIENT_IDENTITY_UNRESOLVED"
