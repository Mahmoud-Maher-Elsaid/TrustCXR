from __future__ import annotations

import json
from pathlib import Path

from scripts.reporting.run_stage18a_grounded_report_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reporting/stage18a_grounded_report_data_readiness.json"


def test_stage18a_enforces_statement_grounding_and_omission() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["grounding"]["statement_requires_structured_source"]
    assert result["grounding"]["statement_requires_evidence_code"]
    assert result["grounding"]["unsupported_behavior"] == ["OMIT", "EXPLICIT_UNCERTAINTY"]
    assert (
        "classifier_negation_from_localization_absence" in result["statement_policy"]["must_omit"]
    )


def test_stage18a_keeps_report_dataset_separate_and_test_locked() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert not result["report_dataset_audit"]["identity_aligned_with_model_cohorts"]
    assert result["report_dataset_audit"]["status"] == "WITHHELD_PATIENT_IDENTITY_UNRESOLVED"
    assert result["locked_test_records_accessed"] == 0
    assert not result["report_generation_performed"]
