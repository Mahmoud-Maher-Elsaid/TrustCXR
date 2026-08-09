from __future__ import annotations

import json
from pathlib import Path

from scripts.reliability.run_stage16e_reliability_acceptance_decision import decide

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reliability/stage16e_reliability_acceptance_decision.json"


def test_stage16e_decisions_preserve_mixed_evidence() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    decisions = {row["model"]: row for row in result["decisions"]}
    assert decisions["stage9_original"]["calibration"] == "ACCEPTED_VALIDATION_ONLY"
    assert (
        decisions["stage13_frontal_only"]["calibration"]
        == "PARTIALLY_ACCEPTED_MIXED_VALIDATION_EVIDENCE"
    )
    assert (
        decisions["stage13_frontal_only"]["selective_prediction"]
        == "NOT_ACCEPTED_NO_EVALUATION_RISK_REDUCTION"
    )
    assert result["ood_status"] == "WITHHELD_NO_GOVERNED_OOD_COHORT"
    assert result["locked_test_records_accessed"] == 0


def test_stage16e_thresholds_are_exactly_frozen() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    decisions = {row["model"]: row for row in result["decisions"]}
    assert decisions["stage9_original"]["abstention_threshold"] == 0.6917239984806322
    assert decisions["stage13_frontal_only"]["abstention_threshold"] == 0.6928039993196862
