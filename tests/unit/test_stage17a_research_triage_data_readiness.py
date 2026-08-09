from __future__ import annotations

import json
from pathlib import Path

from scripts.triage.run_stage17a_research_triage_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/triage/stage17a_research_triage_data_readiness.json"


def test_stage17a_preserves_withholdings_and_safety() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["status"] == "PASSED_RESEARCH_TRIAGE_DATA_READINESS_WITH_WITHHOLDINGS"
    assert "ood_detection" in result["mandatory_withholdings"]
    assert "stage13_selective_prediction" in result["mandatory_withholdings"]
    assert result["reason_code_required"]
    assert not result["treatment_advice_permitted"]
    assert result["locked_test_records_accessed"] == 0
