from __future__ import annotations

import json
from pathlib import Path

from scripts.decision.run_stage20a_accept_revise_defer_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/decision/stage20a_accept_revise_defer_data_readiness.json"


def test_stage20a_is_readiness_only() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["status"] == "PASSED_ACCEPT_REVISE_DEFER_DATA_READINESS_WITH_POLICY_HOLD"
    assert not result["policy_activated"]
    assert result["accept_meaning"].endswith("NOT_CLINICAL_APPROVAL")
    assert result["prospective_status_readiness"]["PARTIALLY_VERIFIED"] == "DEFER_NOT_ACCEPT"
    assert result["prospective_status_readiness"]["CONTRADICTED"] == "DEFER"


def test_stage20a_does_no_prohibited_work() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert not result["next_stage_authorizes_language_model_work"]
    assert not result["language_model_used"]
    assert not result["training_performed"]
    assert not result["report_generation_performed"]
    assert not result["image_inference_performed"]
    assert result["locked_test_records_accessed"] == 0
    assert result["patient_identifiers_used"] == 0
