from __future__ import annotations

import json
from pathlib import Path

from scripts.verification.run_stage19d_verifier_acceptance_decision import decide

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/verification/stage19d_verifier_acceptance_decision.json"


def test_stage19d_acceptance_is_research_only_and_limited() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["status"] == "ACCEPTED_DETERMINISTIC_RESEARCH_ONLY_VERIFIER_CAPABILITY"
    assert result["stage11_maximum_support"] == "PARTIALLY_SUPPORTED"
    assert result["anatomical_proxy_maximum_status"] == "PARTIALLY_VERIFIED"
    assert result["designation"] == "DETERMINISTIC_RESEARCH_ONLY"


def test_stage19d_performs_no_prohibited_work() -> None:
    result = decide(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert not result["next_stage_authorizes_language_model_work"]
    assert not result["language_model_used"]
    assert not result["image_inference_performed"]
    assert not result["training_performed"]
    assert result["real_patient_reports_used"] == 0
    assert result["locked_test_records_accessed"] == 0
