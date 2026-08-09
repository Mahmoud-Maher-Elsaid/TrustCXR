from __future__ import annotations

import json
from pathlib import Path

from scripts.verification.run_stage19c_synthetic_verifier_implementation_validation import (
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/verification/stage19c_synthetic_verifier_implementation_validation.json"


def test_stage19c_all_synthetic_fixtures_pass() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["fixtures_passed"] == 17
    assert result["fixtures_failed"] == 0
    assert result["fixture_results"]["proxy_anatomical_agreement"] == "PARTIALLY_VERIFIED"
    assert (
        result["fixture_results"]["positive_localization_from_proxy"]
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )


def test_stage19c_safety_evidence_is_zero_use() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["real_patient_reports_used"] == 0
    assert result["patient_identifiers_used"] == 0
    assert not result["language_model_used"]
    assert not result["image_inference_performed"]
    assert not result["training_performed"]
    assert result["locked_test_records_accessed"] == 0
