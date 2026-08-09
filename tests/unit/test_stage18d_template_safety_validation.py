from __future__ import annotations

import json
from pathlib import Path

from scripts.reporting.run_stage18d_template_safety_validation import run_suite, validate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reporting/stage18d_template_safety_validation.json"
CONTRACT = ROOT / "reports/stage18/stage18b_deterministic_grounded_report_contract_summary.json"


def test_all_adversarial_synthetic_fixtures_fail_closed() -> None:
    results = run_suite(json.loads(CONTRACT.read_text(encoding="utf-8")))
    assert all(row["observed"] == row["expected"] for row in results)
    assert {row["fixture"] for row in results} >= {
        "missing_provenance",
        "fake_evidence_code",
        "fake_defer_reason_code",
        "unsupported_source_stage",
        "free_text_injection",
        "severity",
        "temporal_change",
        "treatment",
        "patient_history",
        "ood",
        "device_localization",
        "clinical_quality",
        "positive_localization",
        "localization_negation",
        "clinical_certainty",
        "patient_identifier",
        "schema_mismatch",
        "ordering",
    }


def test_stage18d_uses_only_synthetic_fixtures() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["fixtures_failed"] == 0
    assert result["real_patient_identifiers_used"] == 0
    assert result["real_patient_reports_generated"] == 0
    assert not result["indiana_reports_used"]
    assert result["locked_test_records_accessed"] == 0
