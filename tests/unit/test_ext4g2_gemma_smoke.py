from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: E501

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research_extensions/run_ext4g2_gemma_synthetic_smoke.py"


def test_ext4g2_runner_is_new_case_and_frozen_identity():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'CASE_ID = "ext4g2_gemma_synthetic_001"' in text
    assert "research_case_ext4f4_realization_001" not in text
    assert "ext4f_dev_" not in text
    assert "model.generate(" in text
    assert text.count("model.generate(") == 1
    assert '"retry_count": 0' in text


def test_ext4g2_report_template_preserves_tripwires():
    report = json.loads(
        (
            ROOT
            / "reports/research_extensions/ext4g/EXT4G2_GEMMA3_SYNTHETIC_TECHNICAL_SMOKE_REPORT.json"
        ).read_text()
    )
    assert report["synthetic_case_id"] == "ext4g2_gemma_synthetic_001"
    assert report["development_cases_accessed"] == 0
    assert report["frozen_final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False
    assert report.get("generate_call_count", 0) == 1
