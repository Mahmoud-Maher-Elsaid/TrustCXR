# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.ext4f5_benchmark import build_development_cases
from trustcxr.grounded_llm.ext4f_realization import validate_ext4f_realization_response

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/research_extensions/ext4f6/20260819T172740Z_99bbc896"


def test_final_report_records_irrecoverable_gate_failure():
    report = json.loads(
        (
            ROOT
            / "reports/research_extensions/ext4f/EXT4F6_DEVELOPMENT_CANDIDATE_EVALUATION_REPORT.json"
        ).read_text()
    )
    assert report["status"] == "EXT4F6_DEVELOPMENT_GATE_FAILED / NOT_SCIENTIFICALLY_SELECTED"
    assert report["realization_contract_validity_rate"] == pytest.approx(22 / 24)
    assert report["maximum_possible_case_pass_rate"] == pytest.approx(22 / 24)
    assert report["frozen_final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False
    assert report["ext4f7_authorized"] is False


def test_preserved_raw_failures_match_frozen_validator():
    cases = {case.case_id: case for case in build_development_cases()}
    expected = {
        "ext4f_dev_002": "EXT4F_REALIZATION_REQUIRED_SLOT_MISSING",
        "ext4f_dev_003": "EXT4F_REALIZATION_DUPLICATE_SLOT",
    }
    for case_id, error_code in expected.items():
        raw = json.loads((RUN / f"{case_id}_raw.txt").read_text())
        with pytest.raises(Exception, match=error_code):
            validate_ext4f_realization_response(raw, cases[case_id].request)


def test_no_selection_rescue_by_unreviewed_slots():
    report = json.loads(
        (
            ROOT
            / "reports/research_extensions/ext4f/EXT4F6_DEVELOPMENT_CANDIDATE_EVALUATION_REPORT.json"
        ).read_text()
    )
    assert report["semantic_review_disposition"] == (
        "NOT_REQUIRED_FOR_SELECTION_AFTER_MANDATORY_GATE_FAILURE"
    )
    assert report["semantic_review_values"].startswith("UNREVIEWED")
