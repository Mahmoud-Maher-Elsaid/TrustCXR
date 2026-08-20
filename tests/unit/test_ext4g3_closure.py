from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/research_extensions/ext4g/EXT4G3_GEMMA_DEVELOPMENT_EVALUATION_REPORT.json"
LEDGER = ROOT / "artifacts/research_extensions/ext4g3/20260820T014042Z_395f95c3/run_ledger.json"


def test_ext4g3_forensic_closure_is_mathematically_forced():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "EXT4G3_DEVELOPMENT_GATE_FAILED"
    assert report["decision"] == "NOT_SCIENTIFICALLY_SELECTED"
    assert report["cases_attempted"] == 24
    assert report["generate_call_count"] == 24
    assert report["contract_validity_rate"] == 21 / 24
    assert report["maximum_achievable_case_pass_rate"] == 21 / 24
    assert report["contract_validity_rate"] < 1.0
    assert report["maximum_achievable_case_pass_rate"] < 0.95
    assert report["evaluator_defect"] is False
    assert report["semantic_review_disposition"] == (
        "NOT_REQUIRED_FOR_SELECTION_AFTER_AUTOMATIC_GATE_FAILURE"
    )


def test_three_failed_cases_and_masks_are_preserved():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    failures = {item["case_id"]: item for item in report["failed_cases"]}
    assert set(failures) == {"ext4f_dev_010", "ext4f_dev_014", "ext4f_dev_015"}
    assert failures["ext4f_dev_010"]["failure_code"] == "EXT4F_REALIZATION_REQUIRED_SLOT_MISSING"
    assert failures["ext4f_dev_014"]["failure_code"] == "EXT4F_REALIZATION_REQUIRED_SLOT_MISSING"
    assert failures["ext4f_dev_015"]["failure_classification"] == "C_JSON_PARSE_FAILURE"
    assert all(item["attention_mask_present"] for item in failures.values())
    assert all(item["authority_mutations"] == 0 for item in failures.values())
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["generate_call_count"] == 24
    assert ledger["development_cases_accessed"] == 24
    assert ledger["frozen_final_cases_accessed"] == 0
    assert ledger["locked_test_accessed"] is False


def test_top_k_is_inert_checkpoint_metadata_only():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    warning = report["top_k_warning"]
    assert warning["classification"] == "INERT_WARNING_ONLY"
    assert warning["runtime_explicit_top_k"] is False
    assert warning["do_sample"] is False
    assert warning["protocol_deviation"] is False
