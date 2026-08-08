from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from scripts.fusion.run_stage11f_shared_cohort_fusion_validation import (
    require_contract,
    validate_cohort,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage11f_validates_patient_safe_shared_cohort() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE records (nih_image_id TEXT PRIMARY KEY, rsna_sop_uid TEXT UNIQUE, "
        "nih_patient_id TEXT NOT NULL, rsna_patient_hash TEXT NOT NULL, split TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
        [("a", "ra", "p1", "h1", "train"), ("b", "rb", "p2", "h2", "validation")],
    )
    result = validate_cohort(connection)
    assert result["patient_split_violations"] == 0
    assert result["records"] == 2
    assert result["validation_records"] == 1


def test_stage11f_rejects_patient_split_leakage() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE records (nih_image_id TEXT PRIMARY KEY, rsna_sop_uid TEXT UNIQUE, "
        "nih_patient_id TEXT NOT NULL, rsna_patient_hash TEXT NOT NULL, split TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
        [("a", "ra", "p1", "h1", "train"), ("b", "rb", "p1", "h1", "validation")],
    )
    assert validate_cohort(connection)["patient_split_violations"] == 1


def test_stage11f_contract_preserves_frozen_evidence_policy() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11f_shared_cohort_fusion_validation.json").read_text()
    )
    stage11e = {
        "gate": "GO_FOR_STAGE_11F_SHARED_COHORT_FUSION_VALIDATION",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "semantic_relation": config["required_semantic_relation"],
    }
    require_contract(config, stage11e)
    assert config["permitted_evidence_status"] == "PARTIALLY_SUPPORTED"
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    broken = dict(stage11e, patient_reassignments=1)
    with pytest.raises(RuntimeError):
        require_contract(config, broken)
