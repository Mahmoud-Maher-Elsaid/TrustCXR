from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from scripts.fusion.run_stage11j_shared_validation_prediction_coverage import (
    missing_identifiers,
    shared_validation_patient_map,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage11j_selects_only_missing_shared_identifiers() -> None:
    assert missing_identifiers(["a", "b", "c"], {"a", "other"}) == ["b", "c"]
    with pytest.raises(RuntimeError):
        missing_identifiers(["a", "a"], set())


def test_stage11j_preserves_frozen_validation_contract() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11j_shared_validation_prediction_coverage.json").read_text()
    )
    stage11i = {
        "gate": "HOLD_FOR_STAGE_11J_SHARED_VALIDATION_COVERAGE_PREPARATION",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "new_stage9_predictions_generated": False,
    }
    validate_contract(config, stage11i)
    assert config["evaluation_split"] == "validation"
    assert config["expected_missing_predictions"] == 89
    assert config["training_permitted"] is False
    assert config["threshold_tuning_permitted"] is False
    broken = dict(stage11i, locked_test_records_accessed=1)
    with pytest.raises(RuntimeError):
        validate_contract(config, broken)


def test_stage11j_patient_mapping_reads_validation_only(tmp_path: Path) -> None:
    database = tmp_path / "cohort.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records (nih_image_id TEXT, nih_patient_id TEXT, split TEXT)")
    connection.executemany(
        "INSERT INTO records VALUES (?, ?, ?)",
        [
            ("validation-image", "validation-patient", "validation"),
            ("test-image", "test-patient", "test"),
        ],
    )
    connection.commit()
    connection.close()
    assert shared_validation_patient_map(database) == {"validation-image": "validation-patient"}
