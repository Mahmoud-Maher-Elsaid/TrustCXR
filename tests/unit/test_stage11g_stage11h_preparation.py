from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from scripts.fusion.run_stage11h_record_level_fusion_evaluation import (
    shared_validation_rows,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage11h_reads_validation_rows_only(tmp_path: Path) -> None:
    database = tmp_path / "cohort.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE records (nih_image_id TEXT, rsna_sop_uid TEXT, "
        "nih_patient_id TEXT, split TEXT)"
    )
    connection.executemany(
        "INSERT INTO records VALUES (?, ?, ?, ?)",
        [("a", "ra", "p1", "validation"), ("b", "rb", "p2", "test")],
    )
    connection.commit()
    connection.close()
    assert shared_validation_rows(database) == [("a", "ra", "p1")]


def test_stage11h_preserves_partial_support_and_frozen_evaluations() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11h_record_level_fusion_evaluation.json").read_text()
    )
    stage11g = {
        "gate": "GO_FOR_STAGE_11H_RECORD_LEVEL_FUSION_EVALUATION_PREPARATION",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
    }
    validate_contract(config, stage11g)
    assert config["evaluation_split"] == "validation"
    assert config["localization_reliable_for_contradiction"] is False
    assert config["stage9_stage10_frozen_evaluations_may_be_modified"] is False
    broken = dict(stage11g, locked_test_records_accessed=1)
    with pytest.raises(RuntimeError):
        validate_contract(config, broken)
