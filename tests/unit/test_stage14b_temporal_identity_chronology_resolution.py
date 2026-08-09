from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from scripts.temporal.run_stage14b_temporal_identity_chronology_resolution import (
    validate_stage13b_identity,
)


def test_stage14b_forbids_heuristic_chronology_and_data_access() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            root / "configs/temporal/stage14b_temporal_identity_chronology_resolution.json"
        ).read_text()
    )
    assert config["authoritative_chronology_evidence"] == []
    assert config["study_directory_number_chronology_permitted"] is False
    assert config["row_order_chronology_permitted"] is False
    assert config["file_timestamp_chronology_permitted"] is False
    assert config["heuristic_pairing_permitted"] is False
    assert config["locked_test_metadata_access_permitted"] is False
    assert config["locked_test_pixel_access_permitted"] is False
    assert config["pair_construction_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage14b_accepts_exact_frozen_stage13b_gate_and_index(tmp_path: Path) -> None:
    evidence = {
        "stage": "13B",
        "status": "PASSED_STUDY_IDENTITY_RESOLUTION",
        "gate": "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN",
        "development_manifest_records": 1,
        "resolved_records": 1,
        "unresolved_records": 0,
        "missing_explicit_study_structure": 0,
        "patient_identity_used_as_study_identity": False,
        "patient_leakage_violations": 0,
        "study_split_violations": 0,
        "locked_test_records_accessed": 0,
        "image_records_accessed": 0,
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    source_config = tmp_path / "stage13b.json"
    source_config.write_text("{}")
    index = tmp_path / "identity.sqlite"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE study_records (record_key_hash TEXT PRIMARY KEY, "
        "patient_key_hash TEXT, study_key_hash TEXT, split TEXT, view_label TEXT, "
        "identity_evidence TEXT)"
    )
    connection.execute(
        "INSERT INTO study_records VALUES "
        "('record', 'patient', 'study', 'train', 'AP', 'explicit_path')"
    )
    connection.commit()
    connection.close()
    config = {
        "stage13b_evidence": evidence_path.name,
        "stage13b_evidence_sha256": file_sha256(evidence_path),
        "stage13b_config": source_config.name,
        "stage13b_config_sha256": file_sha256(source_config),
        "stage13b_index": index.name,
        "stage13b_index_sha256": file_sha256(index),
        "stage13b_expected_status": "PASSED_STUDY_IDENTITY_RESOLUTION",
        "stage13b_expected_gate": "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN",
        "stage13b_index_table": "study_records",
        "stage13b_required_index_columns": [
            "record_key_hash",
            "patient_key_hash",
            "study_key_hash",
            "split",
            "view_label",
            "identity_evidence",
        ],
    }
    result = validate_stage13b_identity(config, tmp_path, evidence)
    assert result["indexed_development_records"] == 1


def test_stage14b_rejects_the_old_mismatched_gate(tmp_path: Path) -> None:
    evidence = {
        "stage": "13B",
        "status": "PASSED_STUDY_IDENTITY_RESOLUTION",
        "gate": "GO_FOR_STAGE_13C_PATIENT_SAFE_PAIR_DESIGN",
    }
    config = {
        "stage13b_evidence": "missing",
        "stage13b_evidence_sha256": "missing",
        "stage13b_config": "missing",
        "stage13b_config_sha256": "missing",
        "stage13b_index": "missing",
        "stage13b_index_sha256": "missing",
        "stage13b_expected_status": "PASSED_STUDY_IDENTITY_RESOLUTION",
        "stage13b_expected_gate": "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN",
    }
    with pytest.raises(RuntimeError, match="evidence is incompatible"):
        validate_stage13b_identity(config, tmp_path, evidence)
