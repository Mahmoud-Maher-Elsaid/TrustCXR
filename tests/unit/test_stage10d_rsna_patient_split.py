from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.localization.run_stage10d_rsna_patient_split import build_split_index

ROOT = Path(__file__).resolve().parents[2]


def test_stage10d_contract_is_rsna_only_and_test_locked() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10d_rsna_patient_split.json").read_text()
    )
    assert config["dataset"] == "RSNA_Pneumonia"
    assert config["training_permitted"] is False
    assert config["final_test_images_access_permitted"] is False
    assert set(config["withheld_datasets"]) == {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }


def test_stage10d_patient_assignments_have_zero_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    connection.execute(
        "CREATE TABLE identity_records (dataset TEXT, image_hash TEXT, patient_hash TEXT, "
        "study_hash TEXT, annotation_match INTEGER, source_split TEXT)"
    )
    connection.executemany(
        "INSERT INTO identity_records VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("RSNA_Pneumonia", "i1", "p1", "s1", 1, "train_source"),
            ("RSNA_Pneumonia", "i2", "p1", "s2", 1, "train_source"),
            ("RSNA_Pneumonia", "i3", "p2", "s3", 1, "train_source"),
        ],
    )
    connection.commit()
    connection.close()
    config = json.loads(
        (ROOT / "configs/localization/stage10d_rsna_patient_split.json").read_text()
    )
    result = build_split_index(config, source, tmp_path / "splits.sqlite")
    assert result["patient_leakage_violations"] == 0
    assert result["records"] == 3
    assert result["patients"] == 2
