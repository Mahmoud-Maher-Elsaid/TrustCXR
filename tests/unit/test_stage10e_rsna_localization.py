from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import torch

from trustcxr.detection.stage10e_rsna import (
    average_precision_50,
    load_split_patients,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads(
        (ROOT / "configs/localization/stage10e_rsna_localization_baseline.json").read_text()
    )


def test_stage10e_contract_is_rsna_validation_only_and_test_locked() -> None:
    value = config()
    validate_contract(value)
    assert value["dataset"] == "RSNA_Pneumonia"
    assert value["selection"]["final_test_images_accessed"] == 0
    assert set(value["withheld_datasets"]) == {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }


def test_split_loader_refuses_final_test_split(tmp_path: Path) -> None:
    index = tmp_path / "split.sqlite"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE split_records (image_hash TEXT, patient_hash TEXT, "
        "study_hash TEXT, split TEXT)"
    )
    connection.close()
    with pytest.raises(ValueError, match="only train and validation"):
        load_split_patients(index, "test")


def test_ap50_perfect_detection() -> None:
    predictions = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9])}]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}]
    assert average_precision_50(predictions, targets) == 1.0
