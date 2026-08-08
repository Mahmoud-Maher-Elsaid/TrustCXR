from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.quality.run_stage12d_annotation_cohort_construction import (
    assign_patient_split,
    construct,
)

ROOT = Path(__file__).resolve().parents[2]


def write_source(root: Path) -> None:
    source = root / "data/archive/train.csv"
    source.parent.mkdir(parents=True)
    columns = ["Path", "Frontal/Lateral", "AP/PA", "Support Devices"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, view in enumerate((("Frontal", "AP"), ("", "")), start=1):
            patient = f"patient{index:05d}"
            writer.writerow(
                {
                    "Path": f"train/{patient}/study1/view1.jpg",
                    "Frontal/Lateral": view[0],
                    "AP/PA": view[1],
                    "Support Devices": "1.0",
                }
            )


def test_construction_uses_trusted_labels_and_creates_review_queue(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs/quality/stage12d_annotation_cohort_construction.json").read_text()
    )
    config["chexpert_root"] = "data"
    config["chexpert_csv_patterns"] = ["**/train.csv"]
    config["output_root"] = "output"
    write_source(tmp_path)
    stage12d = {
        "gate": "HOLD_FOR_STAGE_12D_ANNOTATION_COHORT_CONSTRUCTION",
        "protocol_version": "1.0.0",
    }
    summary = construct(config, stage12d, tmp_path)
    assert summary["images_accessed"] == 0
    assert summary["annotations_invented"] is False
    assert summary["locked_test_records_written"] == 0
    assert (tmp_path / "output/input_rejection_annotations.csv").read_text().count("\n") == 1


def test_construction_refuses_to_overwrite_review_work(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs/quality/stage12d_annotation_cohort_construction.json").read_text()
    )
    config["chexpert_root"] = "data"
    config["chexpert_csv_patterns"] = ["**/train.csv"]
    config["output_root"] = "output"
    write_source(tmp_path)
    stage12d = {
        "gate": "HOLD_FOR_STAGE_12D_ANNOTATION_COHORT_CONSTRUCTION",
        "protocol_version": "1.0.0",
    }
    construct(config, stage12d, tmp_path)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        construct(config, stage12d, tmp_path)


def test_patient_split_contract_has_locked_test_outcome() -> None:
    outcomes = {assign_patient_split(f"patient{index:05d}") for index in range(100)}
    assert outcomes == {"train", "validation", "test"}
