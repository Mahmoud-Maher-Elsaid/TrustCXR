"""Tests for Stage 4.3 container and identity resolution."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from trustcxr.data.stage4_3_resolution import (
    assign_split,
    choose_column,
    detect_container_format,
    normalize_image_key,
    pairing_key,
    resolve_paired_image_dataset,
    stable_hash,
    write_jsonl,
)


def test_stable_hash_is_deterministic_and_namespaced() -> None:
    first = stable_hash("patient", "example")
    second = stable_hash("patient", "example")
    different = stable_hash("study", "example")

    assert first == second
    assert first != different
    assert len(first) == 24


def test_choose_column_matches_normalized_candidates() -> None:
    columns = ["Image Index", "Patient ID", "Finding Labels"]

    assert choose_column(columns, ("image_index",)) == "Image Index"
    assert choose_column(columns, ("patientid",)) == "Patient ID"
    assert choose_column(columns, ("missing",)) is None


def test_assign_split_is_stable() -> None:
    patient_id = stable_hash("patient", "abc")

    assert assign_split(patient_id) == assign_split(patient_id)
    assert assign_split(patient_id) in {
        "train",
        "validation",
        "test",
    }


def test_normalize_image_key_removes_path_and_extension() -> None:
    assert normalize_image_key(r"folder\Image_001.DCM") == "image_001"
    assert normalize_image_key("folder/image_002.png") == "image_002"


def test_detect_container_format_uses_suffix_and_magic(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")

    hdf_path = tmp_path / "sample.bin"
    hdf_path.write_bytes(b"\x89HDF\r\n\x1a\nrest")

    assert detect_container_format(csv_path) == "csv"
    assert detect_container_format(hdf_path) == "hdf5"


def test_write_jsonl_writes_one_record_per_line(
    tmp_path: Path,
) -> None:
    output = tmp_path / "records.jsonl"
    count = write_jsonl(
        output,
        (
            {"id": 1},
            {"id": 2},
        ),
    )

    lines = output.read_text(encoding="utf-8").splitlines()

    assert count == 2
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == 1


def test_pairing_key_removes_mask_suffix() -> None:
    assert pairing_key(Path("case_001.png")) == "case001"
    assert pairing_key(Path("case_001_mask.png")) == "case001"


def test_paired_dataset_is_withheld_without_patient_identity(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "sample"
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)

    Image.new("L", (8, 8)).save(image_root / "case_1.png")
    Image.new("L", (8, 8)).save(mask_root / "case_1_mask.png")

    local_root = tmp_path / "local"
    result = resolve_paired_image_dataset(
        data_root=data_root,
        folder="sample",
        dataset_id="sample",
        local_manifest_root=local_root,
    )

    assert result["records_written"] == 1
    assert result["matched_pairs"] == 1
    assert result["patient_identity_rate"] == 0.0
    assert result["split_safety"].startswith("WITHHELD")
