"""Tests for concrete dataset adapter utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from trustcxr.data.concrete_adapters import (
    AdapterSpec,
    build_dataset_manifest,
    build_metadata_index,
    discover_dataset_files,
    normalize_image_key,
    normalize_mask_key,
)


def make_spec(folder: str) -> AdapterSpec:
    """Create a minimal NIH-like adapter specification."""
    return AdapterSpec(
        dataset_id="nih_chestxray14",
        folder=folder,
        name="Test Dataset",
        adapter_kind="classification",
        metadata_patterns=("*.csv",),
        image_key_columns=("Image Index",),
        patient_columns=("Patient ID",),
        study_columns=("Follow-up #",),
        label_columns=("Finding Labels",),
        label_mode="single_column",
        identity_policy="VERIFIED_METADATA_PATIENT",
        join_key_mode="basename",
    )


def test_normalize_image_and_mask_keys() -> None:
    assert normalize_image_key("folder/IMAGE_001.PNG") == "image001"
    assert normalize_mask_key("IMAGE_001_mask.png") == "image001"


def test_discover_dataset_files_separates_masks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    Image.new("L", (8, 8)).save(root / "images" / "a.png")
    Image.new("L", (8, 8)).save(root / "masks" / "a_mask.png")
    inventory = discover_dataset_files(root, make_spec("dataset"))
    assert len(inventory.images) == 1
    assert len(inventory.masks) == 1


def test_build_metadata_index_aggregates_rows(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Image Index", "Patient ID", "Finding Labels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Image Index": "a.png",
                "Patient ID": "p1",
                "Finding Labels": "Mass|Effusion",
            }
        )
    index, profile = build_metadata_index([path], make_spec("dataset"))
    assert index["a"].patient_value == "p1"
    assert index["a"].labels == {"Mass", "Effusion"}
    assert profile["metadata_row_count"] == 1


def test_build_dataset_manifest_creates_safe_split(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "dataset"
    dataset_root.mkdir(parents=True)
    Image.new("L", (8, 8)).save(dataset_root / "a.png")
    metadata_path = dataset_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Image Index", "Patient ID", "Finding Labels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Image Index": "a.png",
                "Patient ID": "patient-1",
                "Finding Labels": "Mass",
            }
        )
    result = build_dataset_manifest(
        data_root=data_root,
        local_root=tmp_path / "local",
        spec=make_spec("dataset"),
    )
    assert result["record_count"] == 1
    assert result["patient_resolution_rate"] == 1.0
    assert result["safe_split_status"] == "PATIENT_LEVEL_COMPLETE"
    manifest_path = tmp_path / "local" / "manifests" / "nih_chestxray14.jsonl"
    record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert record["patient_id"].startswith("nih_chestxray14:patient:")
    assert record["split"] in {"train", "validation", "test"}
