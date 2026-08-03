"""Tests for Stage 4 dataset structure validation."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from trustcxr.data.structure_validation import (
    DatasetSpec,
    infer_semantic_fields,
    normalize_column_name,
    validate_dataset_structure,
)


def build_spec(folder: str) -> DatasetSpec:
    """Create a minimal dataset specification for tests."""
    return DatasetSpec(
        dataset_id="sample",
        folder=folder,
        name="Sample Dataset",
        primary_tasks=("classification",),
        required_for_core=True,
    )


def test_column_normalization_and_semantic_inference() -> None:
    assert normalize_column_name("Patient ID") == "patient_id"
    inferred = infer_semantic_fields(
        ["Patient ID", "StudyInstanceUID", "Image Index", "Finding Labels"]
    )
    assert "patient_id" in inferred
    assert "study_id" in inferred
    assert "image_id" in inferred
    assert "labels" in inferred


def test_ready_dataset_detects_metadata_and_image_linkage(tmp_path: Path) -> None:
    dataset_root = tmp_path / "sample"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    Image.new("L", (16, 16)).save(image_root / "image_001.png")

    metadata_path = dataset_root / "labels.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Patient ID", "Image Index", "Finding Labels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Patient ID": "p1",
                "Image Index": "image_001.png",
                "Finding Labels": "Normal",
            }
        )

    result, details = validate_dataset_structure(
        tmp_path,
        build_spec("sample"),
        metadata_file_limit=10,
        row_limit=100,
    )

    assert result["status"] == "READY_FOR_CANONICAL_MAPPING"
    assert result["image_file_count"] == 1
    assert result["metadata_file_count"] == 1
    assert result["patient_split_readiness"] == "DIRECT_PATIENT_LEVEL_SPLIT"
    assert result["image_reference_match_rate"] == 1.0
    assert details


def test_archive_only_dataset_requires_extraction(tmp_path: Path) -> None:
    dataset_root = tmp_path / "sample"
    dataset_root.mkdir()
    (dataset_root / "dataset.zip").write_bytes(b"not-a-real-archive")

    result, _ = validate_dataset_structure(
        tmp_path,
        build_spec("sample"),
        metadata_file_limit=10,
        row_limit=100,
    )

    assert result["status"] == "NEEDS_EXTRACTION"
    assert result["archive_file_count"] == 1
    assert result["image_file_count"] == 0


def test_missing_dataset_is_reported_without_crashing(tmp_path: Path) -> None:
    result, details = validate_dataset_structure(
        tmp_path,
        build_spec("missing"),
        metadata_file_limit=10,
        row_limit=100,
    )

    assert result["status"] == "MISSING"
    assert result["patient_split_readiness"] == "NOT_READY"
    assert details == []
