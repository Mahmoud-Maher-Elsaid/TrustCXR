"""Tests for the read-only dataset audit."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from trustcxr.data.audit import (
    DatasetSpec,
    audit_dataset,
    load_catalog,
    normalized_extension,
    privacy_safe_identifier,
)


def build_specification(folder: str) -> DatasetSpec:
    """Build a test dataset specification."""
    return DatasetSpec(
        dataset_id="sample",
        folder=folder,
        name="Sample Dataset",
        primary_tasks=("classification",),
        required_for_core=True,
        license_status="REVIEW_REQUIRED",
    )


def test_normalized_extension_supports_compound_suffixes() -> None:
    assert normalized_extension("sample.NII.GZ") == ".nii.gz"
    assert normalized_extension("archive.tar.gz") == ".tar.gz"
    assert normalized_extension("image.PNG") == ".png"
    assert normalized_extension("README") == "<no_extension>"


def test_privacy_identifier_does_not_expose_source_path() -> None:
    source_path = "patient_123/study_456/image.png"
    identifier = privacy_safe_identifier(source_path)

    assert source_path not in identifier
    assert "patient_123" not in identifier
    assert len(identifier) == 16


def test_load_catalog_rejects_duplicate_dataset_ids(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "duplicate",
                        "folder": "one",
                        "name": "One",
                        "primary_tasks": ["classification"],
                        "required_for_core": True,
                        "license_status": "REVIEW_REQUIRED",
                    },
                    {
                        "id": "duplicate",
                        "folder": "two",
                        "name": "Two",
                        "primary_tasks": ["segmentation"],
                        "required_for_core": False,
                        "license_status": "REVIEW_REQUIRED",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        load_catalog(catalog_path)
    except ValueError as error:
        assert "duplicate dataset IDs" in str(error)
    else:
        raise AssertionError("Expected duplicate dataset IDs to fail.")


def test_audit_detects_image_and_zero_byte_file(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "sample"
    dataset_root.mkdir()

    image_path = dataset_root / "valid.png"
    Image.new("L", (16, 16)).save(image_path)

    zero_byte_path = dataset_root / "empty.csv"
    zero_byte_path.touch()

    result, issues = audit_dataset(
        tmp_path,
        build_specification("sample"),
        sample_limit=10,
    )

    assert result["file_count"] == 2
    assert result["zero_byte_files"] == 1
    assert result["extension_counts"][".png"] == 1
    assert result["extension_counts"][".csv"] == 1
    assert result["status"] == "READY_WITH_WARNINGS"
    assert issues
    assert all("valid.png" not in str(issue) for issue in issues)


def test_missing_dataset_is_reported_without_failure(
    tmp_path: Path,
) -> None:
    result, issues = audit_dataset(
        tmp_path,
        build_specification("missing"),
        sample_limit=10,
    )

    assert result["status"] == "MISSING"
    assert result["file_count"] == 0
    assert issues == []
