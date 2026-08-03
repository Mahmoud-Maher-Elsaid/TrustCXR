"""Tests for Stage 4.1 dataset identity mapping discovery."""

from __future__ import annotations

from trustcxr.data.identity_mapping import (
    build_identity_rule,
    classify_column,
    infer_filename_rule,
    normalize_column_name,
    sanitize_path_pattern,
)


def test_normalize_column_name() -> None:
    assert normalize_column_name("Patient ID") == "patient_id"
    assert normalize_column_name(" Study-Instance UID ") == ("study_instance_uid")


def test_classify_column_groups() -> None:
    assert "patient" in classify_column("patient_id")
    assert "study" in classify_column("StudyID")
    assert "image" in classify_column("file_path")
    assert "report" in classify_column("impression")
    assert "mask" in classify_column("EncodedPixels")


def test_sanitize_path_pattern_removes_identifiers() -> None:
    source = "patient123/study456/00000001_003.png"
    pattern = sanitize_path_pattern(source)

    assert "123" not in pattern
    assert "456" not in pattern
    assert "00000001" not in pattern
    assert "<N>" in pattern


def test_infer_filename_rule_for_repeated_patient_prefix() -> None:
    filenames = [
        "00000001_001.png",
        "00000001_002.png",
        "00000002_001.png",
        "00000002_002.png",
    ]
    result = infer_filename_rule(filenames)

    assert result["strategy"] == "NUMERIC_PREFIX_BEFORE_SEPARATOR"
    assert result["confidence"] == "MEDIUM"
    assert result["match_ratio"] == 1.0


def test_build_identity_rule_prefers_metadata() -> None:
    profiles = [
        {
            "file_id": "file",
            "semantic_columns": {
                "patient": ["patient_id"],
            },
            "sample_value_profiles": {
                "patient_id": {
                    "sample_nonempty_count": 10,
                    "sample_unique_hash_count": 8,
                }
            },
        }
    ]
    rule = build_identity_rule(
        metadata_profiles=profiles,
        dicom_profile={
            "tag_presence": {
                "PatientID": {
                    "presence_ratio": 1.0,
                }
            }
        },
        filename_rule={
            "confidence": "MEDIUM",
        },
        path_keyword_detected=True,
        semantic_group="patient",
    )

    assert rule["source"] == "METADATA_COLUMN"
    assert rule["column"] == "patient_id"


def test_build_identity_rule_uses_dicom_tag() -> None:
    rule = build_identity_rule(
        metadata_profiles=[],
        dicom_profile={
            "tag_presence": {
                "PatientID": {
                    "presence_ratio": 0.9,
                }
            }
        },
        filename_rule={
            "confidence": "NONE",
        },
        path_keyword_detected=False,
        semantic_group="patient",
    )

    assert rule["source"] == "DICOM_TAG"
    assert rule["tag"] == "PatientID"
