"""Tests for deterministic patient-level split utilities."""

from __future__ import annotations

from trustcxr.data.safe_splits import (
    deterministic_bucket,
    deterministic_split,
    validate_patient_disjointness,
)


def test_deterministic_bucket_is_stable() -> None:
    assert deterministic_bucket("patient-a") == deterministic_bucket("patient-a")


def test_deterministic_split_is_stable() -> None:
    assert deterministic_split("patient-a") == deterministic_split("patient-a")
    assert deterministic_split("patient-a") in {"train", "validation", "test"}


def test_validate_patient_disjointness_accepts_safe_records() -> None:
    result = validate_patient_disjointness(
        [
            {"patient_id": "a", "split": "train"},
            {"patient_id": "a", "split": "train"},
            {"patient_id": "b", "split": "test"},
        ]
    )
    assert result["violation_count"] == 0


def test_validate_patient_disjointness_detects_leakage() -> None:
    result = validate_patient_disjointness(
        [
            {"patient_id": "a", "split": "train"},
            {"patient_id": "a", "split": "validation"},
        ]
    )
    assert result["violation_count"] == 1
