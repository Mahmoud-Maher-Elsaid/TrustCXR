"""Tests for generated dataset adapter contracts."""

from __future__ import annotations

from trustcxr.data.adapters import (
    DatasetAdapterContract,
    IdentityRule,
    canonicalize_identifier,
    resolve_raw_identifier,
    resolve_record_identity,
)


def test_canonicalize_identifier_is_stable_and_private() -> None:
    first = canonicalize_identifier(
        dataset_id="sample",
        namespace="patient",
        value="patient-123",
    )
    second = canonicalize_identifier(
        dataset_id="sample",
        namespace="patient",
        value="patient-123",
    )

    assert first == second
    assert "patient-123" not in first


def test_resolve_raw_identifier_from_metadata() -> None:
    value = resolve_raw_identifier(
        rule=IdentityRule(
            source="METADATA_COLUMN",
            confidence="HIGH",
            column="patient_id",
        ),
        metadata_row={
            "patient_id": "abc",
        },
    )

    assert value == "abc"


def test_resolve_raw_identifier_from_filename() -> None:
    value = resolve_raw_identifier(
        rule=IdentityRule(
            source="FILENAME_REGEX",
            confidence="MEDIUM",
            regex=r"^([0-9]{8})_",
        ),
        relative_path="images/00000001_002.png",
    )

    assert value == "00000001"


def test_resolve_record_identity() -> None:
    contract = DatasetAdapterContract(
        dataset_id="sample",
        folder="sample",
        adapter_status="READY_TO_IMPLEMENT",
        patient_split_ready=True,
        patient_rule=IdentityRule(
            source="METADATA_COLUMN",
            confidence="HIGH",
            column="patient_id",
        ),
        study_rule=IdentityRule(
            source="METADATA_COLUMN",
            confidence="HIGH",
            column="study_id",
        ),
        image_rule=IdentityRule(
            source="RELATIVE_PATH",
            confidence="HIGH",
        ),
    )

    identity = resolve_record_identity(
        contract=contract,
        metadata_row={
            "patient_id": "p1",
            "study_id": "s1",
        },
        relative_path="images/i1.png",
    )

    assert identity["patient_id"] is not None
    assert identity["study_id"] is not None
    assert identity["image_id"] is not None
    assert len(set(identity.values())) == 3
