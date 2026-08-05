from __future__ import annotations

from trustcxr.integration.stage9a_cohort import (
    STANDARD_LABELS,
    normalized_header,
    parse_labels,
    select_column,
)


def test_parse_labels_creates_expected_multihot_vector() -> None:
    vector = parse_labels("Cardiomegaly|Effusion")

    assert len(vector) == 14
    assert vector[STANDARD_LABELS.index("Cardiomegaly")] == 1
    assert vector[STANDARD_LABELS.index("Effusion")] == 1
    assert sum(vector) == 2


def test_no_finding_creates_all_zero_vector() -> None:
    assert parse_labels("No Finding") == (0,) * 14


def test_header_normalization_is_stable() -> None:
    assert normalized_header("Finding Labels") == "finding_labels"
    assert normalized_header("Image-Index") == "image_index"


def test_select_column_supports_normalized_names() -> None:
    fields = ["Image Index", "Finding Labels", "Patient ID"]

    assert select_column(fields, ["image_index"]) == "Image Index"
    assert select_column(fields, ["finding_labels"]) == "Finding Labels"


def test_standard_label_contract_is_unique() -> None:
    assert len(STANDARD_LABELS) == 14
    assert len(set(STANDARD_LABELS)) == 14
    assert "Pleural_Thickening" in STANDARD_LABELS
