"""Tests for the final data readiness gate."""

from __future__ import annotations

from trustcxr.data.readiness_gate import (
    build_summary,
    validate_selection,
)


def sample_selection() -> dict:
    """Return a minimal valid selection with ten datasets."""
    datasets = []

    for index in range(10):
        ready = index < 4
        datasets.append(
            {
                "id": f"dataset_{index}",
                "name": f"Dataset {index}",
                "decision": ("TRAINING_READY" if ready else "WITHHELD_IDENTITY"),
                "patient_level_ready": ready,
                "approved_tasks": (["classification"] if ready else []),
                "role": "test",
            }
        )

    return {
        "split_policy": {
            "allow_image_level_random_split": False,
        },
        "datasets": datasets,
        "training_sequence": [
            {
                "order": 1,
                "stage": "Stage 5",
                "task": "quality",
                "dataset_id": "dataset_0",
                "model": "EfficientNet-B0",
                "input_size": 224,
                "status": "GO",
                "fallback_model": "MobileNetV3-Small",
            }
        ],
        "first_training_stage": {
            "stage": "Stage 5",
            "task": "quality",
            "dataset_id": "dataset_0",
            "model": "EfficientNet-B0",
            "gate": "APPROVED",
        },
    }


def test_valid_selection_passes() -> None:
    validate_selection(sample_selection())


def test_image_level_random_split_is_rejected() -> None:
    selection = sample_selection()
    selection["split_policy"]["allow_image_level_random_split"] = True

    try:
        validate_selection(selection)
    except ValueError as error:
        assert "Image-level random splitting" in str(error)
    else:
        raise AssertionError("Expected unsafe split policy to fail.")


def test_withheld_dataset_cannot_have_approved_tasks() -> None:
    selection = sample_selection()
    selection["datasets"][9]["approved_tasks"] = ["segmentation"]

    try:
        validate_selection(selection)
    except ValueError as error:
        assert "Withheld dataset has approved tasks" in str(error)
    else:
        raise AssertionError("Expected withheld dataset validation to fail.")


def test_summary_has_zero_patient_leakage() -> None:
    summary = build_summary(
        selection=sample_selection(),
        stage4_2={
            "total_leakage_violations": 0,
        },
        stage4_3={
            "total_leakage_violations": 0,
            "overall_patient_level_complete_count": 4,
        },
    )

    assert summary["status"] == "PASSED"
    assert summary["overall_gate"] == "GO_FOR_STAGE_5"
    assert summary["patient_leakage_violations"] == 0
    assert summary["training_ready_dataset_count"] == 4
    assert summary["safely_withheld_dataset_count"] == 6


def test_summary_rejects_previous_leakage() -> None:
    try:
        build_summary(
            selection=sample_selection(),
            stage4_2={
                "total_leakage_violations": 1,
            },
            stage4_3={
                "total_leakage_violations": 0,
                "overall_patient_level_complete_count": 4,
            },
        )
    except ValueError as error:
        assert "Patient leakage violations" in str(error)
    else:
        raise AssertionError("Expected leakage validation to fail.")
