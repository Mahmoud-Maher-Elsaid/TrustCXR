from __future__ import annotations

from pathlib import Path

import numpy as np

from trustcxr.classification.dataset import NIH_LABELS, NIHRecord
from trustcxr.spatial.stage7e import (
    compute_spatial_metrics,
    normalize_affinity_map,
    pearson_correlation,
    select_audit_records,
)


def make_record(
    index: int,
    label: str | None,
    patient_id: str,
) -> NIHRecord:
    labels = () if label is None else (label,)
    return NIHRecord(
        image_name=f"image_{index:04d}.png",
        image_path=Path(f"image_{index:04d}.png"),
        patient_id=patient_id,
        labels=labels,
        split="test",
    )


def test_normalize_affinity_map_bounds_and_constant_input() -> None:
    normalized = normalize_affinity_map(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    constant = normalize_affinity_map(np.ones((3, 3), dtype=np.float64))
    assert np.array_equal(constant, np.zeros((3, 3), dtype=np.float64))


def test_pearson_correlation_identity_and_horizontal_change() -> None:
    array = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert pearson_correlation(array, array) == 1.0
    assert pearson_correlation(array, np.fliplr(array)) < 1.0


def test_spatial_metrics_detect_center_concentration() -> None:
    affinity = np.zeros((37, 37), dtype=np.float64)
    affinity[12:25, 12:25] = 1.0
    metrics = compute_spatial_metrics(
        affinity,
        top_fraction=0.1,
        center_height_fraction=0.8,
        center_width_fraction=0.6,
        border_width_patches=4,
    )
    assert metrics["center_border_density_ratio"] > 1.0
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0
    assert 0.0 <= metrics["top_fraction_concentration"] <= 1.0


def test_audit_selection_is_deterministic_unique_and_complete() -> None:
    records: list[NIHRecord] = []
    index = 0
    for label in NIH_LABELS:
        for repetition in range(3):
            records.append(
                make_record(
                    index,
                    label,
                    f"patient_{label}_{repetition}",
                )
            )
            index += 1
    for repetition in range(2):
        records.append(make_record(index, None, f"normal_patient_{repetition}"))
        index += 1

    first = select_audit_records(
        records,
        positive_images_per_label=2,
        no_finding_images=2,
        seed=42,
    )
    second = select_audit_records(
        records,
        positive_images_per_label=2,
        no_finding_images=2,
        seed=42,
    )

    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len(first) == len(NIH_LABELS) * 2 + 2
    assert len({item.record.patient_id for item in first}) == len(first)
    assert {item.target_label for item in first} == {
        *NIH_LABELS,
        "No Finding",
    }
