from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch

from trustcxr.segmentation.stage8e_final_evaluation import (
    aggregate_metrics,
    counts_from_predictions,
    deterministic_overlay_ids,
    leakage_count,
    load_thresholds,
)


def test_counts_from_predictions_uses_per_organ_thresholds() -> None:
    probabilities = torch.tensor(
        [[[[0.8, 0.1]], [[0.6, 0.4]], [[0.3, 0.9]]]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float32,
    )
    thresholds = torch.tensor([0.5, 0.7, 0.8], dtype=torch.float32)

    tp, fp, fn, tn = counts_from_predictions(
        probabilities,
        targets,
        thresholds,
    )

    assert tp.tolist() == [[1, 0, 1]]
    assert fp.tolist() == [[0, 0, 0]]
    assert fn.tolist() == [[0, 1, 0]]
    assert tn.tolist() == [[1, 1, 1]]


def test_aggregate_metrics_returns_perfect_values() -> None:
    rows = [
        (
            "image.png",
            "patient-1",
            10,
            0,
            0,
            90,
            10,
            0,
            0,
            90,
            10,
            0,
            0,
            90,
        )
    ]

    metrics = aggregate_metrics(rows)

    assert np.isclose(metrics["macro_dice"], 1.0)
    assert np.isclose(metrics["macro_iou"], 1.0)


def test_deterministic_overlay_ids_is_stable_and_bounded() -> None:
    identifiers = [f"image-{index}" for index in range(50)]

    first = deterministic_overlay_ids(identifiers, 8)
    second = deterministic_overlay_ids(identifiers, 8)

    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8


def test_load_thresholds_accepts_mapping(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        '{"left_lung": 0.4, "right_lung": 0.5, "heart": 0.6}',
        encoding="utf-8",
    )

    thresholds = load_thresholds(path)

    assert thresholds == {
        "left_lung": 0.4,
        "right_lung": 0.5,
        "heart": 0.6,
    }


def test_leakage_count_detects_overlapping_patient(tmp_path: Path) -> None:
    database_path = tmp_path / "records.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE records (split TEXT, patient_id TEXT)")
    connection.executemany(
        "INSERT INTO records VALUES (?, ?)",
        [
            ("train", "patient-1"),
            ("validation", "patient-2"),
            ("test", "patient-1"),
        ],
    )
    connection.commit()
    connection.close()

    assert leakage_count(database_path) == 1
