from __future__ import annotations

import numpy as np

from trustcxr.segmentation.stage8d_comparison import (
    aggregate_metrics,
    interval_summary,
    patient_aggregate,
    select_candidate,
)


def test_aggregate_metrics_returns_expected_dice() -> None:
    counts = {
        "tp": np.array([[8.0, 8.0, 8.0]]),
        "fp": np.array([[2.0, 2.0, 2.0]]),
        "fn": np.array([[2.0, 2.0, 2.0]]),
        "tn": np.array([[88.0, 88.0, 88.0]]),
    }

    metrics = aggregate_metrics(counts)

    assert np.isclose(metrics["macro_dice"], 0.8)
    assert np.isclose(metrics["macro_iou"], 2.0 / 3.0)


def test_patient_aggregate_preserves_global_counts() -> None:
    patient_ids = ["p1", "p1", "p2"]
    counts = {
        name: np.arange(9, dtype=np.float64).reshape(3, 3) for name in ("tp", "fp", "fn", "tn")
    }

    patients, aggregated = patient_aggregate(patient_ids, counts)

    assert patients == ["p1", "p2"]

    for name in counts:
        assert np.array_equal(
            aggregated[name].sum(axis=0),
            counts[name].sum(axis=0),
        )


def test_select_candidate_prefers_supported_continuation() -> None:
    selected, basis = select_candidate(
        point_delta=0.001,
        confidence_interval_lower=0.0002,
        confidence_interval_upper=0.0018,
        minimum_improvement=0.0001,
        coverage_complete=True,
    )

    assert selected == "STAGE8C_CONTINUATION"
    assert basis == "PAIRED_PATIENT_BOOTSTRAP_SUPPORTS_STAGE8C"


def test_select_candidate_keeps_baseline_when_delta_is_negative() -> None:
    selected, _ = select_candidate(
        point_delta=-0.001,
        confidence_interval_lower=-0.002,
        confidence_interval_upper=-0.0001,
        minimum_improvement=0.0001,
        coverage_complete=True,
    )

    assert selected == "STAGE8B_BASELINE"


def test_interval_summary_reports_probability() -> None:
    values = np.array([-1.0, 1.0, 2.0, 3.0])
    summary = interval_summary(values, confidence=0.95)

    assert summary["probability_stage8c_greater"] == 0.75
    assert summary["lower"] <= summary["median"] <= summary["upper"]
