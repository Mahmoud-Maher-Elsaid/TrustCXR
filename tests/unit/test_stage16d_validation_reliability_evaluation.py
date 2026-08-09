from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trustcxr.reliability.stage16d_evaluation import (
    interval_row,
    risk_curve,
    select_threshold,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reliability/stage16d_validation_reliability_evaluation.json"


def test_stage16d_contract_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (
        config["contract_fingerprint"]
        == "3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e"
    )
    assert config["statistics"] == {
        "bootstrap_unit": "patient_cluster",
        "replicates": 2000,
        "confidence_level": 0.95,
        "seed": 20260809,
        "minimum_valid_replicates": 1900,
        "not_estimable_policy": "NOT_ESTIMABLE",
    }
    assert config["ood_status"] == "WITHHELD_NO_GOVERNED_OOD_COHORT"
    assert not config["locked_test_access_permitted"]
    assert not config["retraining_permitted"]


def test_abstention_selection_obeys_minimum_coverage() -> None:
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    masks = np.ones_like(targets)
    probabilities = np.array([[0.1], [0.9], [0.6], [0.4]])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.4])
    _, coverage, _ = select_threshold(targets, masks, probabilities, uncertainty, 0.8)
    assert coverage >= 0.8


def test_insufficient_bootstrap_support_is_not_estimable() -> None:
    row = interval_row("model", "metric", 0.5, [0.4, float("nan")], 2, 0.05)
    assert row["status"] == "NOT_ESTIMABLE"
    assert row["ci_low"] is None


def test_risk_curve_uses_supported_trapezoidal_integration() -> None:
    targets = np.array([[0.0], [1.0], [0.0], [1.0]])
    masks = np.ones_like(targets)
    probabilities = np.array([[0.1], [0.9], [0.2], [0.8]])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.4])

    rows, aurc = risk_curve(
        targets,
        masks,
        probabilities,
        uncertainty,
        ["finding"],
        [1.0, 0.5],
    )

    assert [row["coverage"] for row in rows] == [1.0, 0.5]
    assert aurc == pytest.approx(0.00875)
