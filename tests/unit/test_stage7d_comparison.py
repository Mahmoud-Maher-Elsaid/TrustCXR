from __future__ import annotations

import numpy as np

from trustcxr.comparison.stage7d import (
    choose_models,
    difference_interpretation,
    weighted_rank_metrics,
    weighted_threshold_metrics,
)


def test_weighted_rank_metrics_match_known_values() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)

    auroc, auprc = weighted_rank_metrics(
        targets,
        scores,
        weights,
    )

    assert abs(auroc - 0.75) < 1e-12
    assert 0.0 <= auprc <= 1.0


def test_weighted_threshold_metrics_are_bounded() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.9, 0.8, 0.7], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float64)

    f1, sensitivity, specificity = weighted_threshold_metrics(
        targets,
        scores,
        0.5,
        weights,
    )

    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= sensitivity <= 1.0
    assert 0.0 <= specificity <= 1.0


def test_difference_interpretation() -> None:
    assert difference_interpretation(0.01, 0.03) == "RAD_DINO_HIGHER"
    assert difference_interpretation(-0.03, -0.01) == "DENSENET_HIGHER"
    assert difference_interpretation(-0.01, 0.01) == "NO_CLEAR_DIFFERENCE"


def test_primary_selection_preserves_densenet_baseline() -> None:
    bootstrap_rows = [
        {
            "scope": "macro",
            "label": "ALL_LABELS",
            "metric": "auprc",
            "delta_ci_low": -0.01,
            "delta_ci_high": 0.01,
            "interpretation": "NO_CLEAR_DIFFERENCE",
        }
    ]
    config = {
        "decision_policy": {
            "clinically_relevant_margin": 0.005,
            "minimum_ensemble_disagreement": 0.03,
            "minimum_unique_correct_fraction": 0.005,
        }
    }
    complementarity = {
        "overall": {
            "binary_disagreement_fraction": 0.10,
            "densenet_only_correct_fraction": 0.02,
            "rad_dino_only_correct_fraction": 0.02,
        }
    }

    result = choose_models(
        dense_metrics={"macro_auprc": 0.267},
        rad_metrics={"macro_auprc": 0.264},
        bootstrap_rows=bootstrap_rows,
        complementarity=complementarity,
        config=config,
    )

    assert result["primary_classification_model"] == "DenseNet-121"
    assert result["secondary_comparison_model"] == "RAD-DINO linear probe"
    assert result["ensemble_status"] == "VALIDATION_ONLY_ENSEMBLE_RESEARCH_ALLOWED"
