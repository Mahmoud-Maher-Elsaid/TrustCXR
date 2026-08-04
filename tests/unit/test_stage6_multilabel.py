from __future__ import annotations

import numpy as np
import torch

from trustcxr.classification.dataset import NIH_LABELS, stable_bucket
from trustcxr.classification.metrics import (
    calibrate_thresholds,
    compute_multilabel_metrics,
)
from trustcxr.classification.model import (
    build_densenet121,
    set_backbone_trainable,
)


def test_nih_label_contract() -> None:
    assert len(NIH_LABELS) == 14
    assert len(set(NIH_LABELS)) == 14


def test_patient_bucket_is_deterministic() -> None:
    assert stable_bucket("12345") == stable_bucket("12345")
    assert 0 <= stable_bucket("12345") < 10_000


def test_threshold_calibration_and_metrics() -> None:
    targets = np.array(
        [[1, 0], [1, 0], [0, 1], [0, 1]],
        dtype=np.int64,
    )
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        dtype=np.float64,
    )
    thresholds = calibrate_thresholds(targets, probabilities)
    metrics = compute_multilabel_metrics(
        targets,
        probabilities,
        ("A", "B"),
        thresholds,
    )
    assert metrics["macro_auroc"] == 1.0
    assert metrics["macro_auprc"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_densenet_output_shape() -> None:
    model = build_densenet121(14, 0.35, False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 14)


def test_backbone_freezing() -> None:
    model = build_densenet121(14, 0.35, False)
    set_backbone_trainable(model, False)
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
