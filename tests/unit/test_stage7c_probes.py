from __future__ import annotations

import torch

from trustcxr.probes.rad_dino import (
    LinearProbe,
    MLPProbe,
    compute_positive_weights,
    compute_standardization,
    select_champion,
)


def test_linear_probe_output_shape() -> None:
    model = LinearProbe(768, 14)
    outputs = model(torch.randn(8, 768))
    assert outputs.shape == (8, 14)


def test_mlp_probe_output_shape() -> None:
    model = MLPProbe(768, 128, 14, 0.25)
    outputs = model(torch.randn(8, 768))
    assert outputs.shape == (8, 14)


def test_standardization_statistics_are_finite() -> None:
    embeddings = torch.arange(12 * 768, dtype=torch.float32).reshape(12, 768)
    mean, std, details = compute_standardization(embeddings, 5, 1e-6)
    assert mean.shape == (768,)
    assert std.shape == (768,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)
    assert details["std_minimum"] > 0


def test_positive_weights_are_clipped() -> None:
    labels = torch.zeros((100, 14), dtype=torch.uint8)
    labels[:50, 0] = 1
    labels[:2, 1:] = 1
    weights, details = compute_positive_weights(labels, 20.0)
    assert weights.shape == (14,)
    assert weights[0].item() == 1.0
    assert torch.all(weights[1:] == 20.0)
    assert details["Atelectasis"]["positive_count"] == 50


def test_champion_selection_uses_validation_metrics() -> None:
    linear = {
        "name": "linear",
        "parameter_count": 100,
        "validation_metrics": {
            "macro_auprc": 0.25,
            "macro_auroc": 0.81,
        },
    }
    mlp = {
        "name": "mlp",
        "parameter_count": 200,
        "validation_metrics": {
            "macro_auprc": 0.26,
            "macro_auroc": 0.80,
        },
    }
    assert select_champion([linear, mlp])["name"] == "mlp"
