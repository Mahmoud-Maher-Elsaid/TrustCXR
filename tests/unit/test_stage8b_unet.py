from __future__ import annotations

import numpy as np
import torch

from trustcxr.segmentation.stage8b_unet import (
    ResNet34UNet,
    deterministic_subset,
    horizontal_flip_anatomy,
    metrics_from_counts,
    soft_dice_score,
)


def test_soft_dice_is_one_for_perfect_binary_logits() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]] * 3)
    logits = torch.where(targets > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    score = soft_dice_score(logits, targets)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-6)


def test_horizontal_flip_swaps_lung_channels() -> None:
    image = torch.arange(12).reshape(1, 3, 4)
    masks = torch.zeros(3, 2, 4)
    masks[0, :, 0] = 1
    masks[1, :, 3] = 1
    masks[2, :, 1:3] = 1
    _, flipped = horizontal_flip_anatomy(image, masks)
    assert torch.equal(flipped[0], torch.flip(masks[1], dims=(-1,)))
    assert torch.equal(flipped[1], torch.flip(masks[0], dims=(-1,)))


def test_deterministic_subset_is_stable_and_bounded() -> None:
    values = [f"image-{index}" for index in range(100)]
    first = deterministic_subset(values, 10, seed=7)
    second = deterministic_subset(values, 10, seed=7)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_metrics_from_counts_returns_expected_values() -> None:
    counts = {
        "tp": np.array([8.0, 8.0, 8.0]),
        "fp": np.array([2.0, 2.0, 2.0]),
        "fn": np.array([2.0, 2.0, 2.0]),
        "tn": np.array([88.0, 88.0, 88.0]),
    }
    metrics = metrics_from_counts(counts)
    assert np.allclose(metrics["dice"], 0.8)
    assert np.allclose(metrics["iou"], 2.0 / 3.0)
    assert np.isclose(metrics["macro_dice"], 0.8, rtol=0.0, atol=1e-12)


def test_unet_output_shape_matches_input() -> None:
    model = ResNet34UNet(pretrained=False)
    inputs = torch.randn(1, 3, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == (1, 3, 128, 128)
