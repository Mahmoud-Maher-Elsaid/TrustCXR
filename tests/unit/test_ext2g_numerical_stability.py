from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.training.run_ext2g_local import (
    NumericalStabilityError,
    finite_gradients,
    finite_loss_components,
    target_diagnostics,
)


def _sample() -> tuple[torch.Tensor, dict[str, torch.Tensor], SimpleNamespace]:
    image = torch.ones((3, 16, 16), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
        "labels": torch.ones((1,), dtype=torch.int64),
        "image_id": torch.tensor([0]),
    }
    dataset = SimpleNamespace(records=[("synthetic-training-patient", [])])
    return image, target, dataset


def test_finite_fcos_loss_components_are_accepted() -> None:
    image, target, dataset = _sample()
    total, details = finite_loss_components(
        {"classification": torch.tensor(1.0), "bbox_regression": torch.tensor(0.5)},
        image,
        target,
        dataset,
        1,
        1,
        False,
        SimpleNamespace(get_scale=lambda: 1.0),
        5e-5,
    )
    assert float(total) == 1.5
    assert details["loss_components"]["classification"] == 1.0


def test_non_finite_loss_component_fails_closed() -> None:
    image, target, dataset = _sample()
    with pytest.raises(NumericalStabilityError, match="loss component"):
        finite_loss_components(
            {"classification": torch.tensor(float("nan"))},
            image,
            target,
            dataset,
            1,
            1,
            True,
            SimpleNamespace(get_scale=lambda: 128.0),
            5e-5,
        )


def test_invalid_target_box_fails_closed() -> None:
    image, target, dataset = _sample()
    target["boxes"] = torch.tensor([[1.0, 1.0, 20.0, 8.0]])
    with pytest.raises(NumericalStabilityError, match="image bounds"):
        target_diagnostics(image, target, dataset, 1, 1)


def test_non_finite_gradient_fails_closed() -> None:
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(NumericalStabilityError, match="gradients"):
        finite_gradients(model, {})


def test_smoke_mode_is_training_only_and_failed_runs_are_not_selected() -> None:
    source = (
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("scripts/training/run_ext2g_local.py")
        .read_text(encoding="utf-8")
    )
    assert "--smoke-only" in source
    assert "SMOKE_PASSED" in source
    assert '"status": "FAILED_NUMERICAL_STABILITY"' in source
    assert '"selected_checkpoint": None' in source
    wrapper = (
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("scripts/training/run_ext2g_local.ps1")
        .read_text(encoding="utf-8")
    )
    assert "$SmokeOnly" in wrapper
    assert "--smoke-only" in wrapper
    assert (
        "validation_loader"
        not in source.split("if args.smoke_only:", 1)[1].split("# The same", 1)[0]
    )
