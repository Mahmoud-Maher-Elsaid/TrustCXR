from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from trustcxr.explainability.contracts import FROZEN_STAGE9_LABELS
from trustcxr.explainability.gradcam import generate_gradcam


class TinyDenseNetShape(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Module()
        self.features.denseblock4 = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.features.norm5 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.classifier = nn.Linear(4, len(FROZEN_STAGE9_LABELS))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.features.denseblock4(tensor)
        tensor = self.features.norm5(tensor)
        tensor = tensor.mean(dim=(2, 3))
        return self.classifier(tensor)


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_gradcam_captures_finite_bounded_map_and_metadata() -> None:
    model = TinyDenseNetShape().eval()
    tensor = torch.ones(1, 3, 16, 16)
    output = generate_gradcam(model, tensor, "Nodule", resize_to_input=True)
    assert output.result.label_index == 5
    assert output.result.target_layer == "features.norm5"
    assert output.result.attribution_dimensions == (16, 16)
    assert output.result.normalized_attribution_dimensions == (16, 16)
    assert torch.isfinite(output.heatmap).all()
    assert 0.0 <= float(output.heatmap.min()) <= float(output.heatmap.max()) <= 1.0
    assert output.observation.gradient_shape == output.observation.activation_shape


def test_gradcam_is_deterministic_and_does_not_mutate_parameters() -> None:
    torch.manual_seed(7)
    model = TinyDenseNetShape().eval()
    tensor = torch.randn(1, 3, 16, 16)
    before = parameter_digest(model)
    first = generate_gradcam(model, tensor, "Atelectasis")
    second = generate_gradcam(model, tensor, "Atelectasis")
    assert torch.equal(first.heatmap, second.heatmap)
    assert parameter_digest(model) == before
    assert model.training is False


def test_gradcam_uses_requested_class_and_cleans_hooks() -> None:
    model = TinyDenseNetShape().eval()
    tensor = torch.randn(1, 3, 16, 16)
    first = generate_gradcam(model, tensor, "Atelectasis")
    second = generate_gradcam(model, tensor, "Pneumonia")
    assert first.result.label_index != second.result.label_index
    assert len(model.features.norm5._forward_hooks) == 0
    assert len(model.features.norm5._backward_hooks) == 0


def test_invalid_target_layer_and_unknown_label_fail_closed() -> None:
    model = TinyDenseNetShape().eval()
    tensor = torch.ones(1, 3, 16, 16)
    with pytest.raises(ValueError, match="Unknown frozen Stage 9 label"):
        generate_gradcam(model, tensor, "Unknown")
    with pytest.raises(ValueError, match="not approved"):
        generate_gradcam(model, tensor, "Nodule", target_layer="classifier")


def test_zero_gradient_map_fails_closed() -> None:
    model = TinyDenseNetShape().eval()
    for parameter in model.classifier.parameters():
        parameter.data.zero_()
    tensor = torch.ones(1, 3, 16, 16)
    with pytest.raises(ValueError, match="degenerate|zero activation"):
        generate_gradcam(model, tensor, "Nodule")
