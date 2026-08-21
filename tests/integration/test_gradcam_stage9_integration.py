from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from trustcxr.explainability.contracts import FROZEN_STAGE9_CHECKPOINT_SHA256
from trustcxr.explainability.gradcam import (
    PREFERRED_TARGET_LAYER,
    generate_gradcam,
    load_frozen_stage9_model,
    preprocess_stage9,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_frozen_stage9_gradcam_safe_fixture() -> None:
    if not torch.cuda.is_available():
        pytest.fail("EXT-1B requires the governed CUDA environment for integration validation")
    checkpoint = ROOT / "artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == FROZEN_STAGE9_CHECKPOINT_SHA256
    model = load_frozen_stage9_model(ROOT, torch.device("cuda"))
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_hooks = len(model.features.norm5._forward_hooks)
    assert model.training is False
    image = Image.new("RGB", (32, 24), color=(128, 128, 128))
    tensor = preprocess_stage9(image).to("cuda")
    output = generate_gradcam(model, tensor, "Atelectasis", target_layer=PREFERRED_TARGET_LAYER)
    assert output.result.target_layer == PREFERRED_TARGET_LAYER
    assert output.result.finite_values is True
    assert output.observation.activation_shape == (1, 1024, 7, 7)
    assert output.observation.gradient_shape == (1, 1024, 7, 7)
    assert output.result.attribution_dimensions == (7, 7)
    assert output.heatmap.shape == (224, 224)
    assert torch.isfinite(output.heatmap).all()
    assert 0.0 <= float(output.heatmap.min().detach())
    assert float(output.heatmap.max().detach()) <= 1.0
    assert len(model.features.norm5._forward_hooks) == before_hooks
    assert model.training is False
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_parameters, model.parameters(), strict=True)
    )
    del tensor, model
    torch.cuda.empty_cache()
