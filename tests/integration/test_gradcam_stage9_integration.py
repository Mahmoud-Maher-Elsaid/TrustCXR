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
    image = Image.new("RGB", (32, 24), color=(128, 128, 128))
    tensor = preprocess_stage9(image).to("cuda")
    output = generate_gradcam(model, tensor, "Atelectasis", target_layer=PREFERRED_TARGET_LAYER)
    assert output.result.target_layer == PREFERRED_TARGET_LAYER
    assert output.result.finite_values is True
    assert output.heatmap.shape == (224, 224)
    assert torch.isfinite(output.heatmap).all()
    del tensor, model
    torch.cuda.empty_cache()
