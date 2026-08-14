"""Governed native PyTorch Grad-CAM for the frozen Stage 9 classifier.

This module is an isolated research-extension adapter. It does not alter the
serving pipeline, report generation, verifier, decision logic, or UI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.explainability.contracts import (
    FROZEN_STAGE9_ABLATION_CONFIG_SHA256,
    FROZEN_STAGE9_CHECKPOINT,
    FROZEN_STAGE9_CHECKPOINT_SHA256,
    FROZEN_STAGE9_LABELS,
    AttributionResult,
    ExplainabilityMethod,
    FrozenClassifierIdentity,
    label_index,
)
from trustcxr.integration.stage9b_ablation import build_model

TARGET_LAYER_CANDIDATES = ("features.denseblock4", "features.norm5")
PREFERRED_TARGET_LAYER = "features.norm5"
STAGE9_IMAGE_SIZE = 224
STAGE9_MEAN = (0.485, 0.456, 0.406)
STAGE9_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_stage9(image: Image.Image) -> torch.Tensor:
    """Apply the exact frozen Stage 9 original inference preprocessing."""
    rgb = image.convert("RGB")
    resized = vision_functional.resize(
        rgb,
        [STAGE9_IMAGE_SIZE, STAGE9_IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = vision_functional.to_tensor(resized)
    return vision_functional.normalize(tensor, STAGE9_MEAN, STAGE9_STD).unsqueeze(0)


def _module_at_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if path not in TARGET_LAYER_CANDIDATES:
        raise ValueError(f"Target layer is not approved: {path}")
    current: Any = model
    for component in path.split("."):
        if not hasattr(current, component):
            raise ValueError(f"Target layer is unavailable: {path}")
        current = getattr(current, component)
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"Target layer is not a module: {path}")
    return current


@dataclass(frozen=True)
class TargetLayerObservation:
    target_layer: str
    module_type: str
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]
    finite_activation: bool
    finite_gradient: bool
    normalized_range: tuple[float, float]


@dataclass(frozen=True)
class GradCAMOutput:
    result: AttributionResult
    heatmap: torch.Tensor
    observation: TargetLayerObservation


def load_frozen_stage9_model(root: Path, device: torch.device) -> torch.nn.Module:
    """Load the accepted Stage 9 model through its canonical builder."""
    checkpoint_path = root / FROZEN_STAGE9_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != FROZEN_STAGE9_CHECKPOINT_SHA256:
        raise ValueError("Frozen Stage 9 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != FROZEN_STAGE9_ABLATION_CONFIG_SHA256:
        raise ValueError("Frozen Stage 9 loader configuration fingerprint mismatch")
    if checkpoint.get("model_architecture") != "DenseNet121":
        raise ValueError("Frozen Stage 9 architecture mismatch")
    model = build_model(len(FROZEN_STAGE9_LABELS), input_channels=3, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(cam).all():
        raise ValueError("Grad-CAM contains non-finite values")
    cam = torch.relu(cam)
    maximum = cam.amax()
    if not torch.isfinite(maximum) or float(maximum.detach()) <= 1e-12:
        raise ValueError("Grad-CAM is degenerate or has zero activation")
    normalized = cam / maximum
    if not torch.isfinite(normalized).all():
        raise ValueError("Normalized Grad-CAM contains non-finite values")
    return normalized.clamp(0.0, 1.0)


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    label: str,
    *,
    target_layer: str = PREFERRED_TARGET_LAYER,
    resize_to_input: bool = True,
) -> GradCAMOutput:
    """Generate one class-specific Grad-CAM map without changing model weights.

    The backward objective is the selected class *logit*, not a sigmoid
    probability. Gradients are spatially averaged to obtain channel weights;
    their weighted activation sum is rectified and safely normalized.
    """
    class_index = label_index(label)
    layer = _module_at_path(model, target_layer)
    model.eval()
    activation: torch.Tensor | None = None

    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        nonlocal activation
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target output must be a four-dimensional tensor")
        activation = output
        activation.retain_grad()

    hook = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("Grad-CAM expects one model input and a label-logit vector")
            score = torch.sigmoid(logits[0, class_index]).detach()
            logits[0, class_index].backward()
        if activation is None or activation.grad is None:
            raise ValueError("Grad-CAM activation or gradient was not captured")
        gradient = activation.grad
        if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
            raise ValueError("Grad-CAM activation or gradient is non-finite")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        raw_cam = torch.sum(weights * activation, dim=1).squeeze(0)
        normalized = _normalize_cam(raw_cam)
        raw_shape = tuple(int(value) for value in normalized.shape)
        display_map = normalized
        if resize_to_input:
            display_map = (
                torch.nn.functional.interpolate(
                    normalized[None, None],
                    size=tuple(int(value) for value in input_tensor.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            display_map = display_map.clamp(0.0, 1.0)
        display_shape = tuple(int(value) for value in display_map.shape)
        observation = TargetLayerObservation(
            target_layer=target_layer,
            module_type=type(layer).__name__,
            activation_shape=tuple(int(value) for value in activation.shape),
            gradient_shape=tuple(int(value) for value in gradient.shape),
            finite_activation=True,
            finite_gradient=True,
            normalized_range=(
                float(normalized.min().detach()),
                float(normalized.max().detach()),
            ),
        )
        identity = FrozenClassifierIdentity()
        result = AttributionResult(
            model_identity=identity,
            method=ExplainabilityMethod.GRADCAM,
            requested_label=label,
            label_index=class_index,
            target_layer=target_layer,
            original_model_score=float(score),
            input_dimensions=(int(input_tensor.shape[-2]), int(input_tensor.shape[-1])),
            attribution_dimensions=raw_shape,
            normalized_attribution_dimensions=display_shape,
            finite_values=True,
        )
        return GradCAMOutput(result=result, heatmap=display_map.detach(), observation=observation)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
