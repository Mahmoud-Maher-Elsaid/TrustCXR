"""Local request-scoped serving adapter for the accepted EXT-1B baseline."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from threading import Lock

import torch
from PIL import Image

from trustcxr.explainability.gradcam import (
    PREFERRED_TARGET_LAYER,
    generate_gradcam,
    load_frozen_stage9_model,
    preprocess_stage9,
)
from trustcxr.serving.local_inference import LocalReviewError, decode_image
from trustcxr.serving.schemas import GradCAMReviewResponse

SUPPORTED_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)


class GradCAMServingError(LocalReviewError):
    """Sanitized error for optional attribution requests."""


class GradCAMService:
    """Lazy, serialized model owner for bounded local attribution requests."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._lock = Lock()

    def _get_model(self) -> tuple[torch.nn.Module, torch.device]:
        if self._model is None or self._device is None:
            if not torch.cuda.is_available():
                raise GradCAMServingError("CUDA_UNAVAILABLE", 503, "GRADCAM_MODEL_LOAD")
            self._device = torch.device("cuda")
            self._model = load_frozen_stage9_model(self.root, self._device)
        return self._model, self._device

    @staticmethod
    def _png_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _visuals(image: Image.Image, heatmap: torch.Tensor) -> tuple[str, str]:
        values = (heatmap.detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        grayscale = Image.fromarray(values, mode="L").resize(image.size, Image.Resampling.BILINEAR)
        pixels = [(int(value), 0, 255 - int(value)) for value in grayscale.getdata()]
        color = Image.new("RGB", grayscale.size)
        color.putdata(pixels)
        overlay = Image.blend(image.convert("RGB"), color, alpha=0.42)
        return GradCAMService._png_base64(color), GradCAMService._png_base64(overlay)

    def attribute(self, payload: bytes, media_type: str, label: str) -> GradCAMReviewResponse:
        if label not in SUPPORTED_LABELS:
            raise GradCAMServingError("UNSUPPORTED_ATTRIBUTION_LABEL", 422, "GRADCAM_REQUEST")
        try:
            image, _ = decode_image(payload, media_type)
        except LocalReviewError as error:
            raise GradCAMServingError(
                error.reason_code, error.status_code, "GRADCAM_IMAGE_DECODE"
            ) from error
        with self._lock:
            try:
                model, device = self._get_model()
                tensor = preprocess_stage9(image).to(device)
                output = generate_gradcam(
                    model,
                    tensor,
                    label,
                    target_layer=PREFERRED_TARGET_LAYER,
                    resize_to_input=False,
                )
                heatmap = output.heatmap
                heatmap_png, overlay_png = self._visuals(image, heatmap)
                return GradCAMReviewResponse(
                    label=label,
                    label_index=output.result.label_index,
                    model_score=output.result.original_model_score,
                    checkpoint_sha256=output.result.model_identity.checkpoint_sha256,
                    raw_attribution_dimensions=output.result.attribution_dimensions,
                    display_dimensions=image.size,
                    heatmap_png_base64=heatmap_png,
                    overlay_png_base64=overlay_png,
                )
            except GradCAMServingError:
                raise
            except ValueError as error:
                if "degenerate" in str(error).lower() or "zero activation" in str(error).lower():
                    raise GradCAMServingError(
                        "ATTRIBUTION_UNAVAILABLE", 422, "GRADCAM_ATTRIBUTION"
                    ) from error
                raise GradCAMServingError(
                    "INFERENCE_FAILURE", 500, "GRADCAM_ATTRIBUTION"
                ) from error
            finally:
                if "tensor" in locals():
                    del tensor
