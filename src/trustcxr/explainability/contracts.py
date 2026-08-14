"""Typed, fail-closed contracts for the EXT-1A explainability foundation.

These contracts describe a future class-specific attribution request/result.
They do not execute a model, register hooks, or create heatmaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final

FROZEN_STAGE9_LABELS: Final[tuple[str, ...]] = (
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

FROZEN_STAGE9_CHECKPOINT: Final[str] = (
    "artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt"
)
FROZEN_STAGE9_CHECKPOINT_SHA256: Final[str] = (
    "bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de"
)
FROZEN_STAGE9_FREEZE_CONFIG: Final[str] = "configs/training/stage9_final_freeze.json"
FROZEN_STAGE9_ABLATION_CONFIG: Final[str] = (
    "configs/training/stage9b_segmentation_guided_ablation.json"
)
FROZEN_STAGE9_ABLATION_CONFIG_SHA256: Final[str] = (
    "347e2a1bebbad2d48932d9d6163217d0194ae1eff507b2d717bfa932fca84ef4"
)
FROZEN_STAGE9_FREEZE_CONFIGURATION_SHA256: Final[str] = (
    "75933fad8f47e7d637596edd935b02f8b8f5ed9d104acacc3e67500a9abdbd69"
)


class ExplainabilityMethod(StrEnum):
    GRADCAM = "gradcam"


class _Scope(StrEnum):
    ATTRIBUTION_ONLY = "class-specific model attribution only"


def label_index(label: str) -> int:
    """Resolve exactly one frozen Stage 9 label, failing closed otherwise."""
    try:
        return FROZEN_STAGE9_LABELS.index(label)
    except ValueError as error:
        raise ValueError(f"Unknown frozen Stage 9 label: {label!r}") from error


@dataclass(frozen=True)
class FrozenClassifierIdentity:
    """Identity of the frozen classifier that a future attribution may explain."""

    model_family: str = "DenseNet121"
    variant: str = "original"
    output_labels: int = 14
    checkpoint: str = FROZEN_STAGE9_CHECKPOINT
    checkpoint_sha256: str = FROZEN_STAGE9_CHECKPOINT_SHA256
    freeze_config: str = FROZEN_STAGE9_FREEZE_CONFIG
    ablation_config: str = FROZEN_STAGE9_ABLATION_CONFIG
    ablation_config_sha256: str = FROZEN_STAGE9_ABLATION_CONFIG_SHA256
    freeze_configuration_sha256: str = FROZEN_STAGE9_FREEZE_CONFIGURATION_SHA256
    image_size: int = 224
    normalization: str = "ImageNet mean/std: (0.485,0.456,0.406)/(0.229,0.224,0.225)"
    channel_policy: str = "RGB tensor with three channels"

    def __post_init__(self) -> None:
        if self.output_labels != len(FROZEN_STAGE9_LABELS):
            raise ValueError("Frozen classifier label count does not match its contract")
        if PurePosixPath(self.checkpoint).is_absolute():
            raise ValueError("Checkpoint references must be repository-relative")


@dataclass(frozen=True)
class ExplainabilityRequest:
    """Bounded future attribution request; no arbitrary module execution."""

    input_reference: str
    label: str
    model_identity: FrozenClassifierIdentity = field(default_factory=FrozenClassifierIdentity)
    method: ExplainabilityMethod = ExplainabilityMethod.GRADCAM
    target_layer: str = "features.norm5"
    normalize: bool = True

    @property
    def label_index(self) -> int:
        return label_index(self.label)

    def __post_init__(self) -> None:
        if (
            not self.input_reference
            or PurePosixPath(self.input_reference).is_absolute()
            or re.match(r"^[A-Za-z]:[\\/]", self.input_reference)
            or ".." in PurePosixPath(self.input_reference).parts
        ):
            raise ValueError("Input reference must be a bounded, non-empty reference")
        if self.method is not ExplainabilityMethod.GRADCAM:
            raise ValueError("Unsupported explainability method")
        if not self.target_layer or "." not in self.target_layer:
            raise ValueError("Target layer must be an approved module path")
        label_index(self.label)


@dataclass(frozen=True)
class AttributionResult:
    """Future attribution metadata with no clinical interpretation fields."""

    model_identity: FrozenClassifierIdentity
    method: ExplainabilityMethod
    requested_label: str
    label_index: int
    target_layer: str
    original_model_score: float
    input_dimensions: tuple[int, int]
    attribution_dimensions: tuple[int, int]
    preprocessing_contract: str = "stage9_original_224_imagenet_rgb"
    normalized: bool = True
    raw_attribution_dtype: str = "float32"
    warnings: tuple[str, ...] = ()
    scope: _Scope = _Scope.ATTRIBUTION_ONLY

    def __post_init__(self) -> None:
        if self.requested_label not in FROZEN_STAGE9_LABELS:
            raise ValueError("Result label is outside the frozen Stage 9 contract")
        if self.label_index != label_index(self.requested_label):
            raise ValueError("Result label index does not match the frozen label order")
        if not self.model_identity.checkpoint_sha256:
            raise ValueError("Attribution provenance requires a checkpoint fingerprint")
        if self.method is not ExplainabilityMethod.GRADCAM:
            raise ValueError("Unsupported explainability method")
        if not 0.0 <= self.original_model_score <= 1.0:
            raise ValueError("Model score must be a finite bounded research score")
        if min(*self.input_dimensions, *self.attribution_dimensions) <= 0:
            raise ValueError("Attribution dimensions must be positive")
