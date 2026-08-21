from __future__ import annotations

import pytest

from trustcxr.explainability.contracts import (
    FROZEN_STAGE9_CHECKPOINT,
    FROZEN_STAGE9_CHECKPOINT_SHA256,
    FROZEN_STAGE9_LABELS,
    AttributionResult,
    ExplainabilityMethod,
    ExplainabilityRequest,
    FrozenClassifierIdentity,
    label_index,
)


def test_frozen_label_order_and_indices_are_stable() -> None:
    assert len(FROZEN_STAGE9_LABELS) == 14
    assert label_index("Atelectasis") == 0
    assert label_index("Pleural_Thickening") == 12
    assert label_index("Hernia") == 13


def test_unknown_label_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown frozen Stage 9 label"):
        label_index("Lesion")


def test_request_is_gradcam_and_repository_relative() -> None:
    request = ExplainabilityRequest(input_reference="fixtures/safe.png", label="Nodule")
    assert request.method is ExplainabilityMethod.GRADCAM
    assert request.label_index == 5
    assert request.target_layer == "features.norm5"


def test_request_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="bounded"):
        ExplainabilityRequest(input_reference="../patient.png", label="Nodule")


def test_identity_contains_frozen_checkpoint_contract() -> None:
    identity = FrozenClassifierIdentity()
    assert identity.model_family == "DenseNet121"
    assert identity.output_labels == 14
    assert identity.checkpoint == FROZEN_STAGE9_CHECKPOINT
    assert identity.checkpoint_sha256 == FROZEN_STAGE9_CHECKPOINT_SHA256
    assert identity.image_size == 224


def test_result_is_attribution_only_and_requires_matching_label_index() -> None:
    identity = FrozenClassifierIdentity()
    result = AttributionResult(
        model_identity=identity,
        method=ExplainabilityMethod.GRADCAM,
        requested_label="Nodule",
        label_index=5,
        target_layer="features.norm5",
        original_model_score=0.4,
        input_dimensions=(224, 224),
        attribution_dimensions=(7, 7),
    )
    assert result.scope.value == "class-specific model attribution only"
    with pytest.raises(ValueError, match="does not match"):
        AttributionResult(
            model_identity=identity,
            method=ExplainabilityMethod.GRADCAM,
            requested_label="Nodule",
            label_index=4,
            target_layer="features.norm5",
            original_model_score=0.4,
            input_dimensions=(224, 224),
            attribution_dimensions=(7, 7),
        )
