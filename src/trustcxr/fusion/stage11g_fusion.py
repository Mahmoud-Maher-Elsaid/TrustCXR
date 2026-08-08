from __future__ import annotations

from dataclasses import dataclass

from trustcxr.fusion.evidence_contract import FindingEvidence, resolve_localization_evidence


@dataclass(frozen=True)
class FusionInput:
    label: str
    classifier_positive: bool
    localization_available: bool
    localization_positive: bool
    localization_reliable: bool
    outside_image_geometry: bool = False


def fuse_finding(value: FusionInput) -> FindingEvidence:
    """Apply the frozen Stage 11 evidence policy without running either model."""
    return resolve_localization_evidence(
        label=value.label,
        classifier_positive=value.classifier_positive,
        localization_available=value.localization_available,
        localization_positive=value.localization_positive,
        localization_reliable=value.localization_reliable,
        outside_image_geometry=value.outside_image_geometry,
    )
