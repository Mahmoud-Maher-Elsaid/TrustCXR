from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNLOCALIZED = "UNLOCALIZED"
    OUTSIDE_EXPECTED_ANATOMY = "OUTSIDE_EXPECTED_ANATOMY"
    UNCERTAIN = "UNCERTAIN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class FindingEvidence:
    label: str
    status: EvidenceStatus
    classifier_positive: bool
    localization_available: bool
    reason_code: str


def resolve_localization_evidence(
    *,
    label: str,
    classifier_positive: bool,
    localization_available: bool,
    localization_positive: bool,
    localization_reliable: bool,
    outside_image_geometry: bool = False,
) -> FindingEvidence:
    if label != "Pneumonia":
        return FindingEvidence(
            label,
            EvidenceStatus.NOT_APPLICABLE,
            classifier_positive,
            False,
            "NO_STAGE10_LOCALIZATION_CONTRACT_FOR_LABEL",
        )
    if outside_image_geometry:
        return FindingEvidence(
            label,
            EvidenceStatus.OUTSIDE_EXPECTED_ANATOMY,
            classifier_positive,
            localization_available,
            "DETECTION_OUTSIDE_VALID_IMAGE_GEOMETRY",
        )
    if not localization_available or not localization_reliable:
        status = EvidenceStatus.UNLOCALIZED if classifier_positive else EvidenceStatus.UNCERTAIN
        return FindingEvidence(
            label,
            status,
            classifier_positive,
            localization_available,
            "LOCALIZATION_NOT_RELIABLE_FOR_CONTRADICTION",
        )
    if classifier_positive and localization_positive:
        return FindingEvidence(
            label,
            EvidenceStatus.PARTIALLY_SUPPORTED,
            True,
            True,
            "CLASSIFIER_AND_RESEARCH_LOCALIZER_AGREE",
        )
    if localization_positive and not classifier_positive:
        return FindingEvidence(
            label,
            EvidenceStatus.CONTRADICTED,
            False,
            True,
            "LOCALIZER_POSITIVE_CLASSIFIER_NEGATIVE",
        )
    return FindingEvidence(
        label,
        EvidenceStatus.UNLOCALIZED if classifier_positive else EvidenceStatus.NOT_APPLICABLE,
        classifier_positive,
        True,
        "LOCALIZATION_ABSENCE_CANNOT_NEGATE_CLASSIFIER",
    )
