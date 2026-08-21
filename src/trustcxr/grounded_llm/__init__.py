"""Strict structured-evidence grounding and output contracts for EXT-4B/EXT-4C."""

from trustcxr.grounded_llm.contracts import (
    ClaimPermission,
    EvidenceEnvelope,
    EvidenceItem,
    EvidenceStatus,
    GenerationStatus,
    GroundedOutputEnvelope,
    OutputClaim,
    OutputClaimType,
    OutputEvidenceReference,
    OutputLimitation,
    SourceStage,
    build_synthetic_case,
)

__all__ = [
    "ClaimPermission",
    "GenerationStatus",
    "EvidenceEnvelope",
    "EvidenceItem",
    "EvidenceStatus",
    "GroundedOutputEnvelope",
    "OutputClaim",
    "OutputClaimType",
    "OutputEvidenceReference",
    "OutputLimitation",
    "SourceStage",
    "build_synthetic_case",
]
