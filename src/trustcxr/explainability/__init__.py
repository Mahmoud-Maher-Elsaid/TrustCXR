"""Governed contracts for future TrustCXR visual explainability work.

EXT-1A defines interfaces and claims boundaries only.  It intentionally does
not implement Grad-CAM or generate attribution artifacts.
"""

from trustcxr.explainability.contracts import (
    FROZEN_STAGE9_LABELS,
    AttributionResult,
    ExplainabilityMethod,
    ExplainabilityRequest,
    FrozenClassifierIdentity,
)

__all__ = [
    "AttributionResult",
    "ExplainabilityMethod",
    "ExplainabilityRequest",
    "FROZEN_STAGE9_LABELS",
    "FrozenClassifierIdentity",
]
