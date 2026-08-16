# EXT-4C Grounded Output Contract

EXT-4C defines a versioned, fail-closed output object for a future grounded language
extension. It does not implement generation, select a model or provider, or call an API.

## Boundary and authority

`EXT4_OUTPUT_CONTRACT`, version `1`, represents a research-review rendering of governed
structured evidence. It is subordinate to the Frozen Core. Stage 18 deterministic reporting
remains the `FROZEN_DETERMINISTIC_BASELINE`; this output is
`RESEARCH_EXTENSION_ONLY` and cannot replace Stage 18, Stage 19 verification, or Stage 20
research-review disposition.

The only generation statuses are `COMPLETED`, `ABSTAINED`, `DEFERRED`,
`REJECTED_INVALID_INPUT`, and `REJECTED_UNGROUNDED_REQUEST`. Active deterministic DEFER
is preserved and can only produce defer, limitation, contradiction, or reviewer-flag claims.

## Output contract

The strict `GroundedOutputEnvelope` contains a research-safe case reference, status, optional
summary tied to claim IDs, evidence references, structured claims, uncertainty, limitations,
DEFER state, contradiction records, provenance references, reviewer flags, and the immutable
baseline reference. Unknown fields, schema versions, identifiers, image paths, and provider
metadata are rejected.

Every factual `OutputClaim` has a controlled type, support state, evidence IDs, provenance
references, and optional limitation/uncertainty links. A completed output cannot contain an
unreferenced summary or empty claim set. Supported claims cannot upgrade partial, contradicted,
withheld, unavailable, or non-applicable evidence. Contradictions remain explicit and require
at least two evidence references.

Allowed claim types are research summary, view, quality, classifier evidence, uncertainty,
DEFER reason, limitation, provenance, verifier, contradiction, and reviewer flag. Clinical or
unsupported types such as diagnosis, treatment, management, prognosis, severity, laterality,
anatomical/lesion localization, causality, fabricated history, unsupported measurements,
clinical urgency, external validation, and deployment readiness are not in the enum and cannot
be represented.

Limitations are first-class objects, including uncertainty, DEFER, unavailable evidence,
contradiction, technical quality, verifier warning, and withheld localization. EXT-3 remains
`LOCALIZATION_INTEGRATION_WITHHELD`; it must never be rendered as lesion absence or a negative
finding. Grad-CAM remains class-specific attribution only and cannot support localization,
segmentation, diagnosis, causality, severity, or laterality.

Uncertainty uses the existing EXT-4B representation and preserves its source and predictive-only
interpretation. `NOT_AVAILABLE` is neither positive nor negative evidence. `WITHHELD` is a
first-class status, not missing data. Malformed values, non-finite numbers, invalid provenance,
unknown references, patient identity, raw image/path fields, and DEFER overrides fail closed.

The free-text summary is not an escape hatch: when present it must reference structured claim
IDs. Later generation stages may render it from those claims; EXT-4C does not implement an NLP
validator or a model runtime.

## Synthetic coverage and lifecycle

Synthetic non-patient fixtures cover supported evidence, uncertainty, DEFER, withheld
localization, contradiction, missing evidence, malformed support, nonexistent references, and
DEFER override attempts. They do not contain patient rows, images, DICOM, or locked-test data.

EXT-4C is `COMPLETED — OUTPUT CONTRACT ONLY`. Provider/model selection, an LLM runtime, prompts,
external APIs, and UI remain unauthorized/not started. The next authorized stage is
`EXT-4D HALLUCINATION / FAITHFULNESS BENCHMARK`.
