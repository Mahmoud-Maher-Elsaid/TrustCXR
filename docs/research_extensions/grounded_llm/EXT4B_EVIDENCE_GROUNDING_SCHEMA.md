# EXT-4B Evidence Grounding Schema

**Status:** `COMPLETED — SCHEMA ONLY`
**Implementation:** `SCHEMA VALIDATION ONLY; LLM NOT STARTED`
**Next stage:** `EXT-4C OUTPUT CONTRACT`

## Contract identity and boundary

The versioned envelope is `EXT4_GROUNDING_SCHEMA` version `1`. It carries only
strict, aggregate, structured evidence already produced by the TrustCXR Frozen
Core. It carries no image, DICOM, patient identifier, dataset path, locked-test
membership, provider field, prompt, model field, or generated text.

Stages represented by the controlled source vocabulary are `STAGE_5`, `STAGE_9`,
`STAGE_16`, `STAGE_17`, `STAGE_18`, `STAGE_19`, and `STAGE_20`. The current
serving response exposes aggregate fields for view and technical quality,
classifier scores, predictive uncertainty, fusion status, report statements,
verifier statuses, decisions, dispositions, provenance, and limitations. These
are existing-and-allowed candidates; claim-level LLM grounding records and
provider/model revision identity are not currently available and remain
proposed for later schema work.

## Envelope

Each envelope has a schema identity/version, a non-identifying
`research_case_...` reference, evidence items, deterministic decision state,
uncertainty state, controlled limitations, first-class withheld evidence,
verifier state, envelope provenance, and the fixed grounding policy
`EXT4A_GOVERNED_STRUCTURED_EVIDENCE_ONLY`. Unknown fields and unknown versions
fail closed. The case reference is an internal synthetic/research-safe token,
not a patient or dataset identifier.

## Evidence items and values

An evidence item has an ID, controlled source stage, evidence type, frozen
status, optional label/value, provenance, controlled claim permissions, and
optional limitation metadata. `SUPPORTED`, `PARTIALLY_SUPPORTED`, and
`CONTRADICTED` items require provenance whose source stage matches the item.
`WITHHELD` items carry a reason but no value or claim permission. `NOT_AVAILABLE`
and `NOT_APPLICABLE` items cannot support claims.

Values are strict typed records: `TEXT`, finite `NUMBER`, `BOOLEAN`, or
`CATEGORIES`. NaN, infinity, multiple simultaneous value types, and arbitrary
extension keys are rejected. Claim permissions are a closed vocabulary for
research summary, classifier evidence, uncertainty, DEFER reason, limitation,
provenance, quality/view, verifier, and contradiction statements.

## Evidence-state semantics

The states are `SUPPORTED`, `PARTIALLY_SUPPORTED`, `WITHHELD`, `CONTRADICTED`,
`NOT_AVAILABLE`, and `NOT_APPLICABLE`. Missing evidence is neither positive nor
negative evidence. Withheld evidence cannot be inferred around. Contradictions
must be surfaced. Partial support cannot be upgraded. Non-applicability is not
disease absence. A deterministic `DEFER` cannot become a confident conclusion.

EXT-3 localization is represented only by
`LOCALIZATION_INTEGRATION_WITHHELD` with the explanation that it is a
controlled-negative result not accepted for trusted integration. It never has a
positive value payload and never supports lesion localization. Grad-CAM remains
attribution-only and is not a source of localization ground truth.

## Decision, uncertainty, report, verifier, and final state

`DecisionState` preserves the existing deterministic decision vocabulary and
requires `defer_active` to agree with `DEFER`; active DEFER is non-overridable.
`UncertaintyState` exposes only the existing Stage 16 predictive uncertainty,
with source and the limitation `PREDICTIVE_ONLY_NOT_EPISTEMIC`; unavailable
uncertainty carries no value. Stage 18 is represented through traceable report
statement evidence and remains the immutable deterministic baseline. Stage 19
verifier statuses, failures, and evidence references are preserved. Stage 20
final research-review state remains authoritative and cannot be overwritten by
future generated language.

## Provenance

Every supporting item identifies source stage, source component, source schema,
and source field; safe config/checkpoint fingerprints may be added only where
already governed. Envelope provenance records the frozen pipeline version and
component references. These are internal TrustCXR traceability fields, not web
citations. Missing provenance blocks supported claims.

## Privacy and fail-closed rules

Raw images, DICOM, patient names, MRNs, medical-record identifiers, dates of
birth, addresses, contact data, raw IDs, hidden metadata, dataset directories,
locked-test membership, and ungoverned training labels are forbidden by strict
schemas. Malformed evidence, unknown status/stage/version, invalid provenance,
unsupported permissions, non-finite numbers, malformed DEFER, and withheld
evidence presented as support are rejected. No provider, model, client, prompt,
network, or external API is part of EXT-4B.

## Synthetic coverage and compatibility

The fixture set contains six non-patient cases: supported evidence, high
uncertainty, DEFER, withheld localization, conflict, and missing evidence.
Focused tests validate their state semantics and the privacy/claim boundaries.
Schema additions must be backward-compatible optional fields; breaking changes
require a new schema version and explicit migration. Unknown incompatible
versions fail closed.

Quantitative faithfulness thresholds, model/provider selection, output wording,
and generation behavior are reserved for EXT-4C/EXT-4D. No LLM runtime or
external evaluation is authorized by EXT-4B.
