# EXT-4A Grounded LLM Governance

**Status:** `COMPLETED — GOVERNANCE ONLY`
**Implementation:** `NOT STARTED`
**Next stage:** `EXT-4B EVIDENCE GROUNDING SCHEMA`

## Scientific question

EXT-4 asks whether a language model can transform already-governed TrustCXR
structured evidence into useful research-review language while preserving
faithfulness, provenance, uncertainty, DEFER states, and refusal of unsupported
medical claims. It is a text-grounded structured-evidence study, not direct
chest-X-ray diagnosis, clinical decision support, VLM reasoning, or a
replacement for the frozen vision pipeline or Stage 18 deterministic report.

The null or negative outcome is scientifically valid:
`RETAIN_DETERMINISTIC_REPORTING`. A failure may also be recorded as
`WITHHOLD_LLM_DUE_TO_FAITHFULNESS_RISK` or
`TECHNICALLY_INVALID_LLM_EXPERIMENT`; fluency is not success.

## Frozen boundary

The Frozen Core remains authoritative for Stage 5 view/quality evidence, Stage 9
classifier evidence, Stage 16 predictive uncertainty, Stage 17 deterministic
evidence integration and DEFER, Stage 18 deterministic reporting, Stage 19
verification, and Stage 20 final research-review decisions. An eventual LLM
may observe approved structured evidence but may not alter, overwrite, or
reinterpret those outputs.

EXT-3 is a controlled negative RSNA Lung Opacity localization result with an
unfrozen operating point. Localization is withheld and is not accepted for
trusted integration. Grad-CAM remains class-specific model attribution, not
lesion localization. Failed localization does not imply pathology absence.

## Input governance

The current serving response exposes aggregate, structured fields such as view
and technical-quality state, classifier scores, predictive uncertainty, fusion
status, report statements, verifier statuses, decisions, dispositions,
provenance, and limitations. These fields are candidates for governed future
grounding, subject to the EXT-4B schema. Safe source-stage and source-field
references, explicit availability markers, and uncertainty explanations are
conditional inputs, not assumptions that they already exist in the current
contract.

Raw CXR images, DICOM files or metadata, patient names, MRNs, patient IDs,
dates of birth, contact information, ungoverned clinical history, hidden
patient metadata, locked-test membership, direct dataset-directory access,
and raw training labels outside governed case evidence are forbidden. EXT-4A
does not send TrustCXR data to any external service.

## Evidence states

The governed vocabulary is `SUPPORTED`, `PARTIALLY_SUPPORTED`, `WITHHELD`,
`CONTRADICTED`, `NOT_AVAILABLE`, and `NOT_APPLICABLE`. Missing evidence is not
negative or positive evidence. Withheld evidence is not permission to infer.
An active `DEFER` remains `DEFER`; uncertainty must not be silently omitted;
high classifier score is not definitive diagnosis.

## Output and claim boundary

Permitted research-review outputs include concise summaries, structured
evidence explanations, uncertainty and limitation statements, reasons for
DEFER, provenance, surfaced contradictions, and expert-review questions.

The future system must refuse or abstain from definitive diagnosis, treatment
or management recommendations, causal conclusions, prognosis, severity,
laterality, anatomical or lesion localization, progression, patient history or
demographic claims, ungoverned urgency, fabricated measurements, fabricated
uncertainty/provenance, external-validation claims, and clinical-deployment
readiness. It must not upgrade “model evidence suggests” into “the patient
has,” override DEFER, treat WITHHELD as negative, or invent a source citation.

## Faithfulness taxonomy

EXT-4D must detect and report at least:

- `UNSUPPORTED_CLAIM`
- `CONTRADICTED_CLAIM`
- `PROVENANCE_ERROR`
- `OMISSION_OF_MATERIAL_LIMITATION`
- `DEFER_VIOLATION`
- `WITHHELD_EVIDENCE_VIOLATION`
- `FABRICATED_DETAIL`
- `UNSUPPORTED_LOCALIZATION`
- `UNSUPPORTED_SEVERITY`
- `UNSUPPORTED_LATERALITY`
- `EVIDENCE_POLARITY_ERROR`

These definitions are frozen before future benchmark results. Malformed input
fails closed; unavailable evidence is stated as unavailable; conflicts are
surfaced; missing provenance prevents attributed claims; unsupported medical
reasoning is declined.

## Provenance contract requirements

Every future evidence statement must identify its governed source field, source
stage, source status, safe source value where permitted, withheld handling,
uncertainty source, and verifier state. Provenance is internal TrustCXR
evidence traceability, not a fabricated web citation. Claim-level attribution
records and provider/model revision identity are not currently available and
must be added only through EXT-4B.

## Provider and privacy gate

No provider or model is selected by EXT-4A. Local/offline execution, external
API behavior, retention, provider logging, regional privacy, structured output,
version pinning, reproducibility, deterministic controls, cost, availability,
latency, and model revision identity must be evaluated before any later
selection. External-provider use requires a separate explicit privacy decision.

## Baseline and future benchmark

Stage 18 deterministic reporting remains the frozen baseline, safe fallback, and
comparison target. Future evaluation must compare faithfulness, unsupported
claims, contradiction handling, DEFER compliance, provenance, limitation
omission, readability/usefulness, and stability. Benchmark categories must
include complete and high-confidence evidence, high uncertainty, DEFER,
missing evidence, withheld localization, conflicts, positive classifier with
localization unavailable, quality/view issues, malformed context, unsupported
requests, and provenance-sensitive cases. Synthetic structured fixtures are
preferred; locked-test data is prohibited.

No prompt tuning, provider/model switching, case removal, or final-benchmark
tuning is allowed after results are observed without a predefined protocol.
Faithfulness or safety failure blocks integration. Quantitative thresholds are
reserved for EXT-4D and must be frozen before evaluation.

No LLM client, provider, external API, UI, training, or fine-tuning is part of
EXT-4A.
