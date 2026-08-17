# TrustCXR Post-Release Research Extension Roadmap

**Classification:** `OPTIONAL_POST_RELEASE_RESEARCH_EXTENSION`
**Implementation status:** `EXT-4A GOVERNANCE COMPLETED — IMPLEMENTATION NOT STARTED`
**Authorization:** `POST-CORE-RELEASE ONLY`

**EXT-1A Explainability Foundation:** `COMPLETED` (governance and compatibility audit only)
**EXT-1B Grad-CAM Implementation and Technical Validation:** `COMPLETED`
**Grad-CAM UI Integration / EXT-1C:** `COMPLETED`
**EXT-3 Final RSNA Lung Opacity localization:** `CLOSED_CONTROLLED_NEGATIVE_RESULT` (operating point unfrozen; localization integration withheld)
**EXT-4A Grounded LLM Governance:** `COMPLETED — GOVERNANCE ONLY`
**EXT-4B Evidence Grounding Schema:** `COMPLETED — SCHEMA ONLY`
**EXT-4C Output Contract:** `COMPLETED — OUTPUT CONTRACT ONLY`
**EXT-4D Hallucination / Faithfulness Benchmark:** `COMPLETED — FROZEN BENCHMARK DEFINITION ONLY`
**EXT-4E Model Selection Protocol:** `MODEL_SELECTION_PROTOCOL_PREPARED`
**EXT-4E2B Qwen Runtime/Model Identity:** `RUNTIME_AND_MODEL_IDENTITY_FROZEN`
**EXT-4E2C Qwen Load-Only GPU Smoke:** `LOAD_ONLY_PASS — TECHNICALLY LOADABLE; NOT SCIENTIFICALLY EVALUATED`
**Next authorized stage:** `EXT-4E MODEL / PROVIDER SELECTION AND BASELINE EXECUTION` (protocol prepared; execution not started)
**Next local execution step:** `LOCAL DEVELOPMENT CASE EXECUTION`
**True pathology localization follow-on experiments:** `CLOSED — NO EXT-3B/EXT-3C OR ADDITIONAL LOCALIZATION TRAINING`
**Grounded LLM reporting:** `NOT STARTED`
**Multimodal VLM:** `NOT STARTED`

```text
IMPLEMENTATION STATUS: EXT-4A GOVERNANCE COMPLETED — IMPLEMENTATION NOT STARTED
AUTHORIZATION: POST-CORE-RELEASE ONLY
```

This document records optional research directions after closure of the core TrustCXR release. It does not authorize implementation, add a current capability, or change any core scientific claim. Development should begin only after core release closure, on a separate future branch such as `research-extension/explainability-grounded-llm`. That branch is not created by this roadmap.

## Core and extension boundary

### Core TrustCXR

The core remains a deterministic, research-only application with frozen vision models, structured evidence, deterministic reporting and verification, conservative decision support, and expert review. The accepted interface is designated `FROZEN_CORE_RESEARCH_REVIEW_UI`.

The core remains valid and complete without Grad-CAM, true lesion localization, grounded LLM reporting, multimodal VLMs, new datasets, expert active learning, or acquisition of an external-validation cohort. Post-release experiments may not retroactively broaden core claims.

The following core limitations remain active until a new, explicit, evidence-based acceptance gate is passed:

- `NO_RELIABLE_POSITIVE_LESION_LOCALIZATION`
- `LOCALIZATION_ABSENCE_MUST_NOT_CONTRADICT_CLASSIFIER_EVIDENCE`
- attention or attribution is not validated lesion localization
- no finding laterality from localization
- no severity, temporal-change, OOD, diagnosis, treatment, or autonomous-release claim

## Extension 1: visual explainability

### Research question

Evaluate whether a class-specific attribution method, initially Grad-CAM or a scientifically justified alternative, can expose regions that influenced a selected frozen Stage 9 classifier signal.

Permitted future user-facing terms are **Model Attention Map** and **Visual Explanation**. Until localization is independently validated, the terms **Disease Location**, **Lesion Location**, **Confirmed Lesion**, and **Pathology Localization** are prohibited.

### Required prospective protocol

Before generating results, freeze:

- attribution-method selection and scientific rationale;
- exact frozen Stage 9 checkpoint, class label, and target layer;
- preprocessing identity with the classifier contract;
- deterministic implementation and controlled randomness;
- label-specific attention generation and output normalization;
- attribution sanity checks, including parameter/model randomization controls;
- robustness to benign image and preprocessing perturbations;
- repeatability and explanation-stability measurements;
- reproducible environment, code, configuration, and artifact hashes;
- bounded visual overlay semantics and UI accessibility rules;
- a prohibition on converting attribution absence into classifier contradiction.

Acceptance must assess whether the display faithfully represents model attribution. It must not treat visual plausibility alone as localization validity.

## Extension 2: true pathology localization

True localization is a separate research track. Grad-CAM or another attention map must not be used as lesion ground truth.

### Evidence prerequisites

Implementation requires independently governed spatial evidence, such as bounding boxes, segmentation masks, or lesion-level annotations, plus:

- pathology-compatible semantics and pathology-specific scope;
- governed patient identity and proven patient-safe splits;
- approved dataset license, source version, and acquisition record;
- exact annotation and adjudication protocol;
- governed image-to-annotation joins without heuristic identity matching;
- frozen train, validation, test, and appropriately independent evaluation cohorts;
- frozen metrics, confidence intervals, operating-point rules, and evaluation seed;
- prospective failure analysis and acceptance criteria;
- external or appropriately independent localization evidence.

Possible future studies include pathology-specific detection, pathology-specific segmentation, bounding-box localization, and segmentation-guided localization. No single localization model may be claimed to support all 14 frozen Stage 9 labels without label-by-label evidence and acceptance.

The core localization limitations remain unchanged until a new explicit localization acceptance gate succeeds.

## Extension 3: grounded LLM reporting

### Preferred architecture

```text
Frozen vision models
        ↓
Governed structured evidence
        ↓
Evidence packaging / grounding layer
        ↓
Grounded LLM
        ↓
Draft research explanation/report
        ↓
Deterministic verifier
        ↓
ACCEPT / REVISE / DEFER
        ↓
Expert review
```

The LLM should consume only explicitly governed structured evidence. Raw-image input is prohibited unless a separate future multimodal protocol authorizes it.

### Bounded future role

An evaluated grounded LLM may improve readability, produce structured expert-review drafts, explain governed evidence, summarize uncertainty and limitations, and create consistent grounded narratives.

It must not invent findings, measurements, locations, laterality, severity, temporal comparison, treatment, or diagnosis. It must not override deterministic safety decisions or verifier evidence, hide uncertainty, or convert withheld capabilities into supported claims.

### Baseline and fallback

The Stage 18 deterministic renderer remains the safe fallback, baseline comparator, and ablation baseline. Future studies must compare `DETERMINISTIC_REPORTER` with `GROUNDED_LLM_REPORTER`; the deterministic renderer must not be replaced merely because an LLM draft is more fluent.

### Prospective evaluation

Freeze the evaluation protocol before inspecting comparative results. Candidate dimensions are:

- factual grounding and evidence attribution;
- unsupported-claim and hallucination rates;
- completeness and internal consistency;
- deterministic-verifier pass and contradiction rates;
- report stability under controlled decoding;
- preservation of uncertainty and withheld limitations;
- governed expert preference if qualified expert feedback later becomes available.

Clinical usefulness must not be claimed without appropriate expert and clinical validation.

### Model-selection gate

No model is selected or downloaded by this roadmap. A future gate must assess license, offline/local execution, model size, VRAM, reproducibility, context requirements, structured-output support, medical-domain evidence, privacy, latency, and deterministic or controlled decoding. Dependency and artifact hashes must be frozen before evaluation.

## Future multimodal option

A multimodal VLM may be investigated only as a separate gated study. It must be compared prospectively against the frozen vision pipeline and must not be assumed superior. No image-to-LLM or image-to-VLM path is authorized in the core project.

## Extension 4: combined explainability and grounded presentation

An optional combined interface may eventually follow:

```text
Chest X-ray
  → classifier evidence
  → attention/localization evidence where separately validated
  → uncertainty
  → grounded LLM explanation
  → deterministic verification
  → expert review
```

If attribution exists without validated localization, both the interface and LLM must describe it only as **model attention** or **visual attribution**, never as a confirmed lesion location.

## Authorization checklist

Before any extension implementation begins, require a post-core-release authorization record, an isolated research branch, a frozen experiment contract, privacy and locked-test protections, dependency governance, bounded synthetic validation, and an explicit statement that core TrustCXR claims remain unchanged.

EXT-3 is closed as a controlled negative result: its frozen development operating-point gate
was not satisfied, so localization integration remains withheld and the locked test remains
closed. EXT-4A governance, EXT-4B evidence grounding schema, EXT-4C output contract,
and EXT-4D benchmark definition are complete; EXT-4E1 model-selection protocol is prepared;
no
LLM/provider/client has been selected or implemented. The next separately authorized work
is local development candidate execution under the EXT-4E protocol.
