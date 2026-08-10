# TrustCXR: An Evidence-Governed and Uncertainty-Aware Chest X-ray Research System

## A Multi-Stage Research Pipeline for Classification, Predictive Reliability, Deterministic Evidence Verification, and Expert-Review-Oriented Reporting

**Frozen Core Research Release**  
**RESEARCH USE ONLY - NOT A MEDICAL DIAGNOSIS - EXPERT REVIEW REQUIRED**

**Author:** Mahmoud Maher El-Said  
**Affiliation:** *Affiliation intentionally left editable; no formal affiliation is governed in the repository.*

## Abstract

TrustCXR is a local research system for conservative chest X-ray analysis and expert-review-oriented reporting. The frozen core combines view and non-clinical technical-quality assessment, a fourteen-label chest radiograph classifier, model-specific predictive-uncertainty evidence, structured evidence propagation, deterministic research reporting, deterministic verification, and rule-based research-draft decision support. The system is designed around explicit provenance, patient-safe dataset governance, locked-test protection, privacy constraints, and fail-closed handling of unsupported evidence. Its accepted local workflow processes bounded PNG/JPG/JPEG inputs through a FastAPI service and a static research interface. The Stage 9 classifier's frozen internal test evidence is macro AUROC 0.7287231943, macro AUPRC 0.1540464037, and F1 at 0.5 of 0.2072496443 on 17,061 records from 4,715 patients. These values are research evidence, not clinical probabilities. Reliability is expressed as predictive uncertainty only; the system does not claim OOD detection, reliable lesion localization, severity, temporal change, or clinical generalizability. DICOM interoperability is limited to deterministic synthetic, non-patient, single-frame grayscale fixtures. External validation and governed expert-feedback acquisition were not available at core closure. TrustCXR is therefore a reproducible research pipeline for expert review, not diagnostic or autonomous clinical software.

**Keywords:** Chest X-ray; medical imaging; deep learning; computer vision; multi-label classification; predictive uncertainty; evidence verification; reproducible AI.

## 1. Introduction

Chest radiography models can produce useful research signals while still being unsafe to present as autonomous clinical conclusions. A responsible research system must preserve the distinction between a model response, validated evidence, and a clinical interpretation. TrustCXR addresses this engineering and governance problem by coupling frozen vision components with uncertainty evidence, deterministic downstream rules, provenance-aware verification, and an explicit expert-review boundary.

The project is intentionally conservative. It does not claim diagnosis, treatment recommendation, autonomous release, or clinical deployment. Its primary contribution is an auditable research workflow in which supported, limited, withheld, and prohibited claims remain visible in the evidence contract. The final core release is designated `TRUSTCXR_FROZEN_RESEARCH_RELEASE` and its interface is designated `FROZEN_CORE_RESEARCH_REVIEW_UI`.

## 2. Research Objectives and Scope

The core objectives were to:

1. assess supported AP, PA, and LATERAL view classes and a non-clinical technical-quality proxy;
2. expose a frozen fourteen-label classifier as research model signals;
3. preserve model-specific predictive uncertainty without relabeling it as epistemic or clinical certainty;
4. propagate evidence conservatively through deterministic reporting, verification, and research-draft decision logic;
5. provide bounded local FastAPI serving and a static browser UI without telemetry or browser persistence; and
6. establish reproducibility, privacy, dataset governance, and historical evidence integrity sufficient for a frozen research release.

The scope excludes treatment, diagnosis, severity, temporal comparison, OOD detection, reliable lesion localization, active learning, external validation, and language-model functionality.

## 3. Dataset Governance

Raw datasets remain local and intentionally untracked. The governed dataset catalog and final dataset-use summary distinguish development roles, patient-safe split evidence, label semantics, licensing limitations, and external-validation eligibility. NIH ChestXray14 and NIH CheXmask support the Stage 6-9 classifier and pseudo-anatomy evidence; CheXpert Small supports view and multiview experiments; RSNA supports a limited lung-opacity localization baseline and fusion characterization. Several audited candidate datasets remain withheld because identity, annotation joins, licensing, or label compatibility are unresolved.

Patient-level split protection and locked-test protection are mandatory. Dataset labels are not treated as prospective expert feedback. No raw patient images, identifiers, private DICOM metadata, or locked-test records are included in this paper or in tracked release artifacts. No new dataset was acquired for core closure.

## 4. System Architecture

The frozen pipeline is a sequential research path: local raster input, Stage 5 view and quality assessment, Stage 9 classification, Stage 16 predictive uncertainty, governed evidence, Stage 17 deterministic DEFER logic, Stage 18 deterministic reporting, Stage 19 verification, Stage 20 decision support, FastAPI serving, and the frozen research UI. At most one large GPU model is resident at a time under the governed serving contract.

![Figure 1. Frozen TrustCXR Core research architecture.](../assets/trustcxr-core-architecture.svg)

**Figure 1.** Frozen TrustCXR Core research architecture. The repository-owned vector diagram excludes future explainability, localization, language-model, and clinical pathways.

## 5. Methodology

### 5.1 View and technical-quality assessment

Stage 5 supports AP, PA, and LATERAL view assessment and a non-clinical technical-quality proxy. It does not provide a generic OTHER/UNKNOWN clinical capability. The quality signal is used as research evidence and not as a clinical acceptability determination.

### 5.2 Frozen multi-label classifier

The accepted Stage 9 classifier returns the following labels in its frozen contract:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

Each output is a research model signal. Scores are not calibrated clinical disease probabilities and are not diagnoses. The UI may sort signals for readability, but it preserves all fourteen values and their provenance.

### 5.3 Reliability evidence

Stage 16 reports validation-derived, model-specific predictive uncertainty where its contract applies. It does not claim epistemic uncertainty, clinical certainty, or OOD detection. Stage 13 selective prediction was not accepted. Uncertainty is carried into the deterministic downstream pathway without being converted into a diagnosis.

### 5.4 Evidence propagation

Stage 8 pseudo-anatomy evidence and Stage 10 localization research evidence are kept separate from classifier evidence. Stage 11 fusion has a maximum support level of `PARTIALLY_SUPPORTED`; exact governed identity is required and heuristic joins are prohibited. Missing localization cannot contradict classifier evidence, and no finding laterality is inferred from localization.

## 6. Experimental Design

Experiments used governed dataset-specific contracts, patient-safe splits where available, frozen configurations, and explicit selection/test policies. Metrics are transcribed from frozen evidence and are not recomputed for this paper. Results from incompatible cohorts are not compared as superiority claims. The locked-test protection policy prohibits post-test tuning, relabeling, feedback-driven tuning, and active-learning sampling from locked data.

## 7. Results

### 7.1 Frozen Stage 9 internal test evidence

| Metric | Value | Cohort and qualification |
|---|---:|---|
| Macro AUROC | 0.7287231943 | Frozen NIH/CheXmask shared internal test; selected on validation; no post-test tuning |
| Macro AUPRC | 0.1540464037 | Same 14-label research contract; not a clinical probability |
| F1 at 0.5 | 0.2072496443 | Frozen threshold summary; not a clinical operating point |
| Test records | 17,061 | Patient-safe frozen internal test evidence |
| Test patients | 4,715 | Patient count in the frozen evidence |

Displayed values are reproduced from `docs/release/FINAL_METRICS_TABLE.md` and the Stage 9 evidence report. No locked-test record was accessed while preparing this paper.

### 7.2 Additional frozen evidence

The final metrics index records Stage 5 view macro F1 of 0.986133 and technical-quality proxy accuracy of 0.999375 on the CheXpert Small internal test (11,231 records). Stage 8 reports pseudo-anatomy macro Dice 0.971378 and macro IoU 0.944425 on its governed internal evidence; these are not manual clinical segmentation ground truth. Stage 10 reports small-lesion sensitivity 0.036145 at score 0.5 on RSNA validation, with no accepted operating point. Stage 13 reports macro AUROC 0.846686 and macro AUPRC 0.859265 on 3,046 exact pairs under a different frontal-only contract; this must not be compared as superiority over Stage 9.

## 8. Reliability and Predictive Uncertainty

Reliability is a model-specific evidence layer rather than a universal confidence claim. Stage 16 supports predictive uncertainty wording and preserves calibration limitations. The core does not claim OOD detection, epistemic uncertainty, or clinical certainty. Where the reliability contract cannot validly operate, the system withholds the evidence rather than fabricating a value.

## 9. Evidence Fusion and Its Limitations

Fusion combines only explicitly governed evidence. Stage 11 was accepted at maximum `PARTIALLY_SUPPORTED`. In its complete-coverage characterization, 106 records were uncertain and 2 were unlocalized. Anatomy masks are pseudo-anatomy evidence, not lesion ground truth. The RSNA lung-opacity baseline cannot validate all fourteen classifier findings. No heuristic anatomical join, laterality inference, or localization-absence contradiction is permitted.

## 10. Deterministic Research Triage and Reporting

Stage 17 is DEFER-only research triage. Stage 18 is a deterministic, template-bound renderer operating on structured evidence; it is not an LLM and does not generate unrestricted clinical prose. The report preserves uncertainty, limitations, and expert-review requirements. `DEFER` is an evidence or safety limitation, not a clinical diagnosis.

## 11. Evidence Verification and Final Decision Logic

Stage 19 verifies structured textual and provenance evidence. `VERIFIED`, `PARTIALLY_VERIFIED`, `UNVERIFIED`, `CONTRADICTED`, `NOT_APPLICABLE`, and `WITHHELD_INSUFFICIENT_EVIDENCE` remain distinct internal states; missing evidence is not contradiction. Stage 20 preserves the precedence:

`DEFER > REVISE_DETERMINISTICALLY > ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW`

Acceptance means that a research draft is ready for expert review. It is never clinical approval and never authorizes autonomous release.

## 12. Local FastAPI Research Application

The local application accepts bounded PNG/JPG/JPEG inputs through the existing FastAPI architecture. Stage 5 and Stage 9 models execute sequentially under the one-GPU-model-at-a-time contract, then deterministic CPU-side processing produces uncertainty evidence, report content, verification, and decision support. The static UI is served locally at `/ui` and uses no external network resources, telemetry, cookies, localStorage, or sessionStorage. The interface remains visibly marked for research use and expert review.

## 13. DICOM Interoperability Scope

DICOM interoperability is accepted only for deterministic synthetic/non-patient, single-frame grayscale fixtures using uncompressed Explicit VR Little Endian or Implicit VR Little Endian transfer syntax and MONOCHROME1 or MONOCHROME2 photometric interpretation. Compressed DICOM, multi-frame DICOM, generic real-patient DICOM, raw metadata display, and browser DICOM viewing remain withheld. DICOM parsing/display is separate from model inference and cannot create clinical findings.

## 14. Reproducibility

The final environment lock is `requirements/lock-final-research-windows-cu130.txt`, validated for Windows 11, Python 3.12.10, PowerShell 5.1 or 7, PyTorch cu130, and the governed NVIDIA GPU environment. Seven accepted checkpoint/config pairs have SHA-256 evidence. Linux, macOS, CPU-only equivalence, alternative CUDA stacks, and bit-identical CUDA training are not validated. Seeds, split protection, configuration fingerprints, and deterministic downstream components support controlled research reproducibility without promising identical CUDA training numerics.

## 15. Scientific Limitations

The following limitations are part of the frozen release, not optional warnings:

- no clinical diagnosis, treatment recommendation, or autonomous clinical release;
- no reliable positive lesion localization; localization absence must not contradict classifier evidence;
- Stage 11 fusion is capped at `PARTIALLY_SUPPORTED`;
- Stage 13 selective prediction is not accepted;
- OOD detection, temporal change, severity, and device localization are withheld;
- Stage 17 is DEFER-only research triage;
- Stage 18 provides deterministic reporting only;
- Stage 19 verifier semantics remain restricted and evidence-aware;
- Stage 20 gives DEFER highest precedence;
- DICOM is synthetic/non-patient only within the narrow accepted contract;
- human-feedback acquisition and active learning are withheld with `WITHHELD_NOT_FAILED`;
- external validation was not performed and no clinical generalizability claim is allowed;
- reproducibility is validated only for the Windows/CUDA scope described above.

## 16. Privacy, Ethics, and Safety

TrustCXR prohibits patient identifiers, PHI, raw private DICOM metadata, credentials, and raw datasets in tracked artifacts. Local uploads are request-scoped and are not persisted by the browser or sent to external services. Locked-test data cannot be used for tuning or feedback acquisition. The safety designation remains **RESEARCH USE ONLY - NOT A MEDICAL DIAGNOSIS - EXPERT REVIEW REQUIRED**.

## 17. External Validation Status

The final disposition is `SCIENTIFICALLY_WITHHELD_NO_GOVERNED_INDEPENDENT_EXTERNAL_VALIDATION_COHORT`, classified as `WITHHELD_NOT_FAILED`. The release statement is `EXTERNAL_VALIDATION_NOT_PERFORMED`. This paper does not imply independent clinical, multi-institution, prospective, deployment, or clinical-generalizability validation. Future external validation would require a separately governed independent cohort, compatible labels, proven non-overlap, approved licensing, and frozen metrics.

## 18. Post-Release Research Extensions

The roadmap at `docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md` is classified `OPTIONAL_POST_RELEASE_RESEARCH_EXTENSION` with implementation status `NOT_STARTED`. It describes future, separately gated work only:

- Grad-CAM or class-specific visual attribution, described as model attention rather than lesion location;
- true pathology localization using independently governed spatial annotations;
- grounded LLM reporting over structured evidence, with deterministic verification;
- an optional, separately gated multimodal VLM comparison.

None of these capabilities is implemented in the frozen core and none is required for core closure.

## 19. Conclusion

TrustCXR provides a frozen, evidence-governed research pipeline that connects chest X-ray model signals with predictive uncertainty, conservative evidence handling, deterministic reporting, verification, and expert-review-oriented decision support. Its value is the explicit boundary between supported evidence and unsupported claims. The core release is reproducible within its documented Windows/CUDA scope, locally deployable for bounded raster research review, and deliberately incomplete as a clinical system. It should be used only for research and expert review, never as a medical diagnosis or autonomous clinical release.

## References

1. TrustCXR Core Technical Report. `docs/release/TRUSTCXR_CORE_TECHNICAL_REPORT.md`.
2. TrustCXR Final Claims Matrix. `docs/release/FINAL_CLAIMS_MATRIX.md`.
3. TrustCXR Final Frozen Metrics Table. `docs/release/FINAL_METRICS_TABLE.md`.
4. TrustCXR Final Dataset Use Summary. `docs/release/FINAL_DATASET_USE_SUMMARY.md`.
5. TrustCXR Post-Release Research Extension Roadmap. `docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md`.

The repository does not provide complete, verified external bibliographic metadata for additional citations. No DOI, author list, journal, license, or URL is invented here; citation verification is required before external publication.
