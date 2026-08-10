# TrustCXR Final Claims Matrix

This is the authoritative claims contract for the frozen core research release. All project-facing documentation must remain consistent with it.

**Release designation:** `TRUSTCXR_FROZEN_RESEARCH_RELEASE`

**UI designation:** `FROZEN_CORE_RESEARCH_REVIEW_UI`
**Required designation:** **RESEARCH USE ONLY · NOT A MEDICAL DIAGNOSIS · EXPERT REVIEW REQUIRED**

## SUPPORTED

| Claim | Exact accepted scope | Evidence |
|---|---|---|
| Local raster input | Bounded local PNG/JPG/JPEG decode and research inference | `src/trustcxr/serving/local_inference.py`; serving tests |
| View assessment | Model-identified AP, PA, or LATERAL | `reports/stage5/stage5_summary.json` |
| Technical quality | Non-clinical research technical-quality proxy | Stage 5 report and UI wording |
| Frozen classification | Fourteen Stage 9 research model scores in the frozen NIH label order | `reports/stage9/stage9_final_summary.json` |
| Anatomy evidence | Quality-filtered pseudo lung/heart anatomy segmentation research evidence | `reports/stage8/stage8e_summary.json` |
| Localization baseline | RSNA lung-opacity research baseline retained for research characterization | `reports/stage10/stage10n_localization_acceptance_decision_summary.json` |
| Reliability | Accepted model-specific calibration behavior and predictive uncertainty | `reports/stage16/stage16_final_capability_freeze.json` |
| Triage | Deterministic research `DEFER` disposition and reason codes only | `reports/stage17/stage17_final_research_triage_freeze.json` |
| Reporting | Deterministic, template-bound grounded research draft | `reports/stage18/stage18_final_grounded_report_capability_freeze.json` |
| Verification | Deterministic textual/provenance verification within the frozen governed scope | `reports/stage19/stage19_final_verifier_capability_freeze.json` |
| Decision support | Deterministic accept/revise/defer research-draft support with DEFER precedence | `reports/stage20/stage20_final_decision_support_freeze.json` |
| Local serving | Minimal local FastAPI research serving and bounded worker control path | `reports/stage21/stage21h_synthetic_runtime_acceptance_decision_summary.json` |
| Research UI | Local static UI plus current-image Stage 5/9 inference presentation | `FROZEN_CORE_RESEARCH_REVIEW_UI`; UI/serving tests |
| DICOM interoperability | Synthetic/non-patient, single-frame grayscale, uncompressed Explicit/Implicit VR Little Endian, MONOCHROME1/2 | `reports/stage23/stage23d_dicom_interoperability_acceptance_closure.json` |
| Reproducibility | Governed Windows 11 / Python 3.12.10 / PyTorch cu130 research environment | `reports/stage25/stage25b_mlops_reproducibility_closure.json` |

## LIMITED

| Claim | Mandatory qualification |
|---|---|
| Stage 8 anatomy | Pseudo anatomy, not manual clinical ground truth; overlays remain withheld in the core UI. |
| Stage 10 localization | Research baseline only; no accepted operating threshold, clinical localization, or reliable positive lesion localization. |
| Stage 11 fusion | Maximum `PARTIALLY_SUPPORTED`; exact governed identity required; no heuristic joins. |
| Stage 13 multiview | Frontal-only model selected; selective prediction not accepted; its cohort/labels differ from Stage 9. |
| Stage 16 uncertainty | Predictive uncertainty only, not epistemic uncertainty or clinical certainty. |
| Stage 18 report | Deterministic research draft from structured evidence; expert review required. |
| Stage 19 contradiction | Requires explicit accepted structured conflict on exact governed identity; missing evidence is not contradiction. |
| Stage 20 accept | Means research draft ready for expert review, never clinical approval. |
| Local real-image workflow | Research-only local raster inference; no patient workflow or clinical deployment validation. |
| CUDA reproducibility | Frozen environment/seeds support controlled reproducibility; bit-identical CUDA training is not guaranteed. |

## WITHHELD

- Reliable positive lesion localization and finding laterality.
- Contradiction inferred from localization absence.
- Stage 13 selective prediction.
- OOD detection or OOD claims.
- Temporal comparison/change and severity.
- Device localization, treatment, patient-history inference, diagnosis, and clinical certainty.
- Real-patient or generic clinical DICOM handling, compressed DICOM, multi-frame DICOM, and DICOM UI display.
- Stage 8 and Stage 10 UI overlays.
- Human-feedback acquisition and active learning: `SCIENTIFICALLY_WITHHELD_NO_GOVERNED_EXPERT_FEEDBACK_DATA`, `WITHHELD_NOT_FAILED`.
- External validation: `SCIENTIFICALLY_WITHHELD_NO_GOVERNED_INDEPENDENT_EXTERNAL_VALIDATION_COHORT`, `WITHHELD_NOT_FAILED`, `EXTERNAL_VALIDATION_NOT_PERFORMED`.
- Linux, macOS, CPU-only, alternative-CUDA equivalence.
- Grad-CAM/visual attribution, true pathology localization, grounded LLM reporting, and multimodal VLM functionality; these are unimplemented optional future studies.

## PROHIBITED_CLAIMS

`NO CLINICAL DIAGNOSIS`

`NO TREATMENT RECOMMENDATION`

`NO AUTONOMOUS CLINICAL RELEASE`

`NO RELIABLE POSITIVE LESION LOCALIZATION`

The release must never be described as:

- clinically validated, diagnostic, autonomous, or medical-device software;
- production clinical AI or validated for deployment;
- validated for treatment, referral, emergency, or prioritization decisions;
- independently, prospectively, multi-institutionally, or externally validated;
- clinically generalizable;
- capable of reliable lesion localization, laterality, severity, or temporal progression;
- capable of OOD detection;
- using clinician feedback or active learning;
- using an LLM, VLM, or implemented Grad-CAM extension;
- accepting a report as clinical approval or authorizing autonomous release.

## Immutable safety precedence

`DEFER > REVISE_DETERMINISTICALLY > ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW`

`LOCALIZATION_ABSENCE_MUST_NOT_CONTRADICT_CLASSIFIER_EVIDENCE`

`HUMAN_FEEDBACK_DOES_NOT_AUTOMATICALLY_BECOME_TRAINING_TRUTH`
