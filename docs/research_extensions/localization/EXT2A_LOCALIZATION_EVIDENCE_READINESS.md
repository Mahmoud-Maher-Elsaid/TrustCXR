# EXT-2A Localization Evidence Readiness

**Track:** True pathology localization research extension
**Branch:** `research-extension/pathology-localization`
**Status:** `PASSED_FOR_EXT2B_RSNA_LUNG_OPACITY_CONTRACT`
**Core status:** unchanged; Stage 10N remains historical evidence only

## Decision

RSNA Pneumonia Detection Challenge data is selected as the first EXT-2
candidate, with the exact semantic scope **RSNA Lung Opacity**. The annotation
is a bounding box for the challenge's `Lung Opacity` target. It is not silently
renamed to `Pneumonia`, and no claim is made that it is equivalent to any of
the fourteen Stage 9 labels.

The candidate passes the evidence-readiness gate for a new validation-only
research study because the repository evidence records a complete deterministic
image/annotation join, verified patient and study identity, and a patient-safe
split with zero observed leakage. The existing Stage 10 license decision is
`APPROVED_FOR_RESEARCH` with the RSNA attribution source recorded in
`configs/localization/stage10c_governance_adjudication.json`.

This is readiness for a new contract, not evidence that localization is
accepted. No training, model selection, threshold selection, or locked-test
evaluation was performed by EXT-2A.

## Candidate audit

| Dataset | Semantics / annotation | Labeled records | Positive images | Lesions / boxes | Identity and join | Source/license | Split feasibility | EXT-2 disposition |
|---|---|---:|---:|---:|---|---|---|---|
| RSNA Pneumonia Detection Challenge | `Lung Opacity`, bounding boxes | 30,227 rows; 26,684 images | 6,012 | 9,555 boxes | Patient and study identity rate 1.0; 26,684/26,684 joins | RSNA challenge source recorded; research approval recorded, attribution required | Existing patient split: 21,356 train / 2,660 validation / 2,668 locked test; 0 leakage | **Selected; proceed to EXT-2B** |
| VinBigData | Abnormality bounding boxes | 67,914 rows in readiness matrix | Not admitted | Not admitted | Patient identity unresolved; DICOM/annotation mapping unresolved | License requires further governed review | Not eligible | Withheld |
| SIIM-ACR Pneumothorax | Pixel RLE masks | 12,954 rows | Not admitted | Not admitted | Identity unresolved; observed annotation-image matches 0 | License/source recorded but identity gate fails | Not eligible | Withheld |
| TBX11K | Serialized bounding boxes | 8,811 rows | Not admitted | Not admitted | Filename-only identity; no verified patient field | Source recorded; identity gate fails | Not eligible | Withheld |
| CRD Masks | Anatomy masks, not lesion ground truth | 3,311 rows | Not applicable | Not lesion annotations | Source-dependent identity unresolved | Terms pending manual review | Not eligible | Excluded from pathology localization |

Counts for the RSNA row were read from the local governed label CSV and the
existing Stage 10 reports. No raw image pixels were copied, persisted, or
added to Git. The locked test split was not opened.

## Blocking conditions for other candidates

The withheld datasets cannot be used as EXT-2 ground truth without new
governance work. Anatomy masks are explicitly not pathology labels. Identity
resolution and deterministic joins are prerequisites, not optional metadata.

## Gate result

`EXT2A_EVIDENCE_READINESS = PASS_FOR_RSNA_LUNG_OPACITY`

The next gate is EXT-2B: freeze the scientific contract before any training.
