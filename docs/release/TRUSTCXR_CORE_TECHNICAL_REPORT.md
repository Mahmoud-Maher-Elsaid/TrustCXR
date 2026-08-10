# TrustCXR Frozen Core Research Technical Report

## Abstract and objective

TrustCXR is a local, deterministic chest X-ray research pipeline for expert review. It combines governed view/quality assessment, a frozen fourteen-label classifier, predictive-uncertainty evidence, conservative evidence propagation, deterministic reporting and verification, and a private local review interface. It is **research use only, not a medical diagnosis, and requires expert review**.

## Data governance and experimental design

The project uses governed, patient-isolated development splits and preserves locked-test protection. Raw datasets are intentionally untracked. Dataset roles and limitations are consolidated in [FINAL_DATASET_USE_SUMMARY.md](FINAL_DATASET_USE_SUMMARY.md); frozen results are transcribed in [FINAL_METRICS_TABLE.md](FINAL_METRICS_TABLE.md). Metrics from incompatible cohorts are not superiority comparisons.

## Methods and accepted components

The frozen local flow is PNG/JPEG input, Stage 5 AP/PA/LATERAL and non-clinical technical-quality proxy, Stage 9 fourteen-label research scores, Stage 16 predictive uncertainty, governed limited evidence, Stage 17 DEFER-only logic, Stage 18 deterministic reporting, Stage 19 verification, Stage 20 deterministic decision support, FastAPI serving, and the `FROZEN_CORE_RESEARCH_REVIEW_UI`. Stage 8 supplies quality-filtered pseudo anatomy research evidence. Stage 10 is a localization research baseline with no accepted operating threshold. Stage 11 fusion cannot exceed `PARTIALLY_SUPPORTED`. Stage 13 multiview selective prediction was not accepted.

## Reporting, verification, and decision logic

Stage 18 is deterministic and grounded in structured evidence; it is not an LLM. Stage 19 preserves verified, partial, unavailable, and contradiction semantics without treating missing localization as contradiction. Stage 20 gives DEFER highest precedence. ACCEPT means acceptance of a research draft for expert review, never clinical approval.

## Serving, UI, and DICOM

The accepted architecture is one local API process and one sequential GPU execution path with at most one resident GPU model. The UI performs local real PNG/JPEG research inference without external requests or browser persistence. DICOM interoperability is accepted only for deterministic synthetic/non-patient, single-frame grayscale Explicit/Implicit VR Little Endian MONOCHROME1/2 fixtures; real-patient DICOM and browser DICOM viewing remain withheld.

## Human feedback and external validation

Stage 24 closed `WITHHELD_NOT_FAILED`: no governed expert-feedback data exists and active learning is inactive. Stage 26 closed `WITHHELD_NOT_FAILED`: `EXTERNAL_VALIDATION_NOT_PERFORMED`. No independent clinical, multi-institution, prospective, deployment, or generalizability claim is permitted.

## Reproducibility

The Windows research environment is frozen by `requirements/lock-final-research-windows-cu130.txt` and its installation protocol. Seven accepted checkpoint/config pairs have SHA-256 integrity evidence. Linux, macOS, CPU-only equivalence, other CUDA stacks, and bit-identical CUDA training are not validated.

## Limitations, ethics, and privacy

The authoritative [FINAL_CLAIMS_MATRIX.md](FINAL_CLAIMS_MATRIX.md) applies. There is no clinical diagnosis, treatment recommendation, autonomous release, reliable positive lesion localization, severity, temporal change, accepted OOD detection, or external validation. Patient identifiers, PHI, raw private DICOM metadata, credentials, and raw datasets must not be tracked or exposed.

## Future work and conclusion

Optional post-release explainability, true localization, grounded LLM, and multimodal studies are documentation-only in `docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md`; none is implemented or required for core closure. The frozen core is a reproducible research system for expert review, not clinical software.

## Citation audit

Repository dataset manifests, model cards, and stage reports are the authoritative source references for this release. Existing external bibliographic metadata must be verified against its primary source before publication; missing DOI, author, year, journal, URL, or license metadata must be marked for citation verification rather than invented.
