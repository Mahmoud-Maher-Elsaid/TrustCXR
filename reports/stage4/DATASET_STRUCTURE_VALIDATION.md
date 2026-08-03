# TrustCXR Dataset Structure and Canonical Schema Validation

Generated at UTC: `2026-08-03T11:49:26.247583+00:00`

## Result

- Execution status: `PASSED`
- Dataset readiness: `PARTIAL_MAPPING_READINESS`
- Datasets validated: `10`
- Ready for mapping: `9`
- Datasets requiring extraction: `0`
- Datasets requiring patient mapping: `8`

## Dataset results

| Dataset | Structure status | Images | Metadata | Archives | Patient split |
|---|---|---:|---:|---:|---|
| NIH ChestXray14 | READY_FOR_CANONICAL_MAPPING | 112120 | 4 | 0 | DIRECT_PATIENT_LEVEL_SPLIT |
| VinBigData Chest X-ray Abnormalities Detection | READY_FOR_CANONICAL_MAPPING | 18000 | 2 | 1 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| Indiana University Chest X-ray Reports | READY_FOR_CANONICAL_MAPPING | 7470 | 2 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| NIH CheXmask | NO_IMAGES_FOUND | 0 | 1 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| Chest Radiography Database Lung Masks | READY_FOR_CANONICAL_MAPPING | 13244 | 1 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| RSNA Pneumonia Detection Challenge | READY_FOR_CANONICAL_MAPPING | 29684 | 4 | 0 | DIRECT_PATIENT_LEVEL_SPLIT |
| CheXpert Small | READY_FOR_CANONICAL_MAPPING | 223649 | 2 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| SIIM-ACR Pneumothorax Segmentation | READY_FOR_CANONICAL_MAPPING | 3205 | 2 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| TBX11K | READY_FOR_CANONICAL_MAPPING | 11702 | 1 | 0 | REQUIRES_DATASET_SPECIFIC_MAPPING |
| COVID-19 Radiography Database | READY_FOR_CANONICAL_MAPPING | 42330 | 1 | 1 | REQUIRES_DATASET_SPECIFIC_MAPPING |

## Canonical schema policy

- Every canonical record must identify its source dataset and image.
- Patient-level splitting is mandatory whenever a patient identifier can be resolved.
- Dataset-specific labels remain separate until an explicit ontology mapping is approved.
- Image paths, patient identifiers, report text, and row values remain local-only.
- Committed profiles contain column names and aggregate counts only.

## Interpretation

A dataset marked `READY_FOR_CANONICAL_MAPPING` still requires a dedicated adapter.
A dataset marked `NEEDS_EXTRACTION` is present but cannot be mapped until its archive is extracted.
Patient split readiness reports whether a patient identifier was discovered automatically; it does not create splits yet.
