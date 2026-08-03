# TrustCXR Stage 4.2 Adapter and Split Report

Generated at UTC: `2026-08-03T13:10:56.388601+00:00`

## Overall result

- Status: `PASSED`
- Datasets processed: `10`
- Local records: `430412`
- Patient-level complete: `3`
- Identity review required: `5`
- Leakage violations: `0`

## Dataset results

| Dataset | Status | Records | Patient rate | Split safety | Orphan images |
|---|---:|---:|---:|---:|---:|
| NIH ChestXray14 | MANIFEST_BUILT | 112120 | 1.0 | PATIENT_LEVEL_COMPLETE | 0 |
| VinBigData Chest X-ray Abnormalities Detection | MANIFEST_BUILT | 18000 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED | 0 |
| Indiana University Chest X-ray Reports | MANIFEST_BUILT | 7470 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED | 4 |
| NIH CheXmask | CONTAINER_ADAPTER_PENDING | 0 | 0.0 | NOT_CREATED | 0 |
| Chest Radiography Database Lung Masks | MANIFEST_BUILT | 6622 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED | 6622 |
| RSNA Pneumonia Detection Challenge | MANIFEST_BUILT | 29684 | 1.0 | PATIENT_LEVEL_COMPLETE | 0 |
| CheXpert Small | MANIFEST_BUILT | 223649 | 1.0 | PATIENT_LEVEL_COMPLETE | 0 |
| SIIM-ACR Pneumothorax Segmentation | CONTAINER_ADAPTER_PENDING | 0 | 0.0 | NOT_CREATED | 0 |
| TBX11K | MANIFEST_BUILT | 11702 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED | 11702 |
| COVID-19 Radiography Database | MANIFEST_BUILT | 21165 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED | 0 |

## Safety guarantees

- Source datasets were read only.
- Local manifests contain hashed identities and remain Git-ignored.
- Raw patient identifiers are never written to committed reports.
- Records without verified patient identity remain unassigned.
- No random image-level split is used as a patient-level substitute.

## Interpretation

`PATIENT_LEVEL_COMPLETE` means at least 99% of records resolved to a patient identity and no split leakage was detected.
`WITHHELD_IDENTITY_UNRESOLVED` means records were mapped, but training splits remain blocked pending identity review.
