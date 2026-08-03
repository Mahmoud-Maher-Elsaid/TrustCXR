# TrustCXR Stage 4.3 Resolution Report

Generated at UTC: `2026-08-03T13:43:17.217094+00:00`

## Result

- Status: `PASSED`
- Audit mode: `READ_ONLY`
- Datasets reviewed: `7`
- Container adapters resolved: `1`
- Dataset joins resolved: `6`
- Newly patient-level complete datasets: `1`
- Overall patient-level complete datasets: `4`
- Safely withheld datasets: `6`
- Local canonical records: `173658`
- Patient leakage violations: `0`

## Dataset outcomes

| Dataset | Status | Records | Patient rate | Split safety |
|---|---|---:|---:|---|
| nih_chexmask | CONTAINER_ADAPTER_RESOLVED | 112120 | 1.0 | PATIENT_LEVEL_COMPLETE |
| vinbigdata | ADAPTER_RESOLVED_IDENTITY_REVIEW | 18000 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED |
| indiana_reports | STUDY_JOIN_RESOLVED_PATIENT_WITHHELD | 7466 | 0.0 | WITHHELD_PATIENT_IDENTITY_UNRESOLVED |
| crd_masks | PAIRING_REVIEW_REQUIRED | 0 | 0.0 | WITHHELD_PATIENT_IDENTITY_UNRESOLVED |
| siim_pneumothorax | ADAPTER_RESOLVED_IDENTITY_REVIEW | 3205 | 0.0 | WITHHELD_IDENTITY_UNRESOLVED |
| tbx11k | RECORD_ADAPTER_RESOLVED_PATIENT_WITHHELD | 11702 | 0.0 | WITHHELD_PATIENT_IDENTITY_UNRESOLVED |
| covid_radiography | PAIRING_RESOLVED_PATIENT_WITHHELD | 21165 | 0.0 | WITHHELD_PATIENT_IDENTITY_UNRESOLVED |

## Safety decisions

- Patient-level splits are created only when a patient identifier is resolved with at least 99% coverage.
- Study identifiers and image filenames are not promoted to patient identifiers.
- Unresolved datasets remain available for adapter and join work but are withheld from patient-level training splits.
- Raw identifiers, source paths, row values, and local manifests remain excluded from Git.

## Scope

This stage resolves container formats, annotation joins, and identity evidence. It does not modify source datasets.
