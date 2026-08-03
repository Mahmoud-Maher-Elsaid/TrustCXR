# Concrete Dataset Adapters and Safe Splits

## Purpose

Stage 4.2 converts each supported local dataset into a privacy-safe canonical
manifest while leaving the source data unchanged.

## Local outputs

Canonical manifests and detailed issue files are written under:

`reports/stage4_2/local/`

This directory is excluded from Git because it contains local relative paths and
record-level derived information.

## Committed outputs

Only aggregate counts, safety states, adapter configuration, tests, and
implementation code are committed.

## Identity rules

A record receives a patient-level split only when a patient identity is resolved
from an approved source:

- Explicit metadata patient column
- CheXpert patient path component
- DICOM PatientID

Records without a verified patient identity receive:

`unassigned_identity_review`

They are not placed into train, validation, or test.

## Split policy

Verified patient identities are hashed deterministically and assigned to:

- Train: 80%
- Validation: 10%
- Test: 10%

The same patient always receives the same split. Raw patient identifiers are
never written to the canonical manifest or committed reports.

## Adapter states

- `PATIENT_LEVEL_COMPLETE`: at least 99% of records have a patient identity.
- `PATIENT_LEVEL_PARTIAL`: some records have verified patient identity.
- `WITHHELD_IDENTITY_UNRESOLVED`: patient-level splits remain blocked.
- `CONTAINER_ADAPTER_PENDING`: the dataset is stored in a container or table
  that requires a separate cross-dataset adapter.

## Safety guarantees

- Source dataset files are read only.
- Pixel data is not decoded for DICOM identity extraction.
- No random image-level split is presented as a patient-level split.
- Detailed local paths and record-level joins remain excluded from Git.
- The stage fails if any patient appears in more than one split.