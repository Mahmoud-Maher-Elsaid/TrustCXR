# Container and Identity Resolution

## Purpose

Stage 4.3 resolves dataset container formats, annotation joins, and identity
evidence without modifying source data.

## Identity rules

- A verified patient identifier may be used for an 80/10/10 patient-level split.
- A study identifier may group views from one study but is not treated as a
  patient identifier.
- An image identifier identifies one record only.
- Unresolved identities remain withheld from patient-level training splits.
- Patient coverage must reach at least 99 percent before a dataset is marked
  patient-level complete.
- Patient leakage across train, validation, and test is not allowed.

## Container rules

- Tabular CSV, TSV, Parquet, Arrow, ZIP-table, and gzip-table containers are
  streamed without loading the complete file into memory.
- HDF5, NPZ, and SQLite containers are profiled without loading medical arrays.
- NIH CheXmask annotations are joined to NIH ChestXray14 images and patient
  metadata when compatible image identifiers are available.
- SIIM annotations are joined to DICOM images using image identifiers while
  DICOM headers are read without decoding pixel data.

## Privacy

Raw identifiers, source paths, annotation row values, and canonical local
manifests remain under `reports/stage4_3/local/` and are excluded from Git.
Committed reports contain aggregate counts and schema information only.

## Safety

Datasets with unresolved patient identity are not assigned image-level random
splits. They remain available for adapter development and evaluation planning
but are withheld from patient-level training workflows.