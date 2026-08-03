# Dataset Adapter Contracts

Stage 4.1 discovers privacy-safe identifier sources and generates dataset adapter
contracts without modifying source data.

## Contract responsibilities

Each dataset contract defines:

- Dataset identifier and local folder
- Patient identifier source
- Study identifier source
- Image identifier source
- Confidence level for each source
- Patient-level split readiness
- Adapter implementation status

## Supported identity sources

- Metadata column
- DICOM tag
- Filename regular expression
- Path component
- Relative image path
- Unresolved source requiring review

## Privacy

Raw patient identifiers are never committed. Adapter resolution converts source
identifiers into deterministic dataset-scoped hashes before canonical manifests
are produced.

## Safety gate

A dataset with an unresolved patient identifier must not use random image-level
splitting. Stage 4.2 will apply these contracts and validate actual joins before
creating final patient-level train, validation, and test splits.