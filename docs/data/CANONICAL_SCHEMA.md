# Canonical Data Schema

TrustCXR uses a dataset-agnostic record that preserves the original dataset identity while exposing a common interface to downstream models.

## Core principles

- Every record must identify its source dataset and source image.
- Patient-level splitting is mandatory whenever a patient identifier can be resolved.
- Labels from different datasets must not be merged without an explicit ontology mapping.
- Source paths, patient identifiers, radiology text, and patient-level outputs remain local-only.
- Missing patient or study identifiers require a dataset-specific adapter rather than an unsafe filename guess.

## Required fields

- `dataset_id`
- `source_record_id`
- `image_id`
- `image_path`
- `license_status`

## Conditional fields

- `patient_id` is required before patient-level splitting when the source dataset exposes it.
- `study_id` is required for multi-view or longitudinal grouping when available.
- `labels`, `bounding_boxes`, and `mask_path` depend on the task and source annotations.
- `findings` and `impression` are local-only report fields.
- `split` is generated only after leakage-safe grouping rules are implemented.

## Stage 4 boundary

Stage 4 validates structure and metadata coverage. It does not create train, validation, or test splits and does not rewrite source datasets.