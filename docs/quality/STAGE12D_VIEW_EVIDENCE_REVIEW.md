# Stage 12D Metadata-First View Evidence Review

The review joins each unresolved record to its trusted CheXpert metadata and records source view fields, DICOM metadata availability, file format, dimensions, and EXIF orientation. The image may confirm that reviewable image content exists, but appearance is not used to infer projection.

`OTHER` requires positive metadata evidence for a known chest projection outside AP, PA, and LATERAL. `UNKNOWN` is recommended when reliable metadata cannot establish a standard or other known projection. Recommendations are evidence review only and do not modify the annotation CSV.

The patient-level review output remains ignored under `artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0`.
