# Stage 12D Annotation Cohort Readiness

Stage 12D checks local-only development annotation manifests against protocol `1.0.0`. It does not create annotations, read images, train models, or access locked test records.

Three manifests are required under `artifacts/stage12/annotation_cohort`: expanded-view annotations, primary input-disposition annotations, and CheXpert image-level device-presence annotations. Each record must use hashed record and patient keys, identify only a train or validation split, record protocol provenance, and use an allowed label. Patient overlap between development splits is prohibited.

The expanded-view and rejection manifests require reviewed source evidence and adjudication status. Device annotations remain limited to CheXpert `Support Devices`; device type and location are prohibited. Missing labels are not negatives.

If any manifest is missing or invalid, Stage 12D must hold for reviewed development annotations. It cannot promote an empty directory or schema-only file into a training-ready cohort.
