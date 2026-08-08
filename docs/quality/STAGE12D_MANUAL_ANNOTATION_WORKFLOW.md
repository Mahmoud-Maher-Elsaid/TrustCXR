# Stage 12D Manual Annotation Workflow

Run the construction launcher once. It creates local-only files under `artifacts/stage12/annotation_cohort` and refuses to overwrite them. Trusted CheXpert metadata automatically populates AP, PA, LATERAL, and image-level `Support Devices` values for patient-safe train/validation records. Test-assigned records are excluded, and images are not opened.

Review `manual_review_queue.csv`. For each `EXPANDED_VIEW` row, assign only `OTHER` or `UNKNOWN` using protocol 1.0.0, record reviewer role and evidence, set `APPROVED` only after review, and append the completed row to `expanded_view_annotations.csv`. Do not infer `UNKNOWN` solely from missing metadata.

Populate `input_rejection_annotations.csv` only from an approved train/validation development cohort. Each row needs hashed record and patient keys, exactly one primary disposition, reviewer role, evidence, `APPROVED` status, and protocol `1.0.0`. At least one reviewed example is required for each rejection class. Never copy a locked-test record into these files.

Do not change `device_presence_annotations.csv` unless correcting documented source metadata. `Support Devices` is presence-only; device type and location must remain absent.

After review, rerun the Stage 12D readiness audit. It will reject missing classes, invalid provenance, non-development splits, and patient overlap.
