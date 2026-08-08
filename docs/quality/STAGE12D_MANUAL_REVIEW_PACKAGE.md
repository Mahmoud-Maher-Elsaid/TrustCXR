# Stage 12D Manual Review Package

The local-only package is stored at `artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0`.

Open `01_unresolved_view_review.csv` first. Review each of its 17 images and enter `OTHER` or `UNKNOWN`, reviewer name or identifier, evidence, and `APPROVED`. Keep protocol version `1.0.0`. Do not infer `UNKNOWN` merely because metadata is missing.

Then open `02_input_rejection_review.csv`. It has one row for each of the six rejection classes in both train and validation. Select a genuine example for every row, enter its local path or identifier and a stable group identifier, and complete reviewer, evidence, and approval fields. The same group identifier cannot occur in both splits. Do not use locked-test records.

Run `scripts/quality/validate_stage12d_manual_annotations.ps1` after both files are complete. Validation does not train or perform inference.

The metadata-first evidence review was owner-approved for all 17 unresolved view records as `UNKNOWN` under protocol `1.0.0`. No record was assigned `OTHER`. The remaining manual task is the input-rejection review described in `STAGE12D_INPUT_REJECTION_REVIEW.md`.
