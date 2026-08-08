# Stage 12D Input-Rejection Review

The evidence-approved view records are separate from input-rejection review. The remaining local file is `02_input_rejection_review.csv` in the manual package.

For each train and validation row, select one genuine example matching the prefilled rejection class. Enter a local image path or stable identifier, a group identifier that keeps related records in one split, reviewer identity, exact evidence, `APPROVED`, and protocol `1.0.0`. Do not use a locked-test record. Do not assign a class merely to fill a row; if no defensible example exists, leave it incomplete and report the missing class.

Run the existing manual annotation validator before applying the review. The prepared application tool converts only a fully valid review into the local manifest and refuses to overwrite existing annotations.
