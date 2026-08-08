# Stage 12C Annotation and Device-Scope Protocol

Protocol version: `1.0.0`. Approval basis: explicit project-owner Stage 12C mandate.

## View annotation

Exactly one view class is assigned to a valid chest radiograph: `AP`, `PA`, `LATERAL`, `OTHER`, or `UNKNOWN`. `OTHER` requires a known projection outside the three supported classes. `UNKNOWN` means reliable evidence cannot determine the projection; missing metadata alone is not sufficient when another reliable source exists.

## Input disposition

Exactly one primary disposition is assigned using this precedence: `CORRUPT_INPUT`, `UNSUPPORTED_FORMAT`, `NON_CHEST_INPUT`, `INCOMPLETE_ANATOMY`, `INADEQUATE_QUALITY`, `UNSUPPORTED_VIEW`, then `ACCEPTABLE_INPUT`. Secondary flags may preserve additional defects but cannot replace the primary reason.

Corrupt input cannot be decoded reliably. Unsupported format is readable but outside the ingestion contract. Non-chest input requires explicit evidence or qualified review. Incomplete anatomy is used when missing thoracic coverage is decisive. Inadequate quality covers task-preventing degradation when anatomy coverage is not the decisive issue. Unsupported view requires a reliably known view outside the downstream model contract. Acceptable input means only that no rejection criterion applies; it does not imply diagnostic normality.

## Device scope

CheXpert `Support Devices` is authorized only for image-level device presence. It does not identify device type or location. Device localization and a new device-localization dataset are outside this stage.

No annotations, training, inference, or locked-test access occur in Stage 12C. Frozen Stage 5, 9, 10, and 11 evidence remains unchanged.
