# Stage 10H Validation Operating-Point Audit

- Status: `FINALIZED_VALIDATION_OPERATING_POINT_AUDIT`
- Training performed: `false`
- Final test images accessed: `0`
- Operating point selected: `false`

## Sensitivity versus false-positive trade-off

At threshold 0.5, sensitivity is `0.492063`, small-lesion sensitivity is `0.036145`, precision is `0.321577`, and false positives per image are `0.368797`.

At threshold 0.1, sensitivity rises to `0.815873` and small-lesion sensitivity to `0.349398`, but precision falls to `0.091092` and false positives per image rise to `2.892105`.

Lower thresholds recover more lesions, especially small lesions, at the cost of a large false-positive burden. Stage 10H does not select a threshold, claim clinical utility, or authorize final-test access.
